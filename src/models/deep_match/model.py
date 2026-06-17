"""deep_match -- lag-robust deep match-mismatch source identification.

Source identification framed as "which candidate audio envelope best matches the
EEG". Every prior attempt at this on the corpus (classical CCA aad_v2/v3, our
recon_mm) hit chance -- but they all share two weaknesses this model removes:

  1. **Fixed-lag matching.** They align EEG and envelope at a single (or
     ridge-integrated) lag. This corpus has per-trial audio<->EEG jitter (software
     timestamps; peak lags scattered std ~600 ms -- see diag_lag_jitter.py), so a
     fixed lag cannot match. deep_match scores each candidate by its **best lag**
     (a differentiable cross-correlation over a lag window, soft-max-pooled), so a
     per-trial offset no longer breaks the match.
  2. **Bad EEG normalisation.** Per-channel z-scoring (recon_mm) erases the
     relative channel amplitudes a spatial filter needs. deep_match takes RAW
     (re-referenced) EEG and whitens it with an in-model BatchNorm + a learned
     **spatial filter** (a differentiable CSP) before temporal filtering -- and runs
     at full time resolution (no 8x pooling bottleneck).

EEG -> K spatial-temporal components (neural envelope estimates); each candidate
envelope -> K matched components; score = soft-max-over-lags cross-correlation,
averaged over components; cross-entropy over the 6 candidates (5/6 masked). With
table-power-equalised candidates the match is purely temporal -> EEG-honest.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel
from ..factory import MODEL_REGISTRY


def _znorm_time(x):                       # zero-mean unit-norm over the last (time) axis
    x = x - x.mean(-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


class _EEGComp(nn.Module):
    """Raw EEG -> K spatial-temporal components (learned CSP + temporal filter)."""

    def __init__(self, n_chans, K, ksize=17, dropout=0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_chans)                 # whiten channels (replaces per-chan z-score)
        self.spatial = nn.Conv1d(n_chans, K, 1)            # learned spatial filters ~ CSP
        self.temporal = nn.Conv1d(K, K, ksize, padding=ksize // 2, groups=K)
        self.bn1 = nn.BatchNorm1d(K)
        self.drop = nn.Dropout(dropout)

    def forward(self, eeg):                                # (B,C,T)
        h = self.spatial(self.bn0(eeg))
        h = self.bn1(self.temporal(h))
        return self.drop(h)                                # (B,K,T)


class _EnvComp(nn.Module):
    """Candidate envelope -> K components matched to the EEG components."""

    def __init__(self, n_bands, K, ksize=17, dropout=0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_bands)
        self.net = nn.Conv1d(n_bands, K, ksize, padding=ksize // 2)
        self.bn1 = nn.BatchNorm1d(K)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # (B*S, n_bands, T)
        return self.drop(self.bn1(self.net(self.bn0(x))))  # (B*S,K,T)


class _DeepMatch(nn.Module):
    def __init__(self, fd, K=8, ksize=17, max_lag=32, lag_temp=0.5, dropout=0.2):
        super().__init__()
        self.eeg = _EEGComp(fd["n_chans"], K, ksize, dropout)
        self.env = _EnvComp(fd["n_bands"], K, ksize, dropout)
        self.K = K
        self.lags = list(range(-max_lag, max_lag + 1))
        self.lag_temp = lag_temp
        self.scale = nn.Parameter(torch.tensor(5.0))

    def _lag_match(self, E, V):
        """E (B,K,T), V (B,S,K,T) -> (B,S): soft-max-over-lags x-corr, mean over K."""
        E = _znorm_time(E); V = _znorm_time(V)
        B, S, K, T = V.shape
        Eu = E.unsqueeze(1)                                # (B,1,K,T)
        per_lag = []
        for lag in self.lags:
            if lag > 0:
                c = (Eu[..., lag:] * V[..., :T - lag]).sum(-1)
            elif lag < 0:
                c = (Eu[..., :lag] * V[..., -lag:]).sum(-1)
            else:
                c = (Eu * V).sum(-1)
            per_lag.append(c)                              # (B,S,K)
        st = torch.stack(per_lag, -1)                      # (B,S,K,nlags)
        soft = self.lag_temp * torch.logsumexp(st / self.lag_temp, dim=-1)  # ~max over lags
        return soft.mean(-1)                               # (B,S) avg over components

    def forward(self, eeg, cand):
        B, S = cand.shape[0], cand.shape[1]
        E = self.eeg(eeg)                                  # (B,K,T)
        V = self.env(cand.reshape(B * S, cand.shape[2], cand.shape[3]))
        V = V.reshape(B, S, self.K, V.shape[-1])
        return self._lag_match(E, V) * self.scale          # (B,S)


@MODEL_REGISTRY.register("deep_match")
class DeepMatchModel(TorchModel):
    name = "deep_match"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.K = int(cfg.get("K", 8))
        self.ksize = int(cfg.get("ksize", 17))
        self.max_lag = int(cfg.get("max_lag", 32))      # samples (32 @ 64 Hz = 500 ms)
        self.lag_temp = float(cfg.get("lag_temp", 0.5))
        self.dropout = float(cfg.get("dropout", 0.2))

    def build_module(self):
        return _DeepMatch(self.fd, self.K, self.ksize, self.max_lag,
                          self.lag_temp, self.dropout)

    def compute_loss(self, batch):
        logits = self.module(batch["eeg"], batch["cand_env"])
        att = batch["attended"]
        ce = F.cross_entropy(logits, att)
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return ce, {"ce": float(ce), "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        return self.module(batch["eeg"], batch["cand_env"])
