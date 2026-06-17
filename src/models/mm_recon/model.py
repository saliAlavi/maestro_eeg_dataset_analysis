"""mm_recon -- EEG-only envelope match-mismatch decoder (reconstruction + multi-route).

THE task (corrected framing): given a window of EEG, identify WHICH of the
simultaneously-playing talkers the subject is attending to, by matching the EEG to
the attended talker's speech *envelope*. Candidates are the 4 real talkers'
envelopes in PERMUTED order (data=aad_mm_match), so the decoder cannot cheat by
decoding attended DIRECTION -- it must match envelope CONTENT. This makes the
EEG-alone accuracy an honest measure of cortical envelope tracking.

Two corpus-specific obstacles and how this model handles them:
  * **Software-timestamp audio<->EEG jitter (~600 ms std).** Every route matches at
    the BEST lag via a differentiable soft-max-over-lags cross-correlation, so a
    per-trial offset no longer breaks the match.
  * **Down-sampled envelopes are mutually similar** -> a single correlation barely
    separates them. So we score with MULTIPLE ROUTES and train with MULTIPLE LOSSES:
      route R (recon)  : reconstruct the attended envelope from EEG, match it to each
                         candidate (per-component lag-robust x-corr).
      route D (direct) : match EEG spatio-temporal components directly to candidates
                         (the deep_match path), no reconstruction bottleneck.
      route L (learned): an MLP over the per-component match profiles of R and D ->
                         a learned, discriminative combination.
    Losses: CE on the fused logits + deep-supervision CE on routes R and D + an
    envelope reconstruction loss (MSE + (1-corr)) that anchors route R.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..deep_match.model import _EEGComp, _EnvComp, _znorm_time
from ..factory import MODEL_REGISTRY


class _Reconstructor(nn.Module):
    """Raw EEG -> reconstructed attended band-envelope (B, n_bands, W), full res."""

    def __init__(self, n_chans, n_bands, d=64, dropout=0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_chans)
        self.spatial = nn.Conv1d(n_chans, d, 1)
        self.bn1 = nn.BatchNorm1d(d)
        self.temporal = nn.Sequential(
            nn.Conv1d(d, d, 9, padding=4), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(d, d, 9, padding=4), nn.ELU(),
        )
        self.head = nn.Conv1d(d, n_bands, 1)

    def forward(self, eeg):
        h = F.elu(self.bn1(self.spatial(self.bn0(eeg))))
        return self.head(self.temporal(h))                 # (B, n_bands, W)


def _lag_corr_full(E, V, lags):
    """E (B,K,T), V (B,S,K,T) -> (B,S,nlags): per-lag x-corr, averaged over components.

    No pooling over lags here -- the caller decides (joint shared-lag softmax), so a
    candidate cannot independently maximise over its own lag (which lets distractors
    cheat with spurious peaks over a wide window).
    """
    E = _znorm_time(E); V = _znorm_time(V)
    T = V.shape[-1]
    Eu = E.unsqueeze(1)                                    # (B,1,K,T)
    per_lag = []
    for lag in lags:
        if lag > 0:
            c = (Eu[..., lag:] * V[..., :T - lag]).sum(-1)
        elif lag < 0:
            c = (Eu[..., :lag] * V[..., -lag:]).sum(-1)
        else:
            c = (Eu * V).sum(-1)
        per_lag.append(c.mean(-1))                         # (B,S) mean over K components
    return torch.stack(per_lag, -1)                        # (B,S,nlags)


class _MMRecon(nn.Module):
    def __init__(self, fd, K=16, ksize=17, lag_min=-32, lag_max=160, lag_temp=0.5,
                 d_recon=64, dropout=0.2):
        super().__init__()
        nb = fd["n_bands"]
        self.recon = _Reconstructor(fd["n_chans"], nb, d_recon, dropout)
        self.env = _EnvComp(nb, K, ksize, dropout)         # shared: candidates & recon
        self.eeg = _EEGComp(fd["n_chans"], K, ksize, dropout)   # direct route
        self.K = K
        # asymmetric lag window: the audio<->EEG offset on this corpus is ~+1 s with
        # per-trial jitter (oracle attended lags span +100..+2000 ms), so we search
        # roughly -500..+2500 ms, NOT a small symmetric window.
        self.lags = list(range(lag_min, lag_max + 1))
        self.lag_temp = lag_temp
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.route_w = nn.Parameter(torch.zeros(2))        # recon/direct fusion (softplus)

    @staticmethod
    def _joint_logits(corr, scale):
        """corr (B,S,nlags) -> per-candidate logit via JOINT softmax over (S,lag).

        A single softmax over all (candidate, lag) pairs: only the globally-best
        match wins, so a shared lag is enforced and distractors can't cheat at their
        own lags. Marginalise lags -> per-candidate log-prob (used as logits).
        """
        B, S, L = corr.shape
        flat = (corr * scale).reshape(B, S * L)
        lp = F.log_softmax(flat, dim=-1).reshape(B, S, L)  # joint log-prob
        return torch.logsumexp(lp, dim=-1)                 # (B,S) marginal log-prob

    def forward(self, eeg, cand):
        B, S, nb, W = cand.shape
        env_hat = self.recon(eeg)                          # (B, nb, W)
        V = self.env(cand.reshape(B * S, nb, W)).reshape(B, S, self.K, -1)
        E_r = self.env(env_hat)                            # recon -> comps (shared)
        E_d = self.eeg(eeg)                                # direct EEG -> comps
        corr_r = _lag_corr_full(E_r, V, self.lags)         # (B,S,nlags)
        corr_d = _lag_corr_full(E_d, V, self.lags)         # (B,S,nlags)
        w = F.softplus(self.route_w)
        corr = w[0] * corr_r + w[1] * corr_d               # shared lag axis
        logits = self._joint_logits(corr, self.scale)      # (B,S) fused
        s_r = self._joint_logits(corr_r, self.scale)       # route deep-supervision
        s_d = self._joint_logits(corr_d, self.scale)
        return logits, s_r, s_d, env_hat

    def recon_corr(self, env_hat, att_env):
        """Lag-robust correlation of the reconstruction with the attended envelope.

        The nominal window alignment is off by the ~1 s acquisition offset, so a
        pointwise MSE target would fight the physiology. Instead reward env_hat for
        matching the attended envelope at its BEST lag (broadband, soft-max-over-lags)
        -> a lag-invariant reconstruction objective.
        """
        a = _znorm_time(env_hat.mean(1))                   # (B,T) broadband, unit-norm
        b = _znorm_time(att_env.mean(1))                   # (B,T)
        T = a.shape[-1]
        per_lag = []
        for lag in self.lags:
            if lag > 0:
                c = (a[:, lag:] * b[:, :T - lag]).sum(-1)
            elif lag < 0:
                c = (a[:, :lag] * b[:, -lag:]).sum(-1)
            else:
                c = (a * b).sum(-1)
            per_lag.append(c)
        return torch.stack(per_lag, -1).max(-1).values     # (B,) TRUE best-lag corr


@MODEL_REGISTRY.register("mm_recon")
class MMReconModel(TorchModel):
    name = "mm_recon"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.K = int(cfg.get("K", 16))
        self.ksize = int(cfg.get("ksize", 17))
        self.lag_min = int(cfg.get("lag_min", -32))        # -500 ms @ 64 Hz
        self.lag_max = int(cfg.get("lag_max", 160))        # +2500 ms (covers the ~+1 s offset + jitter)
        self.lag_temp = float(cfg.get("lag_temp", 0.5))
        self.d_recon = int(cfg.get("d_recon", 64))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.w_route = float(cfg.get("w_route", 0.3))      # deep-supervision weight per route
        self.w_recon = float(cfg.get("w_recon", 0.5))      # envelope reconstruction weight

    def build_module(self):
        return _MMRecon(self.fd, self.K, self.ksize, self.lag_min, self.lag_max,
                        self.lag_temp, self.d_recon, self.dropout)

    def compute_loss(self, batch):
        cand = batch["cand_env"]
        att = batch["attended"]
        logits, s_r, s_d, env_hat = self.module(batch["eeg"], cand)
        ce = F.cross_entropy(logits, att)
        ce_r = F.cross_entropy(s_r, att)
        ce_d = F.cross_entropy(s_d, att)
        att_env = cand[torch.arange(att.shape[0], device=att.device), att]   # (B,nb,W)
        recon = (1.0 - self.module.recon_corr(env_hat, att_env).mean())      # lag-robust
        loss = ce + self.w_route * (ce_r + ce_d) + self.w_recon * recon
        with torch.no_grad():
            acc = (logits.argmax(1) == att).float().mean()
        return loss, {"ce": float(ce.detach()), "ce_r": float(ce_r.detach()),
                      "ce_d": float(ce_d.detach()), "recon": float(recon.detach()),
                      "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        logits, _, _, _ = self.module(batch["eeg"], batch["cand_env"])
        return logits

    def evaluate(self, view, ctx, prefix="test/", present_override=None):
        fd = getattr(self, "fd", {}) or {}
        pred = self.predict(view, ctx, present_override=present_override)
        true = view.as_numpy()["attended"]
        # permuted task: candidates are talkers in random order, so hemisphere/IO
        # collapses are meaningless -> report plain 4-way accuracy only.
        return compute_aad_metrics(pred, true, prefix=prefix, task_type="match",
                                   n_cand=fd.get("n_candidates", 4))
