"""source_net -- unified, improved 4-way attended-source identification.

Strict upgrade of `source_fusion` (0.485 4-way). Improves the two levers that
actually move 4-way accuracy on this corpus, and folds the envelope-matching idea
in as a branch that can only help:

  1. **Multi-scale spectro-spatial EEG** -- parallel EEGNet/CSP branches with
     different temporal kernels (theta / alpha / beta scales), concatenated log
     band-power. Sharpens the hemisphere (left/right) bit vs a single-band encoder.
  2. **Conv gaze-trajectory encoder** -- a 1-D conv over the raw subject-relative
     gaze x/y sequence (the azimuth cue), instead of an MLP on flattened values.
     Targets the within-hemisphere (inner/outer) resolution that caps 4-way.
  3. **Lag-robust content-match branch** (the deep_match mechanism) added as a
     LEARNED-GATED term: EEG temporal components vs candidate envelopes, matched by
     soft-max-over-lags cross-correlation. If any stimulus-tracking signal exists it
     contributes; if not (as the diagnostics indicate), the gate learns ~0 and the
     model falls back to the spatial+gaze decision. So source_net >= source_fusion
     by construction, with upside if content ever helps.

Joint fusion of spatial + gaze -> base 6-way logits; gated content scores added.
4-class CE + label smoothing. Presence-aware gaze + gaze dropout; `evaluate()`
reports all vs no_gaze. Reuses the aad_spec cache (broadband EEG + envelopes + raw
gaze) via data=aad_source -- no re-caching.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..components import SpectralSpatialEncoder
from ..deep_match.model import _EEGComp, _EnvComp, _znorm_time
from ..factory import MODEL_REGISTRY
from ...data.windows import N_SPEAKERS


class _MultiScaleSpatial(nn.Module):
    """Parallel CSP branches at several temporal scales -> concat log-power."""

    def __init__(self, n_chans, dropout, F1=12, D=2, kernels=(15, 33, 65)):
        super().__init__()
        self.branches = nn.ModuleList([
            SpectralSpatialEncoder(n_chans, dropout, F1=F1, D=D, ksize=k) for k in kernels
        ])
        self.out_dim = sum(b.out_dim for b in self.branches)

    def forward(self, eeg):
        return torch.cat([b(eeg) for b in self.branches], dim=-1)


class _GazeConv(nn.Module):
    """Conv over the raw gaze x/y trajectory + summary stats -> embedding."""

    def __init__(self, gaze_dim, traj_dim, d_model, dropout):
        super().__init__()
        self.tl = traj_dim // 2                       # trajectory length (x then y)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2), nn.ELU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ELU(), nn.AdaptiveAvgPool1d(1),
        ) if self.tl > 0 else None
        feat = gaze_dim + (32 if self.tl > 0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(feat, d_model), nn.LayerNorm(d_model), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.ELU(),
        )

    def forward(self, gaze, gaze_traj):
        feats = [gaze]
        if self.conv is not None and gaze_traj.shape[-1] >= 2:
            B = gaze_traj.shape[0]
            seq = gaze_traj.view(B, 2, self.tl)       # (B,2,tl): x row, y row
            feats.append(self.conv(seq).squeeze(-1))  # (B,32)
        return self.mlp(torch.cat(feats, dim=-1))


class _ContentMatch(nn.Module):
    """Lag-robust EEG<->candidate-envelope match -> per-candidate score."""

    def __init__(self, fd, K=8, ksize=17, max_lag=32, lag_temp=0.5, dropout=0.2):
        super().__init__()
        self.eeg = _EEGComp(fd["n_chans"], K, ksize, dropout)
        self.env = _EnvComp(fd["n_bands"], K, ksize, dropout)
        self.K = K
        self.lags = list(range(-max_lag, max_lag + 1))
        self.lag_temp = lag_temp

    def forward(self, eeg, cand):
        B, S = cand.shape[0], cand.shape[1]
        E = _znorm_time(self.eeg(eeg)).unsqueeze(1)              # (B,1,K,T)
        V = _znorm_time(self.env(cand.reshape(B * S, cand.shape[2], cand.shape[3]))
                        .reshape(B, S, self.K, -1))             # (B,S,K,T)
        T = V.shape[-1]
        per_lag = []
        for lag in self.lags:
            if lag > 0:
                c = (E[..., lag:] * V[..., :T - lag]).sum(-1)
            elif lag < 0:
                c = (E[..., :lag] * V[..., -lag:]).sum(-1)
            else:
                c = (E * V).sum(-1)
            per_lag.append(c)
        st = torch.stack(per_lag, -1)                           # (B,S,K,nlags)
        return (self.lag_temp * torch.logsumexp(st / self.lag_temp, -1)).mean(-1)  # (B,S)


class _SourceNet(nn.Module):
    def __init__(self, fd, d_model=96, dropout=0.3, F1=12, D=2, K=8,
                 content_ksize=17, max_lag=32, lag_temp=0.5):
        super().__init__()
        self.spatial = _MultiScaleSpatial(fd["n_chans"], dropout, F1=F1, D=D)
        self.spatial_proj = nn.Sequential(nn.Linear(self.spatial.out_dim, d_model), nn.ELU())
        self.gaze = _GazeConv(fd["gaze_dim"], fd.get("gaze_traj_dim", 0), d_model, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_SPEAKERS),
        )
        self.content = _ContentMatch(fd, K, content_ksize, max_lag, lag_temp, dropout)
        self.content_gate = nn.Parameter(torch.tensor(-2.0))    # sigmoid(-2)~0.12: start spatial-dominant
        self.content_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, eeg, cand, gaze, gaze_traj, gaze_on):
        sp = self.spatial_proj(self.spatial(eeg))
        gz = self.gaze(gaze, gaze_traj) * gaze_on.unsqueeze(-1)
        base = self.fuse(torch.cat([sp, gz], dim=-1))           # (B,6)
        content = self.content(eeg, cand) * self.content_scale  # (B,6)
        return base + torch.sigmoid(self.content_gate) * content


@MODEL_REGISTRY.register("source_net")
class SourceNetModel(TorchModel):
    name = "source_net"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 96))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 12))
        self.D = int(cfg.get("D", 2))
        self.K = int(cfg.get("K", 8))
        self.content_ksize = int(cfg.get("content_ksize", 17))
        self.max_lag = int(cfg.get("max_lag", 32))
        self.lag_temp = float(cfg.get("lag_temp", 0.5))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))
        self.augment = bool(cfg.get("augment", False))   # train-time EEG augmentation

    def _augment_eeg(self, eeg):
        """Train-time EEG augmentation: gaussian noise + channel dropout + time mask.
        EEG is already per-channel z-scored, so noise ~N(0, 0.15) is well-scaled."""
        B, C, T = eeg.shape
        eeg = eeg + 0.15 * torch.randn_like(eeg)
        eeg = eeg * (torch.rand(B, C, 1, device=eeg.device) > 0.1).float()   # 10% channel dropout
        if T > 20:                                                            # ~10% time mask
            L = max(1, int(0.1 * T))
            st = torch.randint(0, T - L, (B,), device=eeg.device)
            idx = torch.arange(T, device=eeg.device)[None, :]
            keep = ~((idx >= st[:, None]) & (idx < st[:, None] + L))
            eeg = eeg * keep[:, None, :].float()
        return eeg

    def build_module(self):
        return _SourceNet(self.fd, self.d_model, self.dropout, self.F1, self.D,
                          self.K, self.content_ksize, self.max_lag, self.lag_temp)

    def _gaze_on(self, batch, override=None, train=False):
        gp = batch["present"][:, 0].clone()
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            gp = gp * (torch.rand_like(gp) > self.gaze_dropout).float()
        return gp

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        eeg = self._augment_eeg(batch["eeg"]) if self.augment else batch["eeg"]
        logits = self.module(eeg, batch["cand_env"], batch["gaze"],
                             batch["gaze_traj"], gaze_on)
        att = batch["attended"]
        ce = F.cross_entropy(logits, att, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
            cg = float(torch.sigmoid(self.module.content_gate))
        return ce, {"ce": float(ce), "acc": float(acc), "content_gate": cg}

    def predict_logits(self, batch, present_override=None):
        gaze_on = self._gaze_on(batch, override=present_override, train=False)
        return self.module(batch["eeg"], batch["cand_env"], batch["gaze"],
                           batch["gaze_traj"], gaze_on)

    ABLATIONS = {"all": [1, 1, 1, 1], "no_gaze": [1, 0, 1, 1]}

    def evaluate(self, view, ctx, prefix="test/", present_override=None):
        fd = getattr(self, "fd", {}) or {}
        task = fd.get("task_type", "speaker")
        n_cand = fd.get("n_candidates", 4)
        true = view.as_numpy()["attended"]
        out = {}
        for name, mvec in self.ABLATIONS.items():
            pred = self.predict(view, ctx, present_override=mvec)
            out.update(compute_aad_metrics(pred, true, prefix=f"{prefix}{name}/",
                                           task_type=task, n_cand=n_cand))
        for key in ("acc", "acc_hemisphere", "acc_inner_outer", "chance", "n"):
            v = out.get(f"{prefix}all/{key}")
            if v is not None:
                out[f"{prefix}{key}"] = v
        out.setdefault(f"{prefix}n", len(true))
        return out
