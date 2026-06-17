"""source_fusion -- full 4-way attended-SOURCE identification (EEG + rich gaze).

Goal: identify WHICH of the 4 attended sources (speakers 1-4) is attended -- the
4-class decision itself, not the left/right direction collapse. On this corpus the
4 sources are 4 fixed azimuths whose voices rotate every trial, so the decodable
signal is spatial: EEG alpha-lateralisation gives the hemisphere (left/right) bit,
but separating the two sources WITHIN a hemisphere (inner vs outer azimuth) needs
the finer eye-position cue that gaze carries -- which is why EEG-only 4-class tops
out near hemisphere-level and gaze is what lifts it.

Design:
  * EEG  -> EEGNet/CSP spectro-spatial log-power encoder (hemisphere bit).
  * Gaze -> summary stats + the raw subject-relative gaze TRAJECTORY (azimuth),
            MLP-encoded; zeroed + gaze-dropped when absent.
  * JOINT (concat) fusion of the two embeddings -> MLP -> 4-way source logits, so
    the head can use EEG×gaze interactions (e.g. gaze resolves which source on the
    EEG-indicated side). Trained with 4-class cross-entropy + label smoothing.

EEG-only control is the separate `eeg_spatial` model; `evaluate()` here also reports
`all` vs `no_gaze` from one model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..components import SpectralSpatialEncoder
from ..factory import MODEL_REGISTRY
from ...data.windows import N_SPEAKERS


class _SourceFusion(nn.Module):
    def __init__(self, fd: dict, d_model: int = 64, dropout: float = 0.3,
                 F1: int = 16, D: int = 2):
        super().__init__()
        self.eeg = SpectralSpatialEncoder(fd["n_chans"], dropout, F1=F1, D=D)
        self.eeg_proj = nn.Sequential(nn.Linear(self.eeg.out_dim, d_model), nn.ELU())
        gaze_in = fd["gaze_dim"] + fd.get("gaze_traj_dim", 0)
        self.gaze_enc = nn.Sequential(
            nn.Linear(gaze_in, d_model), nn.LayerNorm(d_model), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.ELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_SPEAKERS),
        )

    def forward(self, eeg, gaze_full, gaze_on):
        pe = self.eeg_proj(self.eeg(eeg))                    # (B, d)
        ge = self.gaze_enc(gaze_full) * gaze_on.unsqueeze(-1)  # zeroed when gaze absent
        return self.fuse(torch.cat([pe, ge], dim=-1))         # (B, 6)


@MODEL_REGISTRY.register("source_fusion")
class SourceFusionModel(TorchModel):
    name = "source_fusion"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 64))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 16))
        self.D = int(cfg.get("D", 2))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))

    def build_module(self):
        return _SourceFusion(self.fd, self.d_model, self.dropout, self.F1, self.D)

    def _gaze_full(self, batch):
        return torch.cat([batch["gaze"], batch["gaze_traj"]], dim=-1)

    def _gaze_on(self, batch, override=None, train=False):
        gp = batch["present"][:, 0].clone()
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            gp = gp * (torch.rand_like(gp) > self.gaze_dropout).float()
        return gp

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        logits = self.module(batch["eeg"], self._gaze_full(batch), gaze_on)
        att = batch["attended"]
        ce = F.cross_entropy(logits, att, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return ce, {"ce": float(ce), "acc": float(acc), "gate": float(gaze_on.mean())}

    def predict_logits(self, batch, present_override=None):
        gaze_on = self._gaze_on(batch, override=present_override, train=False)
        return self.module(batch["eeg"], self._gaze_full(batch), gaze_on)

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
