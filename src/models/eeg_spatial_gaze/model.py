"""eeg_spatial_gaze -- eeg_spatial + gaze fusion.

`eeg_spatial` (spectral/lateralisation attended-speaker classifier) plus the
eye-tracker gaze stream, fused so we can measure -- against `eeg_spatial` as the
EEG-only control -- whether overt-orienting gaze helps or hurts. Both EEG (alpha
lateralisation) and gaze (eye direction) encode the *spatial* locus of attention,
so this is the honest multimodal-spatial decoder for this corpus (where envelope
tracking is dead; see eeg_spatial docstring).

Fusion = presence-aware gated late fusion (same contract as recon_mm_gaze):
gaze -> a 6-speaker spatial prior; a gaze-conditioned gate g in [0,1] (forced 0
when gaze is absent) sets logits = s_eeg + g * s_gaze. `evaluate()` reports `all`
(gaze on) vs `no_gaze` (gate off) from one model; gaze modality-dropout keeps the
EEG path honest.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..components import SpectralSpatialEncoder, FeatureToken
from ..factory import MODEL_REGISTRY
from ...data.windows import N_SPEAKERS


class _EEGSpatialGaze(nn.Module):
    def __init__(self, fd: dict, d_model: int = 64, dropout: float = 0.3,
                 F1: int = 16, D: int = 2):
        super().__init__()
        self.enc = SpectralSpatialEncoder(fd["n_chans"], dropout, F1=F1, D=D)
        self.head = nn.Sequential(
            nn.Linear(self.enc.out_dim, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_SPEAKERS),
        )
        self.gaze_tok = FeatureToken(fd["gaze_dim"], d_model, dropout)
        self.gaze_prior = nn.Linear(d_model, N_SPEAKERS)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, eeg, gaze, gaze_on):
        s_eeg = self.head(self.enc(eeg))                 # (B, 6)
        ge = self.gaze_tok(gaze)
        s_gaze = self.gaze_prior(ge)                     # (B, 6)
        gate = torch.sigmoid(self.gate(ge)).squeeze(-1) * gaze_on   # (B,)
        logits = s_eeg + gate.unsqueeze(-1) * s_gaze
        return logits, gate


@MODEL_REGISTRY.register("eeg_spatial_gaze")
class EEGSpatialGazeModel(TorchModel):
    name = "eeg_spatial_gaze"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 64))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 16))
        self.D = int(cfg.get("D", 2))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))

    def build_module(self):
        return _EEGSpatialGaze(self.fd, self.d_model, self.dropout, self.F1, self.D)

    def _gaze_on(self, batch, override=None, train=False):
        gp = batch["present"][:, 0].clone()
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            gp = gp * (torch.rand_like(gp) > self.gaze_dropout).float()
        return gp

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        logits, gate = self.module(batch["eeg"], batch["gaze"], gaze_on)
        att = batch["attended"]
        ce = F.cross_entropy(logits, att)
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return ce, {"ce": float(ce), "acc": float(acc), "gate": float(gate.mean())}

    def predict_logits(self, batch, present_override=None):
        gaze_on = self._gaze_on(batch, override=present_override, train=False)
        logits, _ = self.module(batch["eeg"], batch["gaze"], gaze_on)
        return logits

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
