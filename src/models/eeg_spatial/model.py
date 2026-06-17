"""eeg_spatial -- spectral/spatial (lateralisation) attended-speaker decoder.

Why this exists (the mistake that led here): envelope-tracking decoders -- the
classical CCA backward models AND our reconstruction-driven `recon_mm` -- are all
at chance on this corpus, because audio<->EEG alignment is software-timestamped
(no hardware trigger; ~100-255 ms variable playback latency, lags scattered with
std ~600 ms). Stimulus-envelope reconstruction needs <~50 ms precision, so it is
unrecoverable here. See `analysis/scripts/diag_lag_jitter.py`.

What DOES survive that jitter is **spatial attention**: attending a side/azimuth
modulates parieto-occipital **alpha (8-12 Hz) lateralisation**, a power feature
that needs no precise audio timing. This model decodes the attended speaker
(1..4, which map to fixed azimuths/hemispheres) directly from EEG band-power via
an EEGNet/CSP spectro-spatial encoder + log-variance pooling -- no audio, no
envelope, no reconstruction. It targets the only EEG signal shown to beat chance
here (spectral baseline ~0.58). Requires a cache that keeps alpha (eeg_lp_hz off).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel
from ..components import SpectralSpatialEncoder
from ..factory import MODEL_REGISTRY
from ...data.windows import N_SPEAKERS


class _EEGSpatial(nn.Module):
    def __init__(self, fd: dict, d_model: int = 64, dropout: float = 0.3,
                 F1: int = 16, D: int = 2):
        super().__init__()
        self.enc = SpectralSpatialEncoder(fd["n_chans"], dropout, F1=F1, D=D)
        self.head = nn.Sequential(
            nn.Linear(self.enc.out_dim, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_SPEAKERS),          # 6 logits; speakers 5/6 masked by base
        )

    def features(self, eeg):
        return self.enc(eeg)

    def forward(self, eeg):
        return self.head(self.enc(eeg))              # (B, 6)


@MODEL_REGISTRY.register("eeg_spatial")
class EEGSpatialModel(TorchModel):
    name = "eeg_spatial"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 64))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 16))
        self.D = int(cfg.get("D", 2))

    def build_module(self):
        return _EEGSpatial(self.fd, self.d_model, self.dropout, self.F1, self.D)

    def compute_loss(self, batch):
        logits = self.module(batch["eeg"])
        att = batch["attended"]
        ce = F.cross_entropy(logits, att)
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return ce, {"ce": float(ce), "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        return self.module(batch["eeg"])
