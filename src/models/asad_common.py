"""Shared backbone + augmentation for the asad_eeg / asad_mm AAD nets.

Design rationale (evidence from this corpus):
  * Structural-prior heads and added capacity (factorised, latent-azimuth) LOST to a
    plain flat spectro-spatial net on ~750 within-subject windows. Auxiliary
    supervision was the only thing that helped. So the "top-notch" net here is a
    well-regularised, multi-task spectro-spatial CNN -- NOT a transformer.
  * The attention signal is alpha/beta lateralisation -> a multi-scale CSP front-end
    with log-variance pooling, recalibrated by a cheap SE channel-attention.
  * Tiny data -> training-time EEG augmentation (time/channel masking + noise) is the
    highest-EV regulariser.
"""
from __future__ import annotations

import torch
from torch import nn

from .source_net.model import _MultiScaleSpatial


def augment_eeg(eeg, noise_std=0.1, t_mask_frac=0.2, n_chan_mask=2, p=0.5):
    """Train-time EEG augmentation on (B, C, T). Per-sample, applied w.p. ``p``.

    Channel-wise Gaussian noise (scaled by each channel's own std), a random
    contiguous time mask, and a few zeroed channels. Cheap, label-preserving, and
    the strongest small-data regulariser available here.
    """
    if not torch.is_grad_enabled():          # eval -> no augmentation
        return eeg
    B, C, T = eeg.shape
    out = eeg
    sel = torch.rand(B, device=eeg.device) < p
    if sel.any():
        out = out.clone()
        # additive noise scaled by per-channel std
        if noise_std > 0:
            sd = out.std(dim=-1, keepdim=True)
            out = out + sel.view(B, 1, 1) * noise_std * sd * torch.randn_like(out)
        # contiguous time mask
        if t_mask_frac > 0 and T > 4:
            w = max(1, int(t_mask_frac * T))
            starts = torch.randint(0, max(1, T - w), (B,), device=eeg.device)
            ar = torch.arange(T, device=eeg.device).view(1, 1, T)
            m = (ar >= starts.view(B, 1, 1)) & (ar < (starts + w).view(B, 1, 1))
            out = out.masked_fill(sel.view(B, 1, 1) & m, 0.0)
        # channel mask
        if n_chan_mask > 0 and C > n_chan_mask:
            chan = torch.rand(B, C, device=eeg.device).argsort(dim=1)[:, :n_chan_mask]
            cm = torch.zeros(B, C, dtype=torch.bool, device=eeg.device)
            cm.scatter_(1, chan, True)
            out = out.masked_fill((sel.view(B, 1) & cm).unsqueeze(-1), 0.0)
    return out


class _SEGate(nn.Module):
    """Squeeze-and-excitation feature recalibration on a flat feature vector."""

    def __init__(self, dim, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(4, dim // r)), nn.ELU(),
            nn.Linear(max(4, dim // r), dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class EEGBackbone(nn.Module):
    """Multi-scale spectro-spatial (CSP) front-end + SE gate -> d_model embedding."""

    def __init__(self, n_chans, d_model=96, dropout=0.3, F1=12, D=2):
        super().__init__()
        self.spatial = _MultiScaleSpatial(n_chans, dropout, F1=F1, D=D)
        self.se = _SEGate(self.spatial.out_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.spatial.out_dim, d_model), nn.LayerNorm(d_model), nn.ELU(),
            nn.Dropout(dropout))

    def forward(self, eeg):
        return self.proj(self.se(self.spatial(eeg)))
