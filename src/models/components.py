"""Shared neural building blocks for the EEG/audio match-mismatch decoders."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd import Function


class EEGEncoder(nn.Module):
    """EEGNet-style spatio-temporal front-end -> token sequence (B, L, d)."""

    def __init__(self, n_chans: int, d_model: int, dropout: float = 0.2, pool: int = 8):
        super().__init__()
        F1, D = 16, 2
        self.temporal = nn.Conv1d(n_chans, F1, kernel_size=65, padding=32)
        self.bn1 = nn.BatchNorm1d(F1)
        self.depth = nn.Conv1d(F1, F1 * D, kernel_size=1, groups=F1)
        self.bn2 = nn.BatchNorm1d(F1 * D)
        self.sep = nn.Conv1d(F1 * D, d_model, kernel_size=17, padding=8,
                             groups=math.gcd(F1 * D, d_model))
        self.bn3 = nn.BatchNorm1d(d_model)
        self.pool = nn.AvgPool1d(pool)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # x: (B, C, T)
        h = self.act(self.bn1(self.temporal(x)))
        h = self.act(self.bn2(self.depth(h)))
        h = self.act(self.bn3(self.sep(h)))
        h = self.drop(self.pool(h))             # (B, d, L)
        return h.transpose(1, 2)                # (B, L, d)


class EnvelopeEncoder(nn.Module):
    """Weight-shared 1-D conv encoder: one candidate envelope -> one token."""

    def __init__(self, n_bands: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_bands, 64, 9, padding=4), nn.ELU(),
            nn.Conv1d(64, 96, 9, padding=4), nn.ELU(),
            nn.Conv1d(96, d_model, 9, padding=4), nn.ELU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B*C, n_bands, T)
        h = self.net(x)
        return self.drop(h.mean(dim=-1))        # (B*C, d)


class SpectralSpatialEncoder(nn.Module):
    """EEGNet/CSP-style spectro-spatial power encoder.

    Learned temporal filters discover frequency bands; per-band spatial filters
    (full channel mixing) discover lateralised topographies; **log-variance pooling
    over time** turns each spatio-spectral filter into a band-power feature -- the
    CSP principle that captures attention-modulated alpha lateralisation. Output is
    a fixed power-feature vector, NOT a token sequence, so it needs no millisecond
    audio alignment (robust to this corpus's software-timestamp jitter).
    """

    def __init__(self, n_chans: int, dropout: float = 0.3, F1: int = 16, D: int = 2,
                 ksize: int = 65):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, ksize), padding=(0, ksize // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.drop = nn.Dropout(dropout)
        self.out_dim = F1 * D

    def forward(self, x):                       # x: (B, C, T)
        h = x.unsqueeze(1)                      # (B, 1, C, T)
        h = self.bn1(self.temporal(h))          # (B, F1, C, T)  band-filtered
        h = self.bn2(self.spatial(h))           # (B, F1*D, 1, T) spatially filtered
        h = h.squeeze(2)                        # (B, F1*D, T)
        p = torch.log(h.var(dim=-1) + 1e-6)     # (B, F1*D) log band-power
        return self.drop(p)


class FeatureToken(nn.Module):
    """Project an engineered feature vector to one token."""

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)
