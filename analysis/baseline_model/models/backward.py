"""Backward (stimulus-reconstruction) AAD baselines, evaluated as match-mismatch.

The canonical AAD baseline: reconstruct the attended speech envelope from EEG, then
decide by correlating the reconstruction against candidate envelopes. Here the
candidates are the SAME talker time-shifted (confound-free match-mismatch), so a
correct decision requires genuine cortical envelope tracking.

Two models:
  * LinearBackward  -- the classic Wong/O'Sullivan/Fuglsang linear decoder (a single
    Conv1d integrating a lag window across channels). Closed-form-like, canonical
    reference floor.
  * VLAAIBackward   -- a VLAAI-style modern deep backward network (Accou et al. 2023,
    Nature Sci. Reports): residual fully-convolutional blocks with output-context
    layers, reconstructing the broadband envelope. The modern headline decoder.

Both reconstruct a broadband envelope (B, T); the match-mismatch decision correlates
the reconstruction with each candidate's broadband envelope. Deliberately EEG-only and
single-reconstruction (no learned similarity head, no gaze/video/fusion) -> headroom
for the method paper is preserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EEG_CH = 32


# --------------------------------------------------------------------------- #
# Pearson correlation over the time axis
# --------------------------------------------------------------------------- #
def pearson(a, b, eps=1e-8):
    """a,b: (..., T) -> (...) Pearson r over the last axis."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = torch.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + eps
    return torch.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)


def neg_pearson_loss(r_hat, target):
    return -pearson(r_hat, target).mean()


# --------------------------------------------------------------------------- #
# Linear backward decoder (canonical reference floor)
# --------------------------------------------------------------------------- #
class LinearBackward(nn.Module):
    """Single causal Conv1d integrating an ~integration_window lag across 32 channels
    -> reconstructed envelope(s). n_out=1 broadband; n_out=28 multi-band spectrogram."""

    def __init__(self, n_ch=N_EEG_CH, integration_window=64, n_out=1):
        super().__init__()
        self.pad = integration_window - 1
        self.conv = nn.Conv1d(n_ch, n_out, integration_window)

    def forward(self, eeg):                       # eeg (B,32,T)
        x = self.conv(F.pad(eeg, (self.pad, 0)))  # causal left-pad -> (B,n_out,T)
        return x.squeeze(1) if x.shape[1] == 1 else x


# --------------------------------------------------------------------------- #
# VLAAI-style deep backward decoder (modern headline)
# --------------------------------------------------------------------------- #
class _VLAAIBlock(nn.Module):
    """Residual CNN stack + an output-context temporal-mixing layer. LeakyReLU,
    Dropout, no BatchNorm (avoids train-subject running-stat leak in LOSO)."""

    def __init__(self, hidden, kernel=9, out_context=33, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.LeakyReLU(0.1),
            nn.Dropout(drop),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.LeakyReLU(0.1),
            nn.Conv1d(hidden, hidden, out_context, padding=out_context // 2),  # output context
        )

    def forward(self, x):
        return x + self.net(x)                    # residual / skip


class VLAAIBackward(nn.Module):
    def __init__(self, n_ch=N_EEG_CH, hidden=128, n_blocks=4, kernel=9,
                 out_context=33, drop=0.2, n_out=1):
        super().__init__()
        self.inp = nn.Conv1d(n_ch, hidden, 1)     # learned spatial projection
        self.blocks = nn.ModuleList(
            _VLAAIBlock(hidden, kernel, out_context, drop) for _ in range(n_blocks))
        self.out = nn.Conv1d(hidden, n_out, 1)

    def forward(self, eeg):                       # eeg (B,32,T)
        x = self.inp(eeg)
        for blk in self.blocks:
            x = blk(x)
        x = self.out(x)                           # (B,n_out,T)
        return x.squeeze(1) if x.shape[1] == 1 else x


# --------------------------------------------------------------------------- #
# Multi-scale dilated backward decoder ("vlaai2") -- a smarter, data-efficient
# encoder for the within-subject regime.
# --------------------------------------------------------------------------- #
class _MSBlock(nn.Module):
    """Parallel dilated temporal convolutions (d=1,2,4,8) so one block spans the whole
    0-~400 ms cortical speech-response lag range at low parameter cost, then a 1x1 mix.
    GroupNorm (batch-size-independent -> safe for tiny within-subject folds and LOSO) +
    GELU + Dropout, residual."""

    def __init__(self, hidden, kernel=7, dilations=(1, 2, 4, 8), groups=8, drop=0.3):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel, padding=((kernel - 1) // 2) * d, dilation=d)
            for d in dilations)
        self.mix = nn.Conv1d(hidden * len(dilations), hidden, 1)
        self.norm = nn.GroupNorm(groups, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], 1)
        y = self.drop(self.act(self.norm(self.mix(y))))
        return x + y


class MSDilatedBackward(nn.Module):
    """Learned spatial mixing -> multi-scale dilated residual blocks -> reconstruction.
    n_out=1 broadband, n_out=28 multi-band spectrogram."""

    def __init__(self, n_ch=N_EEG_CH, hidden=128, n_blocks=4, kernel=7,
                 dilations=(1, 2, 4, 8), groups=8, drop=0.3, n_out=1):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(n_ch, hidden, 1), nn.GroupNorm(groups, hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            _MSBlock(hidden, kernel, dilations, groups, drop) for _ in range(n_blocks))
        self.out = nn.Conv1d(hidden, n_out, 1)

    def forward(self, eeg):                                # (B,32,T)
        x = self.inp(eeg)
        for blk in self.blocks:
            x = blk(x)
        x = self.out(x)                                    # (B,n_out,T)
        return x.squeeze(1) if x.shape[1] == 1 else x


def build_backward(name, **kw):
    if name == "linear":
        return LinearBackward(**kw)
    if name == "vlaai":
        return VLAAIBackward(**kw)
    if name == "vlaai2":
        return MSDilatedBackward(**kw)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Match-mismatch decision from a reconstruction
# --------------------------------------------------------------------------- #
def mm_scores(r_hat, cand):
    """Correlation of the reconstruction with each candidate -> scores (B,K).
    Broadband: r_hat (B,T), cand (B,K,T). Multi-band: r_hat (B,C,T),
    cand (B,K,C,T) -> per-band Pearson averaged over the C bands."""
    if cand.dim() == 4:                            # multi-band (B,K,C,T)
        r = pearson(r_hat.unsqueeze(1), cand)                 # (B,K,C)
        r = torch.atanh(r.clamp(-0.999, 0.999))              # Fisher-z before band mean
        return r.mean(-1)                                    # (B,K)
    return pearson(r_hat.unsqueeze(1), cand)       # broadband (B,K)
