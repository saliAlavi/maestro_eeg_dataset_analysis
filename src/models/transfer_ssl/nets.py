"""Compact whole-trial EEG encoder + gaze classifier, with masked-channel SSL.

The decodable attended-source signal here is fragile fine channel-covariance
structure (alpha-lateralization) + overt gaze. This model:
  * Enc   -- channel-mixing temporal conv -> per-time features (the first conv mixes
             channels, so masked channels are reconstructable from the others ->
             SSL learns the spatial covariance structure WITHOUT noise/dropout, which
             corrupt the signal).
  * Clf   -- Enc -> global temporal pool -> concat gaze stats -> 4-way head.
  * Recon -- Enc -> 1x1 conv back to channels, for masked-channel SSL pretraining.

Training recipes (used by the transfer_ssl runner):
  scratch  : random init, per-subject 5-fold.
  transfer : supervised pretrain on the other subjects -> per-subject fine-tune.
  ssl      : masked-channel reconstruction pretrain (pooled, unlabeled) -> fine-tune.
  combo    : ssl-init -> supervised pretrain on others -> fine-tune (best).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def zs(x, ax=-1):
    return (x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)


class Enc(nn.Module):
    def __init__(self, C=32, h=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, h, 25, padding=12), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(h, h, 15, padding=7), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(h, h, 15, padding=7), nn.BatchNorm1d(h), nn.ELU())

    def forward(self, x):
        return self.net(x)                                   # (B,h,T)


class Clf(nn.Module):
    def __init__(self, gdim, C=32, h=64, dropout=0.3, enc=None):
        super().__init__()
        self.enc = enc or Enc(C, h, dropout)
        self.head = nn.Sequential(nn.Linear(h + gdim, 64), nn.ELU(), nn.Dropout(dropout), nn.Linear(64, 4))

    def forward(self, eeg, g):
        return self.head(torch.cat([self.enc(eeg).mean(-1), g], 1))


class Recon(nn.Module):
    def __init__(self, C=32, h=64, dropout=0.3):
        super().__init__()
        self.enc = Enc(C, h, dropout)
        self.dec = nn.Conv1d(h, C, 1)

    def forward(self, x):
        return self.dec(self.enc(x))


def _batches(n, bs, shuffle=True):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_clf(model, Xe, Xg, y, epochs, dev, lr=1e-3, wd=1e-3, bs=64, ls=0.05):
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xe, Xg, y = (torch.as_tensor(t, device=dev) for t in (Xe, Xg, y))
    for _ in range(epochs):
        for b in _batches(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            opt.zero_grad()
            F.cross_entropy(model(Xe[bi], Xg[bi]), y[bi], label_smoothing=ls).backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model, Xe, Xg, y, dev):
    model.eval()
    p = model(torch.as_tensor(Xe, device=dev), torch.as_tensor(Xg, device=dev)).argmax(1).cpu().numpy()
    return float((p == y).mean())


def ssl_pretrain(allE, dev, epochs=60, h=64, mask_frac=0.3, bs=128):
    m = Recon(C=allE.shape[1], h=h).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    E = torch.as_tensor(allE, device=dev)
    for _ in range(epochs):
        for b in _batches(len(E), bs):
            x = E[torch.as_tensor(b, device=dev)]
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=dev) > mask_frac).float()
            opt.zero_grad()
            rec = m(x * mask)
            loss = (((rec - x) ** 2) * (1 - mask)).sum() / ((1 - mask).sum() * x.shape[-1] + 1e-6)
            loss.backward()
            opt.step()
    return m.enc
