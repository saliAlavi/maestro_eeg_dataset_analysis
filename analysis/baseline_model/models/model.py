"""NeuroCLIP-AAD — a modern, leakage-safe, deliberately-not-best EEG baseline.

A CLIP/CLAP-style contrastive EEG<->speech-envelope match-mismatch decoder for the
orienting-free 4-class attended-talker task. EEG is the ONLY brain modality; the
four loudness-matched, per-window-permuted 28-band gammatone candidate envelopes
are the decode target. Both streams are projected into a shared L2-normalized
d=128 space and scored by a plain frame-wise mean cosine — a **deliberately
LINEAR** readout. That linearity is the point: it is the honest floor, and the
learned similarity head (bilinear / cross-attention / deep-CCA) is left as the
first lever for the separate method paper.

Why each choice is leakage-safe (see docs/DESIGN.md for the full argument):
  * L2-normalized frame cosine  -> absolute envelope energy (loudness) cannot
    enter any score. The +3..18 dB attended-loudness confound is inert.
  * weight-shared, position-blind stimulus encoder (no slot embedding) -> the
    output is permutation-EQUIVARIANT by construction, so no slot->talker or
    slot->direction shortcut is representable (unit-tested).
  * InstanceNorm1d (not BatchNorm) -> no train-subject running statistics leak
    into a held-out LOSO subject.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EEG_CH = 32
N_BANDS = 28
D_MODEL = 128


def _conv_block(cin, cout, k, groups=1, drop=0.3, pool=True):
    layers = [
        nn.Conv1d(cin, cout, k, padding=k // 2, groups=groups),
        nn.InstanceNorm1d(cout, affine=True),
        nn.ELU(),
        nn.Dropout(drop),
    ]
    if pool:
        layers.append(nn.AvgPool1d(2))
    return nn.Sequential(*layers)


class EEGEncoder(nn.Module):
    """(B,32,T) -> (B,128,T/4), L2-normalized per frame."""

    def __init__(self, d=D_MODEL, drop=0.3):
        super().__init__()
        self.innorm = nn.InstanceNorm1d(N_EEG_CH, affine=True)
        self.spatial = nn.Conv1d(N_EEG_CH, 64, 1)          # learned spatial/CSP-like mix
        self.act = nn.ELU()
        self.temporal = _conv_block(64, 64, 17, groups=64, drop=drop)   # depthwise ~265ms TRF
        self.mix = _conv_block(64, d, 9, drop=drop)
        self.proj = nn.Conv1d(d, d, 1)

    def forward(self, eeg):                                # eeg (B,32,T)
        x = self.innorm(eeg)
        x = self.act(self.spatial(x))
        x = self.temporal(x)                               # (B,64,T/2)
        x = self.mix(x)                                    # (B,128,T/4)
        x = self.proj(x)
        return F.normalize(x, dim=1)                       # per-frame L2 over 128


class StimEncoder(nn.Module):
    """(B,4,28,T) -> (B,4,128,T/4), weight-shared & position-blind, L2-normalized."""

    def __init__(self, d=D_MODEL, drop=0.3):
        super().__init__()
        self.innorm = nn.InstanceNorm1d(N_BANDS, affine=True)
        self.c1 = _conv_block(N_BANDS, 64, 17, drop=drop)
        self.c2 = _conv_block(64, d, 9, drop=drop)
        self.proj = nn.Conv1d(d, d, 1)

    def forward(self, cand):                               # cand (B,4,28,T)
        b, k, bands, t = cand.shape
        x = cand.reshape(b * k, bands, t)                  # weight-shared over 4 candidates
        x = self.innorm(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.proj(x)
        x = F.normalize(x, dim=1)                          # per-frame L2 over 128
        return x.reshape(b, k, x.shape[1], x.shape[2])     # (B,4,128,T/4)


class NeuroCLIPAAD(nn.Module):
    def __init__(self, d=D_MODEL, drop=0.3, temp_init=0.07, lambda_clip=0.5):
        super().__init__()
        self.eeg_enc = EEGEncoder(d, drop)
        self.stim_enc = StimEncoder(d, drop)
        # CLIP log-temperature; logit_scale = exp(log_temp), clamped <= 100
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / temp_init)))))
        self.lambda_clip = lambda_clip

    def logit_scale(self):
        return self.log_temp.exp().clamp(max=100.0)

    def embed(self, eeg, cand):
        return self.eeg_enc(eeg), self.stim_enc(cand)      # (B,128,W'), (B,4,128,W')

    @staticmethod
    def frame_cosine(z_e, z_s):
        """Zero-lag frame-wise mean cosine -> (B,4). Inputs already per-frame L2."""
        # z_e (B,128,W'), z_s (B,4,128,W')
        sim = (z_e.unsqueeze(1) * z_s).sum(dim=2)          # (B,4,W') dot over 128
        return sim.mean(dim=2)                             # (B,4)

    def scores(self, eeg, cand):
        z_e, z_s = self.embed(eeg, cand)
        return self.frame_cosine(z_e, z_s)                 # (B,4) raw cosine in [-1,1]

    # ---- training objective --------------------------------------------------
    def compute_loss(self, eeg, cand, attended, label_smoothing=0.05):
        z_e, z_s = self.embed(eeg, cand)
        s = self.frame_cosine(z_e, z_s)                    # (B,4)
        scale = self.logit_scale()
        # within-scene 4-way InfoNCE: attended vs the 3 co-present talkers (hard negs)
        l_scene = F.cross_entropy(scale * s, attended, label_smoothing=label_smoothing)
        # in-batch cross-scene CLIP regularizer on pooled embeddings
        b = eeg.shape[0]
        v_att = z_s[torch.arange(b, device=eeg.device), attended]     # (B,128,W')
        e_pool = F.normalize(z_e.mean(dim=2), dim=1)                   # (B,128)
        a_pool = F.normalize(v_att.mean(dim=2), dim=1)                 # (B,128)
        m = scale * e_pool @ a_pool.t()                               # (B,B)
        tgt = torch.arange(b, device=eeg.device)
        l_clip = 0.5 * (F.cross_entropy(m, tgt) + F.cross_entropy(m.t(), tgt))
        loss = l_scene + self.lambda_clip * l_clip
        with torch.no_grad():
            scene_acc = (s.argmax(1) == attended).float().mean()
            clip_acc = (m.argmax(1) == tgt).float().mean()
        return loss, {"loss": float(loss.detach()), "l_scene": float(l_scene.detach()),
                      "l_clip": float(l_clip.detach()), "scene_acc": float(scene_acc),
                      "clip_acc": float(clip_acc), "logit_scale": float(scale.detach())}

    @torch.no_grad()
    def predict_scores(self, eeg, cand):
        """(B,4) zero-lag cosine scores; argmax = predicted attended slot."""
        return self.scores(eeg, cand)
