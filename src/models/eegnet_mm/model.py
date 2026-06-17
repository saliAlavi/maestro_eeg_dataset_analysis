"""EEGNet match-mismatch decoder -- EEG + audio, no orienting modalities.

The EEG window is encoded to a query embedding; each of the six candidate
speaker envelopes is encoded (weight-shared) to a key; the attended speaker is
the candidate whose key best matches the query (scaled dot-product, cross-
entropy over candidates). This is the modern neural AAD workhorse (Accou et al.,
ICASSP Auditory-EEG-Decoding) and the EEG-and-audio-only sibling of MAESTRO-Net.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..base import TorchModel
from ..components import EEGEncoder, EnvelopeEncoder
from ..factory import MODEL_REGISTRY


class _EEGNetMM(nn.Module):
    def __init__(self, fd: dict, d_model: int, dropout: float):
        super().__init__()
        self.eeg_enc = EEGEncoder(fd["n_chans"], d_model, dropout)
        self.env_enc = EnvelopeEncoder(fd["n_bands"], d_model, dropout)
        self.q_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.cand_bias = nn.Parameter(torch.zeros(1))
        self.d_model = d_model

    def forward(self, eeg, cand):                # eeg (B,C,T), cand (B,6,B_bands,T)
        B, S = cand.shape[0], cand.shape[1]
        q = self.q_proj(self.eeg_enc(eeg).mean(1))           # (B,d)
        cand_flat = cand.reshape(B * S, cand.shape[2], cand.shape[3])
        cemb = self.env_enc(cand_flat).reshape(B, S, self.d_model)
        logits = (q.unsqueeze(1) * cemb).sum(-1) / math.sqrt(self.d_model)
        return logits + self.cand_bias                        # (B,6)


@MODEL_REGISTRY.register("eegnet_mm")
class EEGNetMMModel(TorchModel):
    name = "eegnet_mm"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 128))
        self.dropout = float(cfg.get("dropout", 0.2))

    def build_module(self):
        return _EEGNetMM(self.fd, self.d_model, self.dropout)

    def compute_loss(self, batch):
        logits = self.module(batch["eeg"], batch["cand_env"])
        loss = nn.functional.cross_entropy(logits, batch["attended"])
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == batch["attended"]).float().mean()
        return loss, {"ce": float(loss), "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        return self.module(batch["eeg"], batch["cand_env"])
