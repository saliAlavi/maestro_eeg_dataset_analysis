"""recon_mm -- reconstruction-driven match-mismatch AAD decoder (EEG + audio).

What's new vs ``eegnet_mm``. The standard neural match-mismatch model encodes the
EEG to a *query* and each candidate envelope to a *key* and picks the attended
speaker by a learned dot product. ``recon_mm`` instead **reconstructs the
attended speech envelope from the EEG** and selects the attended source as the
candidate whose envelope *correlates* most with that reconstruction. One decoder
is shared by two jointly-trained objectives:

  * reconstruction  -- EEG -> attended band-envelope (MSE + (1-corr));
  * classification  -- cross-entropy over per-candidate correlation scores.

This unifies the classical *backward* stimulus-reconstruction model (the field's
interpretable floor) with neural match-mismatch, end to end. Correlation scoring
is amplitude-invariant, so once the six candidates are loudness-equalised by the
documented table power, the 4-way speaker decision can only be solved from EEG.

Inputs are confined to the ~10 Hz cortical speech-tracking band (set in the data
config). The reconstruction target is the low-dimensional band envelope -- never
the raw waveform -- per the project's "full-audio reconstruction is infeasible".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel
from ..components import EEGEncoder
from ..factory import MODEL_REGISTRY


def corr_scores(g: torch.Tensor, cand: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-candidate Pearson correlation of a reconstruction with each envelope.

    g    : (B, n_bands, W)        reconstructed attended envelope
    cand : (B, S, n_bands, W)     S candidate envelopes
    -> (B, S)   correlation over the flattened (n_bands * W) axis.
    """
    B, S, nb, W = cand.shape
    gf = g.reshape(B, 1, nb * W)
    cf = cand.reshape(B, S, nb * W)
    gf = (gf - gf.mean(-1, keepdim=True)) / (gf.std(-1, keepdim=True) + eps)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + eps)
    return (gf * cf).mean(-1)


class _ReconMM(nn.Module):
    def __init__(self, fd: dict, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.eeg_enc = EEGEncoder(fd["n_chans"], d_model, dropout)
        self.dec = nn.Sequential(
            nn.Conv1d(d_model, d_model, 5, padding=2), nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, fd["n_bands"], 1),
        )
        # learned, bounded temperature on the correlation logits
        self.logit_scale = nn.Parameter(torch.tensor(2.3026))   # exp() ~ 10
        self.n_bands = fd["n_bands"]

    def reconstruct(self, eeg: torch.Tensor, W: int) -> torch.Tensor:
        h = self.eeg_enc(eeg).transpose(1, 2)                   # (B, d, L)
        g = self.dec(h)                                         # (B, n_bands, L)
        if g.shape[-1] != W:
            g = F.interpolate(g, size=W, mode="linear", align_corners=False)
        return g

    def forward(self, eeg: torch.Tensor, cand: torch.Tensor):
        g = self.reconstruct(eeg, cand.shape[-1])
        scores = corr_scores(g, cand) * self.logit_scale.exp().clamp(max=100.0)
        return g, scores


@MODEL_REGISTRY.register("recon_mm")
class ReconMMModel(TorchModel):
    name = "recon_mm"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.w_cls = float(cfg.get("w_cls", 1.0))
        self.w_recon = float(cfg.get("w_recon", 1.0))

    def build_module(self):
        return _ReconMM(self.fd, self.d_model, self.dropout)

    def _recon_loss(self, g, att_env):
        mse = F.mse_loss(g, att_env)
        corr = corr_scores(g, att_env.unsqueeze(1)).mean()      # (B,1)->scalar
        return mse + (1.0 - corr)

    def compute_loss(self, batch):
        g, scores = self.module(batch["eeg"], batch["cand_env"])
        att = batch["attended"]
        ce = F.cross_entropy(scores, att)
        att_env = batch["cand_env"][torch.arange(att.shape[0], device=att.device), att]
        recon = self._recon_loss(g, att_env)
        total = self.w_cls * ce + self.w_recon * recon
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (scores.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return total, {"ce": float(ce), "recon": float(recon),
                       "acc": float(acc), "total": float(total)}

    def predict_logits(self, batch, present_override=None):
        _, scores = self.module(batch["eeg"], batch["cand_env"])
        return scores
