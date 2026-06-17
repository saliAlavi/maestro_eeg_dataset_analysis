"""recon_mix -- multi-representation reconstruction match-mismatch AAD decoder.

``recon_mm`` matches the EEG only to the low-level **envelope**, which needs ms-precise
EEG<->audio timing this corpus lacks (software sync, per-trial jitter) -- the reason
envelope decoding sits near chance here. ``recon_mix`` reconstructs the attended stream
from the EEG in THREE representation spaces and lets the model learn which one carries
the decodable signal:

  * ``env`` -- 28-band gammatone envelope        (fast, ms-precise -> jitter-fragile)
  * ``w2v`` -- HuBERT layer-9 (PCA)               (auditory/phonetic, ~5 Hz)
  * ``sem`` -- GPT-2 surprisal/entropy/onset      (semantic, N400-scale ~400 ms -> jitter-robust)

For each space the EEG predicts a representation; candidates are scored by Pearson
correlation in that space; the three per-space scores are combined by a **learned
softmax mixture** (logged as ``mix_env/w2v/sem`` -- directly reports which representation
the EEG uses). Candidates are the permuted real talkers (``mm_task=match``), so the
decision is content-based, not spatial. Hypothesis: the slower semantic/auditory spaces
survive the timing jitter that kills the envelope.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel
from ..components import EEGEncoder
from ..factory import MODEL_REGISTRY

SPACES = ("env", "w2v", "sem")


def corr_scores(g: torch.Tensor, cand: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-candidate Pearson corr of a reconstruction (B,D,W) with cands (B,S,D,W)."""
    B, S, D, W = cand.shape
    gf = g.reshape(B, 1, D * W)
    cf = cand.reshape(B, S, D * W)
    gf = (gf - gf.mean(-1, keepdim=True)) / (gf.std(-1, keepdim=True) + eps)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + eps)
    return (gf * cf).mean(-1)


class _Dec(nn.Module):
    def __init__(self, d: int, out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(d, out, 1),
        )

    def forward(self, h: torch.Tensor, W: int) -> torch.Tensor:
        g = self.net(h)
        if g.shape[-1] != W:
            g = F.interpolate(g, size=W, mode="linear", align_corners=False)
        return g


class _ReconMix(nn.Module):
    def __init__(self, fd: dict, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.eeg_enc = EEGEncoder(fd["n_chans"], d_model, dropout)
        dims = {"env": fd["n_bands"], "w2v": fd["w2v_dim"], "sem": fd["sem_dim"]}
        self.dec = nn.ModuleDict({k: _Dec(d_model, dims[k], dropout) for k in SPACES})
        self.scale = nn.ParameterDict({k: nn.Parameter(torch.tensor(2.3026)) for k in SPACES})
        self.mix = nn.Parameter(torch.zeros(len(SPACES)))     # learned softmax mixture

    def forward(self, eeg: torch.Tensor, cand: dict):
        h = self.eeg_enc(eeg).transpose(1, 2)                 # (B, d, L)
        W = cand["env"].shape[-1]
        per, recon = {}, {}
        for k in SPACES:
            g = self.dec[k](h, W)                             # (B, Dk, W)
            recon[k] = g
            per[k] = corr_scores(g, cand[k]) * self.scale[k].exp().clamp(max=100.0)
        mw = torch.softmax(self.mix, 0)
        logits = sum(mw[i] * per[k] for i, k in enumerate(SPACES))
        return logits, per, recon, mw


@MODEL_REGISTRY.register("recon_mix")
class ReconMixModel(TorchModel):
    name = "recon_mix"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.w_recon = float(cfg.get("w_recon", 1.0))
        self.w_aux = float(cfg.get("w_aux", 0.3))

    def build_module(self):
        return _ReconMix(self.fd, self.d_model, self.dropout)

    @staticmethod
    def _cand(batch):
        return {"env": batch["cand_env"], "w2v": batch["cand_w2v"], "sem": batch["cand_sem"]}

    def compute_loss(self, batch):
        cand = self._cand(batch)
        att = batch["attended"]
        logits, per, recon, mw = self.module(batch["eeg"], cand)
        ce = F.cross_entropy(logits, att)
        aux = sum(F.cross_entropy(per[k], att) for k in SPACES) / len(SPACES)
        idx = torch.arange(att.shape[0], device=att.device)
        rl = 0.0
        for k in SPACES:
            ac = cand[k][idx, att]                            # attended candidate (B,Dk,W)
            rl = rl + F.mse_loss(recon[k], ac) + (1.0 - corr_scores(recon[k], ac.unsqueeze(1)).mean())
        rl = rl / len(SPACES)
        total = ce + self.w_aux * aux + self.w_recon * rl
        with torch.no_grad():
            acc = (logits.argmax(1) == att).float().mean()
            mwd = mw.detach()
        return total, {"ce": float(ce), "aux": float(aux), "recon": float(rl),
                       "acc": float(acc), "total": float(total),
                       "mix_env": float(mwd[0]), "mix_w2v": float(mwd[1]), "mix_sem": float(mwd[2])}

    def predict_logits(self, batch, present_override=None):
        logits, _, _, _ = self.module(batch["eeg"], self._cand(batch))
        return logits
