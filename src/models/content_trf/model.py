"""content_trf -- whole-trial spectrogram-reconstruction content decoder for AAD.

Rebuilt to the recipe that prior work proved is the ONLY trustworthy content model on this
corpus (see the envelope_tracking_weak findings):
  * MULTI-SCALE DILATED encoder (spatial 1x1 -> dilated temporal convs 1/2/4/8 -> proj):
    the single architecture lever that moved content (+0.056 over single-scale).
  * Reconstruct the 28-BAND SPECTROGRAM (not broadband envelope, not onsets -- onsets are a
    documented dead lever; spectrogram was the biggest single gain, ~0.386 vs 0.353 env).
  * WHOLE-TRIAL correlation at LAG 0. A per-candidate lag search is a SELECTION-BIAS ARTIFACT
    here (every candidate finds a spurious peak -> wrong talkers win equally -> chance); the
    honest estimate is lag-0 over the full trial.
  * Train on 5 s windows (sample count), score by correlating the reconstruction with each
    candidate's FULL-TRIAL spectrogram -- correlation SNR accumulates over the trial.

Candidates = 4 attendable talkers, permuted + loudness-equalised -> content decision, no
spatial/loudness shortcut. MANDATORY control: an EEG-shuffle null (shuffle_eeg=true) that
breaks the EEG<->trial pairing must collapse to chance for any content number to be trusted.
Reports trial 4-class (chance 0.25), binary (attended vs unattended, chance 0.5), hemisphere.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..factory import MODEL_REGISTRY
from ..multipath_mm.model import _BandFilter, corr_scores

log = logging.getLogger("model")


class _MultiScaleEnc(nn.Module):
    """Spatial 1x1 -> parallel dilated temporal convs (1,2,4,8) -> project to n_bands."""

    def __init__(self, n_chans: int, n_bands: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.spatial = nn.Conv1d(n_chans, hidden, 1)
        self.bn = nn.BatchNorm1d(hidden)
        self.dils = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8)])
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden * 4, n_bands, 1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:         # (B,C,T) -> (B,n_bands,T)
        h = F.elu(self.bn(self.spatial(eeg)))
        h = torch.cat([F.elu(c(h)) for c in self.dils], dim=1)
        return self.proj(self.drop(h))


@MODEL_REGISTRY.register("content_trf")
class ContentTRFModel(TorchModel):
    name = "content_trf"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.hidden = int(cfg.get("hidden", 64))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.w_recon = float(cfg.get("w_recon", 1.0))
        self.band = str(cfg.get("band", "broad"))             # broad | dt
        self.shuffle_eeg = bool(cfg.get("shuffle_eeg", False))  # EEG-shuffle NULL control
        self.n_bands = int(self.fd["n_bands"])
        self.sr = float(self.fd.get("sr", 64.0))
        self._bf = None

    def build_module(self):
        m = _MultiScaleEnc(self.fd["n_chans"], self.n_bands, self.hidden, self.dropout)
        if self.band == "dt":
            self._bf = _BandFilter(self.fd["n_chans"], self.sr, 1.0, 9.0)
        return m

    def _eeg(self, eeg):
        if self.shuffle_eeg:                                   # break EEG<->label pairing
            eeg = eeg.roll(1, dims=0)
        if self._bf is not None:
            self._bf = self._bf.to(eeg.device)
            return self._bf(eeg)
        return eeg

    def compute_loss(self, batch):
        att = batch["attended"]
        recon = self.module(self._eeg(batch["eeg"]))             # (B,n_bands,W)
        cand = batch["cand_env"]                                 # (B,S,n_bands,W)
        scores = corr_scores(recon, cand) * 10.0
        ce = F.cross_entropy(scores, att)
        idx = torch.arange(att.shape[0], device=att.device)
        ac = cand[idx, att]
        rl = F.mse_loss(recon, ac) + (1.0 - corr_scores(recon, ac.unsqueeze(1)).mean())
        total = ce + self.w_recon * rl
        with torch.no_grad():
            acc = (scores.argmax(1) == att).float().mean()
        return total, {"total": float(total), "ce": float(ce), "recon": float(rl), "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        return corr_scores(self.module(self._eeg(batch["eeg"])), batch["cand_env"])

    # ---- whole-trial, lag-0 spectrogram correlation -------------------------
    def evaluate(self, view, ctx, prefix="test/", present_override=None) -> dict:
        self.module.eval()
        device = ctx.device
        seen, recs = set(), []
        for wi in view.indices:
            if wi.rec_ptr not in seen:
                seen.add(wi.rec_ptr)
                recs.append(view.records[wi.rec_ptr])
        if not recs:
            return {f"{prefix}n": 0}
        recons, cands, atts = [], [], []
        with torch.no_grad():
            for r in recs:
                eeg = torch.as_tensor(np.ascontiguousarray(r.eeg), dtype=torch.float32, device=device)
                eeg = ((eeg - eeg.mean(1, keepdim=True)) / (eeg.std(1, keepdim=True) + 1e-6))
                if self._bf is not None:
                    eeg = self._bf.to(device)(eeg.unsqueeze(0))[0]
                recons.append(self.module(eeg.unsqueeze(0))[0])               # (n_bands,T)
                cands.append(torch.as_tensor(np.ascontiguousarray(r.env[:4]),
                                             dtype=torch.float32, device=device))  # (4,n_bands,T)
                atts.append(int(r.attended) - 1)
        if self.shuffle_eeg:                                  # NULL: mispair EEG with candidates
            recons = recons[1:] + recons[:1]
        preds, binwins = [], []
        for recon, cand, att in zip(recons, cands, atts):
            T = min(recon.shape[-1], cand.shape[-1])
            recon = recon[:, :T]
            cand = cand[:, :, :T]
            # match TRAINING's norm_cand: per-(speaker,band) z-score over time, then the
            # exact training corr_scores. Scoring raw env here was a train/eval mismatch.
            cand = (cand - cand.mean(-1, keepdim=True)) / (cand.std(-1, keepdim=True) + 1e-6)
            sc = corr_scores(recon.unsqueeze(0), cand.unsqueeze(0))[0]   # (4,) lag-0 whole-trial
            preds.append(int(sc.argmax()))
            for k in range(4):
                if k != att:
                    binwins.append(float(sc[att] > sc[k]))
        preds = np.array(preds); trues = np.array(atts)
        out = compute_aad_metrics(preds, trues, prefix=f"{prefix}trial_",
                                  task_type="speaker", n_cand=4)
        out[f"{prefix}acc"] = out.get(f"{prefix}trial_acc", float("nan"))
        out[f"{prefix}binary"] = float(np.mean(binwins)) if binwins else float("nan")
        return out
