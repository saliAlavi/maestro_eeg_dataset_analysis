"""asad_eeg -- EEG-only attended-source detector (publication backbone).

A well-regularised, multi-task spectro-spatial CNN -- the architecture the evidence
on this corpus supports (added capacity / structural-prior heads all LOST to a flat
spectro-spatial net; auxiliary supervision was the only winner).

  backbone : multi-scale CSP (alpha/beta lateralisation) + SE channel-attention
  reg      : train-time EEG augmentation (time/channel masking + noise), dropout, WD
  heads    : speaker (4-way, main) + hemisphere + inner/outer (auxiliary BCE)

The auxiliary hemisphere/eccentricity heads decompose the 4-way decision along the
two geometric axes (EEG decodes hemisphere ~0.75 well, inner/outer is the hard bit),
giving the shared embedding direct supervision on both -- which is what lifted the
EEG-only path before. No gaze: this is the honest EEG result.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..asad_common import EEGBackbone, augment_eeg
from ..base import HEMISPHERE, INNER_OUTER, TorchModel, compute_aad_metrics
from ..factory import MODEL_REGISTRY
from ...data.windows import N_SPEAKERS


class _AsadEeg(nn.Module):
    def __init__(self, fd, d_model=96, dropout=0.3, F1=12, D=2):
        super().__init__()
        self.backbone = EEGBackbone(fd["n_chans"], d_model, dropout, F1, D)
        self.speaker = nn.Linear(d_model, N_SPEAKERS)
        self.hemi = nn.Linear(d_model, 1)
        self.ecc = nn.Linear(d_model, 1)

    def forward(self, eeg):
        z = self.backbone(eeg)
        return self.speaker(z), self.hemi(z).squeeze(-1), self.ecc(z).squeeze(-1)


@MODEL_REGISTRY.register("asad_eeg")
class AsadEegModel(TorchModel):
    name = "asad_eeg"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 96))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 12))
        self.D = int(cfg.get("D", 2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))
        self.aux_weight = float(cfg.get("aux_weight", 0.3))
        a = cfg.get("augment", {})
        self.aug = dict(noise_std=float(a.get("noise_std", 0.1)),
                        t_mask_frac=float(a.get("t_mask_frac", 0.2)),
                        n_chan_mask=int(a.get("n_chan_mask", 2)),
                        p=float(a.get("p", 0.5)))

    def build_module(self):
        return _AsadEeg(self.fd, self.d_model, self.dropout, self.F1, self.D)

    def compute_loss(self, batch):
        eeg = augment_eeg(batch["eeg"], **self.aug)
        att = batch["attended"]
        logits, h, e = self.module(eeg)
        ce = F.cross_entropy(logits[:, :4], att, label_smoothing=self.label_smoothing)
        hemi_t = logits.new_tensor([HEMISPHERE[int(a)] for a in att.tolist()])
        ecc_t = logits.new_tensor([INNER_OUTER[int(a)] for a in att.tolist()])
        aux = (F.binary_cross_entropy_with_logits(h, hemi_t)
               + F.binary_cross_entropy_with_logits(e, ecc_t))
        loss = ce + self.aux_weight * aux
        with torch.no_grad():
            acc = (logits[:, :4].argmax(1) == att).float().mean()
        return loss, {"ce": float(ce.detach()), "aux": float(aux.detach()), "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        logits, _, _ = self.module(batch["eeg"])
        return logits

    def evaluate(self, view, ctx, prefix="test/", present_override=None):
        fd = getattr(self, "fd", {}) or {}
        pred = self.predict(view, ctx, present_override=present_override)
        true = view.as_numpy()["attended"]
        return compute_aad_metrics(pred, true, prefix=prefix,
                                   task_type=fd.get("task_type", "speaker"),
                                   n_cand=fd.get("n_candidates", 4))
