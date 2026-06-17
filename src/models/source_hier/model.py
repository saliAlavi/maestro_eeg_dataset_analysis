"""source_hier -- geometry-factorised 4-way attended-source identification.

Upgrade of `source_net` that keeps the backbone that works on this corpus
(multi-scale spectro-spatial EEG + conv gaze-trajectory encoder) but replaces the
flat 6-way readout with a **factorised hemisphere x eccentricity head**:

  The 4 attendable speakers are the product of two binary geometric factors --
    hemisphere  : {1,2}=Left,  {3,4}=Right     (carried by EEG spatial lateralisation)
    eccentricity: {2,3}=Inner, {1,4}=Outer     (carried by gaze azimuth)
  so the attended-speaker decision is exactly (hemisphere, eccentricity). We predict
  two scalar logits h (Right>0) and e (Outer>0) from the fused EEG+gaze embedding and
  COMPOSE the per-speaker logits from the fixed geometry instead of learning a flat
  6-way softmax. Each binary bit is also supervised directly (auxiliary BCE).

Why this beats the flat head on this corpus:
  * Injects the known speaker geometry as inductive bias -- the flat softmax has to
    rediscover that the 4 classes factor, from only ~750 within-subject windows/fold.
  * Pools ALL windows for each binary bit (both hemispheres train the eccentricity
    head, etc.) -> far more samples per sub-decision, less overfitting.
  * Routes the comparison cleanly: with gaze dropped, the hemisphere bit stays strong
    (EEG) while eccentricity falls back to ~chance, so the factorised 4-way degrades
    gracefully instead of collapsing across hemispheres -> a fairer EEG-only readout.

The lag-robust content-match branch from source_net is intentionally dropped: this
corpus's diagnostics show envelope/content tracking is absent (its gate learned ~0)
and the 130-lag backward dominated compute. `evaluate()` reports all vs no_gaze.
Reuses the aad_spec cache via data=aad_source.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import HEMISPHERE, INNER_OUTER, TorchModel, compute_aad_metrics
from ..factory import MODEL_REGISTRY
from ..source_net.model import _GazeConv, _MultiScaleSpatial
from ...data.windows import N_SPEAKERS


class _SourceHier(nn.Module):
    def __init__(self, fd, d_model=96, dropout=0.3, F1=12, D=2):
        super().__init__()
        self.spatial = _MultiScaleSpatial(fd["n_chans"], dropout, F1=F1, D=D)
        self.spatial_proj = nn.Sequential(nn.Linear(self.spatial.out_dim, d_model), nn.ELU())
        self.gaze = _GazeConv(fd["gaze_dim"], fd.get("gaze_traj_dim", 0), d_model, dropout)
        # factorised head: two scalar logits from the fused EEG+gaze embedding.
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ELU(), nn.Dropout(dropout))
        self.hemi_head = nn.Linear(d_model, 1)   # >0 -> Right
        self.ecc_head = nn.Linear(d_model, 1)    # >0 -> Outer
        # fixed speaker geometry signs for idx 0..3 (speaker 1..4).
        sign_h = torch.tensor([+1.0 if HEMISPHERE[i] == 1 else -1.0 for i in range(4)])
        sign_e = torch.tensor([+1.0 if INNER_OUTER[i] == 1 else -1.0 for i in range(4)])
        self.register_buffer("sign_h", sign_h)   # (4,)
        self.register_buffer("sign_e", sign_e)   # (4,)

    def factors(self, eeg, gaze, gaze_traj, gaze_on):
        sp = self.spatial_proj(self.spatial(eeg))
        gz = self.gaze(gaze, gaze_traj) * gaze_on.unsqueeze(-1)
        z = self.fuse(torch.cat([sp, gz], dim=-1))
        return self.hemi_head(z).squeeze(-1), self.ecc_head(z).squeeze(-1)   # (B,),(B,)

    def compose(self, h, e):
        # compose the 4 speaker logits from the geometry, pad to N_SPEAKERS so the
        # base predict() can mask the never-attended speakers 5/6.
        base4 = h.unsqueeze(-1) * self.sign_h + e.unsqueeze(-1) * self.sign_e   # (B,4)
        base = base4.new_full((base4.shape[0], N_SPEAKERS), -1e4)
        base[:, :4] = base4
        return base

    def forward(self, eeg, gaze, gaze_traj, gaze_on):
        h, e = self.factors(eeg, gaze, gaze_traj, gaze_on)
        return self.compose(h, e)


@MODEL_REGISTRY.register("source_hier")
class SourceHierModel(TorchModel):
    name = "source_hier"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 96))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 12))
        self.D = int(cfg.get("D", 2))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))
        self.aux_weight = float(cfg.get("aux_weight", 0.5))

    def build_module(self):
        return _SourceHier(self.fd, self.d_model, self.dropout, self.F1, self.D)

    def _gaze_on(self, batch, override=None, train=False):
        gp = batch["present"][:, 0].clone()
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            gp = gp * (torch.rand_like(gp) > self.gaze_dropout).float()
        return gp

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        eeg, gaze, gtraj = batch["eeg"], batch["gaze"], batch["gaze_traj"]
        att = batch["attended"]
        h, e = self.module.factors(eeg, gaze, gtraj, gaze_on)
        logits = self.module.compose(h, e)
        # CE over the 4 valid (attendable) speaker columns -- avoids the label-smoothing
        # mass landing on the -inf-masked never-attended speakers 5/6.
        ce = F.cross_entropy(logits[:, :4], att, label_smoothing=self.label_smoothing)
        # auxiliary direct supervision on each geometric bit.
        hemi_t = self.module.sign_h.new_tensor([HEMISPHERE[int(a)] for a in att.tolist()])
        ecc_t = self.module.sign_e.new_tensor([INNER_OUTER[int(a)] for a in att.tolist()])
        aux = (F.binary_cross_entropy_with_logits(h, hemi_t)
               + F.binary_cross_entropy_with_logits(e, ecc_t))
        loss = ce + self.aux_weight * aux
        with torch.no_grad():
            acc = (logits[:, :4].argmax(1) == att).float().mean()
        return loss, {"ce": float(ce.detach()), "aux": float(aux.detach()),
                      "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        gaze_on = self._gaze_on(batch, override=present_override, train=False)
        return self.module(batch["eeg"], batch["gaze"], batch["gaze_traj"], gaze_on)

    ABLATIONS = {"all": [1, 1, 1, 1], "no_gaze": [1, 0, 1, 1]}

    def evaluate(self, view, ctx, prefix="test/", present_override=None):
        fd = getattr(self, "fd", {}) or {}
        task = fd.get("task_type", "speaker")
        n_cand = fd.get("n_candidates", 4)
        true = view.as_numpy()["attended"]
        out = {}
        for name, mvec in self.ABLATIONS.items():
            pred = self.predict(view, ctx, present_override=mvec)
            out.update(compute_aad_metrics(pred, true, prefix=f"{prefix}{name}/",
                                           task_type=task, n_cand=n_cand))
        for key in ("acc", "acc_hemisphere", "acc_inner_outer", "chance", "n"):
            v = out.get(f"{prefix}all/{key}")
            if v is not None:
                out[f"{prefix}{key}"] = v
        out.setdefault(f"{prefix}n", len(true))
        return out
