"""source_rel -- source_net's flat fusion + learned gaze reliability + EEG aux supervision.

Two prior structural-prior heads (source_hier factorised, source_azi latent-azimuth)
both LOST to source_net's plain flat fusion: imposing geometry bottlenecked capacity
and hurt on this small within-subject data. So source_rel KEEPS source_net's flexible
flat 6-way head unchanged and adds only the two changes the data actually justifies:

  1. **Per-sample learned gaze reliability gate.** The cached data shows gaze quality
     is wildly heterogeneous (S2 full azimuth, S3 hemisphere-only, S1 dead/flat) and it
     also varies window-to-window (the subject overtly orients on some trials, not
     others). A scalar reliability r=sigmoid(head(gaze_emb)) in [0,1] scales the gaze
     contribution per window, on top of the presence/dropout gate. A dead gaze window
     can be down-weighted; a clean one trusted. r is initialised ~0.9 (trust, then
     learn to distrust).

  2. **EEG hemisphere/eccentricity auxiliary supervision.** Two linear heads off the
     EEG embedding predict hemisphere and inner/outer (BCE). This injects the known
     geometry as EXTRA supervision (multi-task) WITHOUT replacing the flat head, and
     -- because it trains the EEG embedding directly -- it strengthens the EEG-only
     (no_gaze) path, the weak spot.

The content-match branch from source_net is dropped (envelope tracking is absent on
this corpus; its gate learned ~0 and it cost ~2 min/batch). source_rel >= source_net
by construction: the gate can learn r->1 and the aux loss can be ignored.
`evaluate()` reports all (EEG+gaze) vs no_gaze (EEG-only). Reuses the aad_spec cache.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import HEMISPHERE, INNER_OUTER, TorchModel, compute_aad_metrics
from ..factory import MODEL_REGISTRY
from ..source_net.model import _GazeConv, _MultiScaleSpatial
from ...data.windows import N_SPEAKERS


class _SourceRel(nn.Module):
    def __init__(self, fd, d_model=96, dropout=0.3, F1=12, D=2):
        super().__init__()
        self.spatial = _MultiScaleSpatial(fd["n_chans"], dropout, F1=F1, D=D)
        self.spatial_proj = nn.Sequential(nn.Linear(self.spatial.out_dim, d_model), nn.ELU())
        self.gaze = _GazeConv(fd["gaze_dim"], fd.get("gaze_traj_dim", 0), d_model, dropout)
        # per-sample gaze reliability gate (from the gaze embedding) -> [0,1].
        self.gaze_rel = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ELU(), nn.Linear(d_model // 2, 1))
        nn.init.constant_(self.gaze_rel[-1].bias, 2.0)        # start ~sigmoid(2)=0.88 (trust gaze)
        # flat fusion head (unchanged from source_net).
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_SPEAKERS))
        # auxiliary geometry heads off the EEG embedding (strengthen EEG-only).
        self.hemi_head = nn.Linear(d_model, 1)
        self.ecc_head = nn.Linear(d_model, 1)

    def encode(self, eeg, gaze, gaze_traj, gaze_on):
        sp = self.spatial_proj(self.spatial(eeg))             # (B,d) EEG embedding
        gz_raw = self.gaze(gaze, gaze_traj)                   # (B,d)
        rel = torch.sigmoid(self.gaze_rel(gz_raw)).squeeze(-1)  # (B,) learned reliability
        gz = gz_raw * (rel * gaze_on).unsqueeze(-1)           # gate by reliability AND presence
        return sp, gz, rel

    def forward(self, eeg, gaze, gaze_traj, gaze_on):
        sp, gz, _ = self.encode(eeg, gaze, gaze_traj, gaze_on)
        return self.fuse(torch.cat([sp, gz], dim=-1))         # (B,6)


@MODEL_REGISTRY.register("source_rel")
class SourceRelModel(TorchModel):
    name = "source_rel"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 96))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 12))
        self.D = int(cfg.get("D", 2))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))
        self.aux_weight = float(cfg.get("aux_weight", 0.3))

    def build_module(self):
        return _SourceRel(self.fd, self.d_model, self.dropout, self.F1, self.D)

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
        sp, gz, rel = self.module.encode(eeg, gaze, gtraj, gaze_on)
        logits = self.module.fuse(torch.cat([sp, gz], dim=-1))
        ce = F.cross_entropy(logits[:, :4], att, label_smoothing=self.label_smoothing)
        # EEG aux supervision: hemisphere + eccentricity off the EEG embedding.
        hemi_t = sp.new_tensor([HEMISPHERE[int(a)] for a in att.tolist()])
        ecc_t = sp.new_tensor([INNER_OUTER[int(a)] for a in att.tolist()])
        aux = (F.binary_cross_entropy_with_logits(self.module.hemi_head(sp).squeeze(-1), hemi_t)
               + F.binary_cross_entropy_with_logits(self.module.ecc_head(sp).squeeze(-1), ecc_t))
        loss = ce + self.aux_weight * aux
        with torch.no_grad():
            acc = (logits[:, :4].argmax(1) == att).float().mean()
        return loss, {"ce": float(ce.detach()), "aux": float(aux.detach()),
                      "acc": float(acc), "gaze_rel": float(rel.mean())}

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
