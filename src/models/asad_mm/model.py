"""asad_mm -- EEG + gaze attended-source detector (publication multimodal net).

Same hardened EEG backbone as `asad_eeg` (multi-scale CSP + SE + augmentation +
multi-task aux), plus the gaze fusion that the evidence here supports:

  * **Per-sample learned gaze reliability gate.** Gaze quality is wildly
    heterogeneous (cached data: some subjects carry full azimuth, some only
    hemisphere, some are dead) and varies window-to-window. A scalar
    r=sigmoid(head(gaze_emb)) in [0,1] scales the gaze contribution, on top of the
    presence/dropout gate -- a dead-gaze window is down-weighted, a clean one
    trusted. (This is what let source_rel match/beat the flat fusion without
    regressing.)
  * EEG auxiliary hemisphere/eccentricity supervision off the EEG embedding, so the
    no_gaze (EEG-only) path is supervised on both geometric axes.

`evaluate()` reports `all` (EEG+gaze) vs `no_gaze` (EEG-only) -- the latter is the
exact gaze-ablation of THIS model, so the comparison is clean. Reuses the aad_spec
cache via data=aad_source.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..asad_common import EEGBackbone, augment_eeg
from ..base import HEMISPHERE, INNER_OUTER, TorchModel, compute_aad_metrics
from ..factory import MODEL_REGISTRY
from ..source_net.model import _GazeConv
from ...data.windows import N_SPEAKERS


class _AsadMM(nn.Module):
    def __init__(self, fd, d_model=96, dropout=0.3, F1=12, D=2, use_rel_gate=True):
        super().__init__()
        self.use_rel_gate = use_rel_gate
        self.backbone = EEGBackbone(fd["n_chans"], d_model, dropout, F1, D)
        self.gaze = _GazeConv(fd["gaze_dim"], fd.get("gaze_traj_dim", 0), d_model, dropout)
        self.gaze_rel = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ELU(), nn.Linear(d_model // 2, 1))
        nn.init.constant_(self.gaze_rel[-1].bias, 2.0)        # start ~0.88 (trust gaze)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_SPEAKERS))
        self.hemi = nn.Linear(d_model, 1)
        self.ecc = nn.Linear(d_model, 1)

    def encode(self, eeg, gaze, gaze_traj, gaze_on):
        sp = self.backbone(eeg)
        gz_raw = self.gaze(gaze, gaze_traj)
        if self.use_rel_gate:
            rel = torch.sigmoid(self.gaze_rel(gz_raw)).squeeze(-1)
        else:
            rel = torch.ones_like(gaze_on)                    # plain fusion (source_net-style)
        gz = gz_raw * (rel * gaze_on).unsqueeze(-1)
        return sp, gz, rel

    def forward(self, eeg, gaze, gaze_traj, gaze_on):
        sp, gz, _ = self.encode(eeg, gaze, gaze_traj, gaze_on)
        return self.fuse(torch.cat([sp, gz], dim=-1))


@MODEL_REGISTRY.register("asad_mm")
class AsadMMModel(TorchModel):
    name = "asad_mm"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 96))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 12))
        self.D = int(cfg.get("D", 2))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))
        self.aux_weight = float(cfg.get("aux_weight", 0.3))
        self.use_rel_gate = bool(cfg.get("use_rel_gate", True))
        a = cfg.get("augment", {})
        self.aug = dict(noise_std=float(a.get("noise_std", 0.1)),
                        t_mask_frac=float(a.get("t_mask_frac", 0.2)),
                        n_chan_mask=int(a.get("n_chan_mask", 2)),
                        p=float(a.get("p", 0.5)))

    def build_module(self):
        return _AsadMM(self.fd, self.d_model, self.dropout, self.F1, self.D,
                       use_rel_gate=self.use_rel_gate)

    def _gaze_on(self, batch, override=None, train=False):
        gp = batch["present"][:, 0].clone()
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            gp = gp * (torch.rand_like(gp) > self.gaze_dropout).float()
        return gp

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        eeg = augment_eeg(batch["eeg"], **self.aug)
        att = batch["attended"]
        sp, gz, rel = self.module.encode(eeg, batch["gaze"], batch["gaze_traj"], gaze_on)
        logits = self.module.fuse(torch.cat([sp, gz], dim=-1))
        ce = F.cross_entropy(logits[:, :4], att, label_smoothing=self.label_smoothing)
        hemi_t = sp.new_tensor([HEMISPHERE[int(a)] for a in att.tolist()])
        ecc_t = sp.new_tensor([INNER_OUTER[int(a)] for a in att.tolist()])
        aux = (F.binary_cross_entropy_with_logits(self.module.hemi(sp).squeeze(-1), hemi_t)
               + F.binary_cross_entropy_with_logits(self.module.ecc(sp).squeeze(-1), ecc_t))
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
        for key in ("acc", "acc_4class", "acc_hemisphere", "acc_inner_outer", "chance", "n"):
            v = out.get(f"{prefix}all/{key}")
            if v is not None:
                out[f"{prefix}{key}"] = v
        out.setdefault(f"{prefix}n", len(true))
        return out
