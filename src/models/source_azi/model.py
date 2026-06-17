"""source_azi -- attended source as a latent AZIMUTH with reliability-weighted
multi-observer fusion.

Motivation (grounded in this corpus's data):
  * The 4 attendable speakers lie on an ordered azimuth line
    (idx 0..3 = Left-outer, Left-inner, Right-inner, Right-outer). A flat 4/6-way
    softmax throws this ordering away; inner/outer stays at chance.
  * Gaze reliability is wildly heterogeneous across subjects: some carry full
    azimuth (gaze-x monotone in speaker), some only hemisphere (flat within a
    hemisphere), some are dead (no overt orienting -> chance). A fixed-trust fusion
    is wrong for all but one of them.

Design: treat each modality as a NOISY OBSERVER of one shared latent azimuth.
  EEG observer  : multi-scale spectro-spatial (CSP) encoder -> (mu_e, log-prec_e)
  gaze observer : calibrated gaze trajectory/stats         -> (mu_g, log-prec_g)
Each emits a mean azimuth AND a per-sample precision (how much to trust itself).
Fuse by precision (Bayesian):  mu = sum(prec_i mu_i)/sum(prec_i),  prec = sum(prec_i).
Read out the speaker by an azimuth-anchored Gaussian likelihood:
  logit_k = -0.5 * prec * (mu - anchor_k)^2   over the 4 ordered speaker anchors.

Losses:
  * 4-class CE on the anchored logits (top-level decision).
  * Per-observer heteroscedastic Gaussian NLL over azimuth:
      NLL_i = 0.5*prec_i*(mu_i - anchor[att])^2 - 0.5*log_prec_i
    -- trains each observer to predict its OWN reliability, which is what makes the
    fusion subject-adaptive (a dead gaze observer learns low precision and is
    ignored; a full-azimuth one dominates). Also keeps the EEG observer individually
    supervised so gaze can't hijack all the gradient.

EEG-only vs EEG+gaze is the gaze-observer ablation (prec_g -> 0): same model, so the
comparison is exact. Reuses the aad_spec cache via data=aad_source -- no re-caching.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..factory import MODEL_REGISTRY
from ..source_net.model import _GazeConv, _MultiScaleSpatial
from ...data.windows import N_SPEAKERS

_LOGPREC_MIN, _LOGPREC_MAX = -4.0, 6.0   # clamp predicted log-precision for stability
_MU_SCALE = 1.2                           # bound mu to (-1.2, 1.2); anchors in [-1, 1]


class _Observer(nn.Module):
    """Map an embedding -> (mu azimuth, log-precision)."""

    def __init__(self, in_dim, d_model, dropout):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.ELU())
        self.mu = nn.Linear(d_model, 1)
        self.logprec = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = _MU_SCALE * torch.tanh(self.mu(h)).squeeze(-1)             # (B,)
        lp = self.logprec(h).squeeze(-1).clamp(_LOGPREC_MIN, _LOGPREC_MAX)
        return mu, lp


class _SourceAzi(nn.Module):
    def __init__(self, fd, d_model=96, dropout=0.3, F1=12, D=2):
        super().__init__()
        # EEG observer: multi-scale CSP -> azimuth.
        self.spatial = _MultiScaleSpatial(fd["n_chans"], dropout, F1=F1, D=D)
        self.eeg_obs = _Observer(self.spatial.out_dim, d_model, dropout)
        # gaze observer: conv over the (calibrated) gaze trajectory + stats -> azimuth.
        # a learned affine on the raw gaze stats is the per-subject self-calibration.
        self.gaze_cal = nn.Linear(fd["gaze_dim"], fd["gaze_dim"])
        self.gaze_enc = _GazeConv(fd["gaze_dim"], fd.get("gaze_traj_dim", 0), d_model, dropout)
        self.gaze_obs = _Observer(d_model, d_model, dropout)
        # 4 ordered speaker azimuth anchors (Left-outer .. Right-outer).
        anchors = torch.tensor([-1.0, -1.0 / 3, 1.0 / 3, 1.0])
        self.register_buffer("anchors", anchors)                       # (4,)

    def observers(self, eeg, gaze, gaze_traj):
        mu_e, lp_e = self.eeg_obs(self.spatial(eeg))
        gz = self.gaze_enc(self.gaze_cal(gaze), gaze_traj)
        mu_g, lp_g = self.gaze_obs(gz)
        return (mu_e, lp_e), (mu_g, lp_g)

    def fuse(self, mu_e, lp_e, mu_g, lp_g, gaze_on):
        prec_e = torch.exp(lp_e)
        prec_g = torch.exp(lp_g) * gaze_on                              # gate gaze precision
        denom = prec_e + prec_g + 1e-6
        mu = (prec_e * mu_e + prec_g * mu_g) / denom                   # (B,)
        prec = prec_e + prec_g                                          # (B,)
        return mu, prec

    def logits(self, mu, prec):
        d = mu.unsqueeze(-1) - self.anchors                            # (B,4)
        base4 = -0.5 * prec.unsqueeze(-1) * d * d                      # (B,4)
        out = base4.new_full((base4.shape[0], N_SPEAKERS), -1e4)
        out[:, :4] = base4
        return out

    def forward(self, eeg, gaze, gaze_traj, gaze_on):
        (mu_e, lp_e), (mu_g, lp_g) = self.observers(eeg, gaze, gaze_traj)
        mu, prec = self.fuse(mu_e, lp_e, mu_g, lp_g, gaze_on)
        return self.logits(mu, prec)


@MODEL_REGISTRY.register("source_azi")
class SourceAziModel(TorchModel):
    name = "source_azi"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 96))
        self.dropout = float(cfg.get("dropout", 0.3))
        self.F1 = int(cfg.get("F1", 12))
        self.D = int(cfg.get("D", 2))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.05))
        self.nll_weight = float(cfg.get("nll_weight", 0.5))

    def build_module(self):
        return _SourceAzi(self.fd, self.d_model, self.dropout, self.F1, self.D)

    def _gaze_on(self, batch, override=None, train=False):
        gp = batch["present"][:, 0].clone()
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            gp = gp * (torch.rand_like(gp) > self.gaze_dropout).float()
        return gp

    @staticmethod
    def _nll(mu, lp, target_azi):
        prec = torch.exp(lp)
        d = mu - target_azi
        return 0.5 * prec * d * d - 0.5 * lp                           # (B,)

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        eeg, gaze, gtraj = batch["eeg"], batch["gaze"], batch["gaze_traj"]
        att = batch["attended"]
        (mu_e, lp_e), (mu_g, lp_g) = self.module.observers(eeg, gaze, gtraj)
        mu, prec = self.module.fuse(mu_e, lp_e, mu_g, lp_g, gaze_on)
        logits = self.module.logits(mu, prec)
        ce = F.cross_entropy(logits[:, :4], att, label_smoothing=self.label_smoothing)
        # per-observer heteroscedastic azimuth NLL toward the true speaker's anchor.
        tgt = self.module.anchors[att]                                 # (B,)
        nll_e = self._nll(mu_e, lp_e, tgt).mean()
        nll_g_all = self._nll(mu_g, lp_g, tgt)
        w = gaze_on
        nll_g = (nll_g_all * w).sum() / (w.sum() + 1e-6)               # only present/kept gaze
        loss = ce + self.nll_weight * (nll_e + nll_g)
        with torch.no_grad():
            acc = (logits[:, :4].argmax(1) == att).float().mean()
            pg = float(torch.exp(lp_g).mean()); pe = float(torch.exp(lp_e).mean())
        return loss, {"ce": float(ce.detach()), "nll_e": float(nll_e.detach()),
                      "nll_g": float(nll_g.detach()), "acc": float(acc),
                      "prec_e": pe, "prec_g": pg}

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
