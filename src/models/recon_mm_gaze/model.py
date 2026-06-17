"""recon_mm_gaze -- recon_mm + gaze, with explicit multimodal fusion.

Same reconstruction-driven match-mismatch backbone as ``recon_mm`` (EEG ->
attended band-envelope -> correlation scoring), with the eye-tracker gaze stream
fused in so we can measure -- against ``recon_mm`` as the EEG-only control --
whether overt-orienting gaze *helps or hurts* the attended-source decision.

Fusion is deliberately explicit and presence-aware (gaze is missing on some
trials), with two strategies selectable from the config:

  * ``gated`` (default) -- LATE fusion. Gaze produces a spatial prior over the six
    loudspeakers (gaze direction ~ attended location); a learned, gaze-conditioned
    gate ``g in [0,1]`` (forced to 0 when gaze is absent) scales how much of that
    prior is added to the EEG correlation scores:  logits = s_eeg + g * s_gaze.
    The gate makes the gaze contribution measurable and ablatable.
  * ``film`` -- gated late fusion PLUS feature-wise modulation: the gaze embedding
    FiLM-conditions the EEG token sequence before reconstruction, letting gaze
    steer (not replace) the EEG decoder.

Gaze modality-dropout during training keeps the EEG path strong on its own, so
the ``no_gaze`` evaluation (gate forced off) is a fair EEG-only readout from the
*same* trained weights -- complementing the separate ``recon_mm`` model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel, compute_aad_metrics
from ..components import EEGEncoder, FeatureToken
from ..factory import MODEL_REGISTRY
from ..recon_mm.model import corr_scores
from ...data.windows import N_SPEAKERS


class _ReconMMGaze(nn.Module):
    def __init__(self, fd: dict, d_model: int = 128, dropout: float = 0.2,
                 fusion: str = "gated"):
        super().__init__()
        self.fusion = fusion
        self.eeg_enc = EEGEncoder(fd["n_chans"], d_model, dropout)
        self.dec = nn.Sequential(
            nn.Conv1d(d_model, d_model, 5, padding=2), nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, fd["n_bands"], 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.3026))
        self.gaze_tok = FeatureToken(fd["gaze_dim"], d_model, dropout)
        self.gaze_prior = nn.Linear(d_model, N_SPEAKERS)        # spatial prior over 6 speakers
        self.gate = nn.Linear(d_model, 1)                       # gaze-conditioned mix gate
        if fusion == "film":
            self.film = nn.Linear(d_model, 2 * d_model)         # gamma, beta on EEG tokens
        self.n_bands = fd["n_bands"]

    def reconstruct(self, eeg, W, gaze_emb=None):
        h = self.eeg_enc(eeg)                                   # (B, L, d)
        if self.fusion == "film" and gaze_emb is not None:
            gamma, beta = self.film(gaze_emb).chunk(2, dim=-1)
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        g = self.dec(h.transpose(1, 2))                         # (B, n_bands, L)
        if g.shape[-1] != W:
            g = F.interpolate(g, size=W, mode="linear", align_corners=False)
        return g

    def forward(self, eeg, cand, gaze, gaze_on):
        """gaze_on: (B,) in [0,1] -- per-sample gaze availability/ablation mask."""
        ge = self.gaze_tok(gaze)                                # (B, d)
        g = self.reconstruct(eeg, cand.shape[-1],
                             ge if self.fusion == "film" else None)
        s_eeg = corr_scores(g, cand) * self.logit_scale.exp().clamp(max=100.0)
        s_gaze = self.gaze_prior(ge)                            # (B, 6)
        gate = torch.sigmoid(self.gate(ge)).squeeze(-1) * gaze_on   # (B,)
        logits = s_eeg + gate.unsqueeze(-1) * s_gaze
        return g, logits, gate


@MODEL_REGISTRY.register("recon_mm_gaze")
class ReconMMGazeModel(TorchModel):
    name = "recon_mm_gaze"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.w_cls = float(cfg.get("w_cls", 1.0))
        self.w_recon = float(cfg.get("w_recon", 1.0))
        self.fusion = str(cfg.get("fusion", "gated"))
        self.gaze_dropout = float(cfg.get("gaze_dropout", 0.2))

    def build_module(self):
        return _ReconMMGaze(self.fd, self.d_model, self.dropout, self.fusion)

    def _gaze_on(self, batch, override=None, train=False):
        """Per-sample gaze availability, with modality-dropout (train) and an
        optional [eeg,gaze,imu,video] override (eval ablation)."""
        gp = batch["present"][:, 0].clone()                    # 1 where gaze is real
        if override is not None:
            gp = gp * float(override[1])
        elif train and self.gaze_dropout > 0:
            keep = (torch.rand_like(gp) > self.gaze_dropout).float()
            gp = gp * keep
        return gp

    def _recon_loss(self, g, att_env):
        mse = F.mse_loss(g, att_env)
        corr = corr_scores(g, att_env.unsqueeze(1)).mean()
        return mse + (1.0 - corr)

    def compute_loss(self, batch):
        gaze_on = self._gaze_on(batch, train=True)
        g, logits, gate = self.module(batch["eeg"], batch["cand_env"],
                                      batch["gaze"], gaze_on)
        att = batch["attended"]
        ce = F.cross_entropy(logits, att)
        att_env = batch["cand_env"][torch.arange(att.shape[0], device=att.device), att]
        recon = self._recon_loss(g, att_env)
        total = self.w_cls * ce + self.w_recon * recon
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            acc = (logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return total, {"ce": float(ce), "recon": float(recon), "acc": float(acc),
                       "gate": float(gate.mean()), "total": float(total)}

    def predict_logits(self, batch, present_override=None):
        gaze_on = self._gaze_on(batch, override=present_override, train=False)
        _, logits, _ = self.module(batch["eeg"], batch["cand_env"],
                                   batch["gaze"], gaze_on)
        return logits

    # Read the gaze contribution off the single trained model: with vs without.
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
        # Mirror "all" to the canonical keys the runner aggregates.
        for key in ("acc", "acc_hemisphere", "acc_inner_outer", "chance", "n"):
            v = out.get(f"{prefix}all/{key}")
            if v is not None:
                out[f"{prefix}{key}"] = v
        out.setdefault(f"{prefix}n", len(true))
        return out
