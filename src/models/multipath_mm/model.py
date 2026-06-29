"""multipath_mm -- multi-path match-mismatch AAD decoder with a frozen first stage.

Motivation (why this model exists; see README for the full story):
  ``recon_mix`` showed that letting the EEG match the attended talker in several
  *content* spaces (envelope / HuBERT / GPT-2-surprisal) and learning a softmax
  mixture is a clean, leak-free design -- candidates are the permuted real talkers
  (``mm_task=match``) so direction gives no shortcut, and trial-disjoint within-subject
  splits mean the frozen audio embeddings cannot leak any audio->label prior.

  This model takes that idea further on two axes the neuroscience demands:

  1. **Per-band EEG front-ends.** Cortical envelope/content tracking lives in delta-theta
     (~1-9 Hz); spatial attention shows up as alpha (8-14 Hz) lateralisation. A single
     shared EEG encoder (as in recon_mix) conflates them. Here each path gets the EEG
     band it should physiologically use.

  2. **A directional path.** The permuted-candidate match task deliberately destroys the
     slot<->direction mapping so content paths stay position-blind. We add ONE path that
     is *allowed* to use direction: an alpha-band CSP decodes the PHYSICAL attended
     position, which is then re-indexed into candidate-slot order via ``cand_pos``. This
     lets us cleanly attribute performance to covert content tracking vs spatial/orienting.

  Stage 1 = the per-path matchers (each trained by its own CE + reconstruction loss).
  Stage 2 = a learned softmax mixture over the per-path scores, trained on DETACHED
  branch outputs so the fusion can never bias stage 1 ("first stage frozen"). At
  inference the same mixture combines the (non-detached) scores.

Paths (enable any subset via cfg.paths):
  * ``env`` -- EEG(delta-theta) -> 28-band gammatone envelope        (content)
  * ``w2v`` -- EEG(broadband)   -> HuBERT layer-9 PCA                 (content, auditory)
  * ``sem`` -- EEG(delta-theta) -> GPT-2 surprisal/entropy/onset      (content, semantic)
  * ``dir`` -- EEG(alpha) CSP   -> physical position -> cand slots    (directional/spatial)
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin
from torch import nn

from ..base import TorchModel
from ..components import EEGEncoder, SpectralSpatialEncoder
from ..factory import MODEL_REGISTRY

log = logging.getLogger("model")

CONTENT_PATHS = ("env", "w2v", "sem")
ALL_PATHS = ("env", "w2v", "sem", "dir", "gaze")
# EEG band each content path reads (broadband = no filter).
PATH_BAND = {"env": "dt", "w2v": "broad", "sem": "dt", "dir": "alpha"}


def corr_scores(g: torch.Tensor, cand: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-candidate Pearson corr of a reconstruction (B,D,W) with cands (B,S,D,W)."""
    B, S, D, W = cand.shape
    gf = g.reshape(B, 1, D * W)
    cf = cand.reshape(B, S, D * W)
    gf = (gf - gf.mean(-1, keepdim=True)) / (gf.std(-1, keepdim=True) + eps)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + eps)
    return (gf * cf).mean(-1)                                   # (B, S)


def _zc(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score per-sample across the candidate axis -> scale-free fusion inputs."""
    return (z - z.mean(-1, keepdim=True)) / (z.std(-1, keepdim=True) + eps)


class _BandFilter(nn.Module):
    """Fixed (non-trainable) FIR band-pass applied identically to every EEG channel.

    The band is a physiological prior, not a learned parameter, so the weights are
    frozen. Implemented as a depthwise conv1d so all 32 channels share one kernel.
    """

    def __init__(self, n_chans: int, sr: float, low: float, high: float, numtaps: int = 65):
        super().__init__()
        ny = 0.5 * sr
        high = min(high, ny - 1e-3)
        taps = firwin(numtaps, [low / ny, high / ny], pass_zero=False).astype(np.float32)
        w = torch.from_numpy(taps).view(1, 1, -1).repeat(n_chans, 1, 1)   # (C,1,K)
        self.register_buffer("weight", w)
        self.pad = numtaps // 2
        self.groups = n_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B,C,T)
        return F.conv1d(x, self.weight, padding=self.pad, groups=self.groups)


class _Dec(nn.Module):
    """EEG latent (B,d,L) -> reconstructed representation (B,out,W)."""

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


class _ContentPath(nn.Module):
    """EEG(band) -> reconstruct a target representation -> corr-score candidates."""

    def __init__(self, n_chans: int, out_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.enc = EEGEncoder(n_chans, d_model, dropout)
        self.dec = _Dec(d_model, out_dim, dropout)
        self.scale = nn.Parameter(torch.tensor(2.3026))           # exp() ~ 10

    def forward(self, eeg_band: torch.Tensor, cand: torch.Tensor):
        h = self.enc(eeg_band).transpose(1, 2)                    # (B,d,L)
        g = self.dec(h, cand.shape[-1])                           # (B,out,W)
        scores = corr_scores(g, cand) * self.scale.exp().clamp(max=100.0)
        return scores, g                                          # (B,S), recon


class _DirPath(nn.Module):
    """EEG(alpha) CSP -> physical-position logits, re-indexed into candidate order."""

    def __init__(self, n_chans: int, n_cand: int, dropout: float):
        super().__init__()
        self.csp = SpectralSpatialEncoder(n_chans, dropout=dropout, ksize=65)
        self.head = nn.Sequential(
            nn.Linear(self.csp.out_dim, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, n_cand),
        )

    def forward(self, eeg_alpha: torch.Tensor, cand_pos: torch.Tensor):
        pos_logits = self.head(self.csp(eeg_alpha))               # (B, n_pos) over PHYSICAL positions
        return torch.gather(pos_logits, 1, cand_pos)              # (B, S) in candidate-slot order


class _GazePath(nn.Module):
    """The prior best signal: overt orienting. Gaze stats + trajectory -> physical
    position -> re-indexed into candidate slots (gaze points at the attended speaker)."""

    def __init__(self, gaze_dim: int, traj_dim: int, n_cand: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(gaze_dim + traj_dim, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, n_cand),
        )

    def forward(self, gaze: torch.Tensor, gaze_traj: torch.Tensor, cand_pos: torch.Tensor):
        pos_logits = self.head(torch.cat([gaze, gaze_traj], -1))
        return torch.gather(pos_logits, 1, cand_pos)


class _MultiPath(nn.Module):
    def __init__(self, fd: dict, paths, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.paths = list(paths)
        n_chans, sr = fd["n_chans"], fd["sr"]
        n_cand = fd.get("n_candidates", 4)
        self.dt = _BandFilter(n_chans, sr, 1.0, 9.0)
        self.alpha = _BandFilter(n_chans, sr, 8.0, 14.0)
        out_dims = {"env": fd["n_bands"], "w2v": fd["w2v_dim"], "sem": fd["sem_dim"]}
        self.content = nn.ModuleDict({
            p: _ContentPath(n_chans, out_dims[p], d_model, dropout)
            for p in self.paths if p in CONTENT_PATHS
        })
        self.dir = _DirPath(n_chans, n_cand, dropout) if "dir" in self.paths else None
        self.gaze = (_GazePath(fd["gaze_dim"], fd.get("gaze_traj_dim", 0), n_cand, dropout)
                     if "gaze" in self.paths else None)
        self.mix = nn.Parameter(torch.zeros(len(self.paths)))     # learned softmax fusion

    def _eeg(self, eeg: torch.Tensor, band: str) -> torch.Tensor:
        if band == "dt":
            return self.dt(eeg)
        if band == "alpha":
            return self.alpha(eeg)
        return eeg                                                # broadband

    def forward(self, eeg, cand: dict, cand_pos, gaze=None, gaze_traj=None):
        per, recon = {}, {}
        for p in self.paths:
            if p == "dir":
                per[p] = self.dir(self._eeg(eeg, "alpha"), cand_pos)
            elif p == "gaze":
                per[p] = self.gaze(gaze, gaze_traj, cand_pos)
            else:
                s, g = self.content[p](self._eeg(eeg, PATH_BAND[p]), cand[p])
                per[p], recon[p] = s, g
        mw = torch.softmax(self.mix, 0)
        # fusion is trained on DETACHED branch scores -> stage 1 stays frozen wrt fusion
        fused_detached = sum(mw[i] * _zc(per[p].detach()) for i, p in enumerate(self.paths))
        fused = sum(mw[i] * _zc(per[p]) for i, p in enumerate(self.paths))   # for inference
        return per, recon, fused_detached, fused, mw


@MODEL_REGISTRY.register("multipath_mm")
class MultiPathMMModel(TorchModel):
    name = "multipath_mm"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 128))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.w_aux = float(cfg.get("w_aux", 1.0))      # per-path CE (trains each branch)
        self.w_recon = float(cfg.get("w_recon", 1.0))  # content reconstruction
        self.w_fuse = float(cfg.get("w_fuse", 1.0))    # fusion CE (trains only the mixture)
        paths = list(cfg.get("paths", list(ALL_PATHS)))
        bad = [p for p in paths if p not in ALL_PATHS]
        if bad:
            raise ValueError(f"unknown paths {bad}; valid: {ALL_PATHS}")
        self.paths = paths

    def build_module(self):
        return _MultiPath(self.fd, self.paths, self.d_model, self.dropout)

    @staticmethod
    def _cand(batch):
        c = {"env": batch["cand_env"]}
        if "cand_w2v" in batch:
            c["w2v"] = batch["cand_w2v"]
            c["sem"] = batch["cand_sem"]
        return c

    def _run(self, batch):
        """Forward the module from a batch dict (routes gaze tensors to the gaze path)."""
        return self.module(batch["eeg"], self._cand(batch), batch["cand_pos"],
                           gaze=batch.get("gaze"), gaze_traj=batch.get("gaze_traj"))

    def compute_loss(self, batch):
        att = batch["attended"]
        cand = self._cand(batch)
        per, recon, fused_detached, fused, mw = self._run(batch)
        idx = torch.arange(att.shape[0], device=att.device)

        aux = sum(F.cross_entropy(per[p], att) for p in self.paths) / len(self.paths)
        rl = att.new_zeros((), dtype=torch.float32)
        for p in self.paths:
            if p in recon:
                ac = cand[p][idx, att]                            # attended candidate (B,D,W)
                rl = rl + F.mse_loss(recon[p], ac) + (1.0 - corr_scores(recon[p], ac.unsqueeze(1)).mean())
        rl = rl / max(1, len(recon))
        fuse_ce = F.cross_entropy(fused_detached, att)
        total = self.w_fuse * fuse_ce + self.w_aux * aux + self.w_recon * rl

        with torch.no_grad():
            acc = (fused.argmax(1) == att).float().mean()
            logs = {"total": float(total), "fuse_ce": float(fuse_ce), "aux": float(aux),
                    "recon": float(rl), "acc": float(acc)}
            mwd = mw.detach()
            for i, p in enumerate(self.paths):
                logs[f"mix_{p}"] = float(mwd[i])
                logs[f"acc_{p}"] = float((per[p].argmax(1) == att).float().mean())
        return total, logs

    def predict_logits(self, batch, present_override=None):
        _, _, _, fused, _ = self._run(batch)
        return fused

    def evaluate(self, view, ctx, prefix="test/", present_override=None) -> dict:
        """Window-level slot accuracy PLUS trial-level PHYSICAL-speaker accuracy.

        Each window has its own candidate permutation, so window 'slot' posteriors
        are not directly averagable. We map each window's slot-softmax back to the 4
        PHYSICAL speakers via cand_pos, accumulate per trial, then argmax -> one
        physical-speaker decision per trial. This matches the published 4-class
        baselines (which decide once per ~30 s trial) and lets us report hemisphere /
        inner-outer (meaningful only on physical-speaker indices)."""
        import collections

        import numpy as np
        from ..base import compute_aad_metrics

        self.module.eval()
        device = ctx.device
        loader = view.as_torch_loader(self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        slot_pred, slot_true, phys_post = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch, device)
                logits = self.predict_logits(batch, present_override=present_override)
                logits = logits.masked_fill(~batch["cand_mask"].bool(), float("-inf"))
                soft = torch.softmax(logits, 1)                       # (B, S) over slots
                pp = torch.zeros_like(soft).scatter(1, batch["cand_pos"], soft)  # (B,4) physical
                phys_post.append(pp.cpu().numpy())
                slot_pred.append(logits.argmax(1).cpu().numpy())
                slot_true.append(batch["attended"].cpu().numpy())
        if not slot_true:
            return {f"{prefix}n": 0}
        phys_post = np.concatenate(phys_post)
        out = compute_aad_metrics(np.concatenate(slot_pred), np.concatenate(slot_true),
                                  prefix=prefix, task_type="match", n_cand=4)
        # trial-level aggregation (windows materialise in view-index order, shuffle=False)
        recs, idxs = view.records, view.indices
        agg = collections.defaultdict(lambda: np.zeros(4, np.float64))
        att = {}
        for k, wi in enumerate(idxs):
            r = recs[wi.rec_ptr]
            key = (int(r.subject), int(r.trial_k))
            agg[key] += phys_post[k]
            att[key] = int(r.attended) - 1
        keys = list(agg)
        tr_pred = np.array([agg[t].argmax() for t in keys])
        tr_true = np.array([att[t] for t in keys])
        out.update(compute_aad_metrics(tr_pred, tr_true, prefix=f"{prefix}trial_",
                                       task_type="speaker", n_cand=4))
        # export per-trial physical-speaker posteriors (test only) for late fusion with
        # the gaze-logreg baseline. Each trial is out-of-sample in its CV fold.
        if prefix == "test/" and getattr(ctx, "model_dir", None) is not None:
            import pandas as pd
            recs2 = []
            for (subj, tr) in keys:
                v = agg[(subj, tr)]
                v = v / max(v.sum(), 1e-9)
                recs2.append(dict(subject=subj, trial=tr, attended=att[(subj, tr)],
                                  p0=v[0], p1=v[1], p2=v[2], p3=v[3]))
            ctx.model_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(recs2).to_parquet(ctx.model_dir / f"post_{ctx.split_name}.parquet")
        return out

    # ---- two-stage training: branches first, then OUT-OF-FOLD fusion ----------
    def fit(self, train_view, val_view, ctx) -> dict:
        # Phase 1: train the per-path matchers only (fusion disabled). The mixture
        # gets no gradient here, so it stays uniform; branches are selected by
        # best-on-val (early stop), which guards against the memorisation that made
        # the joint-trained fusion trust overfit content paths.
        saved = self.w_fuse
        self.w_fuse = 0.0
        try:
            best = super().fit(train_view, val_view, ctx)
        finally:
            self.w_fuse = saved
        # Phase 2: refit the softmax mixture on the held-out val split with the
        # branches FROZEN. The fusion now sees per-path posteriors on data the
        # branches did not train their weights on -> it can down-weight paths that
        # only memorised (the documented "tune fusion out-of-fold" fix).
        if val_view is not None and len(val_view) and len(self.paths) > 1:
            self._fit_fusion_oof(val_view, ctx)
        return best

    def _fit_fusion_oof(self, view, ctx, steps: int = 400, lr: float = 0.05) -> None:
        self.module.eval()
        device = ctx.device
        loader = view.as_torch_loader(self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        Z = {p: [] for p in self.paths}
        Y = []
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch, device)
                per, _, _, _, _ = self._run(batch)
                for p in self.paths:
                    Z[p].append(_zc(per[p]).detach())
                Y.append(batch["attended"])
        if not Y:
            return
        Zc = {p: torch.cat(Z[p], 0) for p in self.paths}     # (N, S) per path
        y = torch.cat(Y, 0)
        mix = self.module.mix.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([mix], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            mw = torch.softmax(mix, 0)
            fused = sum(mw[i] * Zc[p] for i, p in enumerate(self.paths))
            F.cross_entropy(fused, y).backward()
            opt.step()
        with torch.no_grad():
            self.module.mix.copy_(mix.detach())
        mw = torch.softmax(self.module.mix, 0).tolist()
        log.info("[%s] %s: OOF fusion mix = %s", self.name, ctx.split_name,
                 {p: round(float(mw[i]), 3) for i, p in enumerate(self.paths)})
