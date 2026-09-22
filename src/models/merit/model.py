"""MERIT: the model-layer entry point.

Training differs from a conventional loop in two places, both of which follow
from the same premise -- that accuracy is not evidence:

  * the **objective** contains the permutation test per stream, so a decoder
    that ignores a stream cannot reach a low training loss (``losses.py``);
  * the **checkpoint criterion** is not validation accuracy.  Where a shortcut
    exists, the accuracy criterion selects the epoch that exploits it best.  We
    retain the epoch maximising ``contribution + beta * (worst per-stream
    credit)``, so an epoch that is accurate by riding one stream loses to one
    that is slightly less accurate but reads all of them.  Both criteria are
    computed on every epoch and both are reported, so the criterion's effect is
    measured rather than assumed.

Hinge weights are self-gating: each epoch, a stream already contributing on
inner validation has its hinge weight relaxed toward zero, and a dead stream has
it raised.  The term therefore pushes a stream that is being ignored and leaves
alone one that is already being used -- it cannot manufacture a contribution
that is not in the data, which the permutation null on the test partition then
verifies independently.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..factory import MODEL_REGISTRY
from .attribution import ContributionEvaluator
from .losses import total_loss
from .modules import MeritNet

log = logging.getLogger("model.merit")

ANTI = dict(anti_shortcut=True, w_clip=1.0, w_null=0.5, w_vic=0.1, w_adv=0.3,
            margin=0.5, smoothing=0.1)


@MODEL_REGISTRY.register("merit")
class MeritModel:
    """Wraps :class:`MeritNet` behind the runner's constant model interface."""

    is_neural = True
    name = "merit"

    def __init__(self, cfg, feature_dims=None):
        self.cfg = cfg
        self.fd = feature_dims or {}
        t = cfg.get("train", {})
        self.epochs = int(t.get("epochs", 50))
        self.patience = int(t.get("patience", 12))
        self.batch_size = int(t.get("batch_size", 32))
        self.lr = float(t.get("lr", 1e-3))
        self.weight_decay = float(t.get("weight_decay", 1e-4))
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.val_shuffles = int(t.get("val_shuffles", 3))
        self.test_shuffles = int(t.get("test_shuffles", 20))
        self.select = str(t.get("select", "credit"))        # credit | contribution | acc
        self.beta = float(t.get("select_beta", 1.0))
        self.hinge_target = float(t.get("hinge_target", 0.05))
        # gated: relax the hinge on streams already contributing (default).
        # uniform: the same weight on every stream, as an ablation of the gating.
        # gated  : absolute threshold on the stream's fused contribution
        # solo   : push a stream by how much of its OWN capability the fusion discards
        # uniform: same weight on every stream (ablation)
        self.hinge_mode = str(t.get("hinge_mode", "solo"))
        self.trial_level = bool(cfg.get("trial_level", False))
        self.n_subjects = int(cfg.get("n_subjects", 0))
        self.module = None
        self.loss_cfg = self._loss_cfg()

    # -- config ------------------------------------------------------------
    def _loss_cfg(self):
        c = dict(ANTI) if self.cfg.get("anti_shortcut", True) else \
            dict(anti_shortcut=False, smoothing=0.1)
        c["w_aux"] = float(self.cfg.get("w_aux", 0.3))
        c["w_contrib"] = float(self.cfg.get("w_contrib", 0.5))
        c["contrib_margin"] = float(self.cfg.get("contrib_margin", 0.3))
        c["contrib_zeros"] = bool(self.cfg.get("contrib_zeros", True))
        return c

    def build_module(self):
        m = self.cfg
        return MeritNet(
            time_mods=tuple(m.get("time_mods", ("eeg", "gaze", "imu", "video"))),
            static_mods=tuple(m.get("static_mods", ("fovea",))),
            D=int(m.get("dim", 16)), layers=int(m.get("layers", 5)),
            spatial_filters=int(m.get("spatial_filters", 8)),
            dropout=float(m.get("dropout", 0.1)),
            coupling=bool(m.get("coupling", True)),
            orientation=tuple(m.get("orientation", ("eeg", "gaze", "imu", "video", "fovea"))),
            adversary=bool(m.get("adversary", True)),
            modality_dropout=float(m.get("modality_dropout", 0.3)),
            lag_samples=int(m.get("lag_samples", 0)),
            brain_dir=str(m.get("brain_dir", "centred")),
            eeg_channels=m.get("eeg_channels", None),
            gates=bool(m.get("gates", True)),
            encoder=str(m.get("encoder", "v2")),
            head=str(m.get("head", "corr")),
            n_subjects=self.n_subjects,
            subject_rank=int(m.get("subject_rank", 2)),
            audio_channels=int(m.get("audio_channels", 1)))

    # -- criteria ----------------------------------------------------------
    def _criterion(self, b):
        if self.select == "acc":
            return b["acc"]
        if self.select == "contribution":
            return b["contribution"]
        worst = b.get("min_shapley")
        if worst is None:
            per = [v for k, v in b.items() if k.startswith("contrib_")]
            worst = min(per) if per else 0.0
        return b["contribution"] + self.beta * worst

    def _hinge_weights(self, b, streams):
        """How hard to push each stream this epoch.

        The absolute gate (``gated``) switches a stream off once its fused
        contribution clears a fixed threshold, which in practice meant the
        strongest stream -- the one most worth defending -- was never pushed at
        all. ``solo`` instead measures what the stream reaches through its own
        pathway and pushes in proportion to how much of that the fusion is
        discarding, so a stream that is individually capable but ignored is
        pushed hardest, and one already contributing all it can is left alone.
        """
        if self.hinge_mode == "uniform":
            return {m: 1.0 for m in streams}
        w = {}
        for m in streams:
            fused = b.get(f"contrib_{m}", 0.0)
            if self.hinge_mode == "solo":
                solo = b.get(f"solo_contrib_{m}", 0.0)
                w[m] = float(np.clip(max(0.0, solo - fused) / max(solo, 1e-3), 0.0, 1.0))
            else:
                w[m] = float(np.clip((self.hinge_target - fused) /
                                     max(self.hinge_target, 1e-6), 0.0, 1.0))
        return w

    # -- training ----------------------------------------------------------
    def fit(self, train_view, val_view, ctx):
        device = ctx.device
        if self.module is None:
            self.module = self.build_module()
        self.module.to(device)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                         patience=5, min_lr=1e-6)
        tr_loader = train_view.as_torch_loader(self.batch_size, subject_batches=True)
        vl_loader = val_view.as_torch_loader(self.batch_size)
        n_par = sum(p.numel() for p in self.module.parameters())
        log.info("[merit] %s: %d params | %d train / %d val windows on %s",
                 ctx.split_name, n_par, len(train_view), len(val_view), device)

        hinge_w, bad = None, 0
        best = {"sel": (-9e9, None), "acc": (-9e9, None)}
        history = []
        for ep in range(1, self.epochs + 1):
            self.module.train()
            tot, nb, drops_acc = 0.0, 0, {}
            pbar = tqdm(tr_loader, desc=f"merit {ctx.split_name} ep{ep}",
                        leave=False, unit="batch")
            for mods, audio, lab, spk_of_slot, att, subj, _trial in pbar:
                mods = {m: v.to(device) for m, v in mods.items()}
                audio = [a.to(device) for a in audio]
                lab, att = lab.to(device), att.to(device)
                spk_of_slot, subj = spk_of_slot.to(device), subj.to(device)
                out = self.module(mods, audio, spk_of_slot, subj=subj)
                loss, parts, drops = total_loss(self.module, out, lab, att, subj,
                                                spk_of_slot, self.loss_cfg,
                                                hinge_weights=hinge_w)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
                opt.step()
                tot += float(loss.detach()); nb += 1
                for k, v in drops.items():
                    drops_acc[k] = drops_acc.get(k, 0.0) + v
                if nb % 20 == 0:
                    pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()})

            ev = ContributionEvaluator(self.module, vl_loader, device,
                                       trial_level=self.trial_level)
            b = ev.battery(n_shuffle=self.val_shuffles, seed=7, shapley=True,
                           shapley_draws=2)
            crit = self._criterion(b)
            sch.step(crit)
            hinge_w = self._hinge_weights(b, ev.streams)
            row = {"epoch": ep, "loss": tot / max(nb, 1), "criterion": crit,
                   **{f"val/{k}": v for k, v in b.items() if isinstance(v, float)},
                   **{f"train/drop_{k}": v / max(nb, 1) for k, v in drops_acc.items()},
                   **{f"hinge_w/{k}": v for k, v in hinge_w.items()}}
            history.append(row)
            ctx.log({f"credit/{ctx.split_name}/{k}": v for k, v in row.items()})

            if crit > best["sel"][0]:
                best["sel"] = (crit, {k: v.detach().cpu().clone()
                                      for k, v in self.module.state_dict().items()})
                bad = 0
            else:
                bad += 1
            if b["acc"] > best["acc"][0]:
                best["acc"] = (b["acc"], {k: v.detach().cpu().clone()
                                          for k, v in self.module.state_dict().items()})
            if ep == 1 or ep % 5 == 0:
                log.info("[merit] %s ep%d loss=%.4f acc=%.4f null=%.4f contrib=%+.4f "
                         "crit=%+.4f", ctx.split_name, ep, row["loss"], b["acc"],
                         b["null_mean"], b["contribution"], crit)
            if bad >= self.patience:
                log.info("[merit] %s early stop @ep%d", ctx.split_name, ep)
                break

        self._best = best
        self.history = history
        if best["sel"][1] is not None:
            self.module.load_state_dict(best["sel"][1])
        return {"val/criterion": best["sel"][0], "val/acc": best["acc"][0],
                "n_params": n_par}

    # -- evaluation --------------------------------------------------------
    def evaluate(self, test_view, ctx, prefix="test/"):
        """Score the retained checkpoint once, under both selection criteria."""
        device = ctx.device
        loader = test_view.as_torch_loader(self.batch_size)
        res = {}
        for sel in ("sel", "acc"):
            state = getattr(self, "_best", {}).get(sel, (None, None))[1]
            if state is None:
                continue
            self.module.load_state_dict(state)
            ev = ContributionEvaluator(self.module, loader, device,
                                       strata=test_view.strata(),
                                       trial_level=self.trial_level)
            b = ev.battery(n_shuffle=self.test_shuffles, shapley=True)
            tag = "selected" if sel == "sel" else "sel_acc"
            res.update({f"{prefix}{tag}/{k}": v for k, v in b.items()})
        if getattr(self, "_best", {}).get("sel", (None, None))[1] is not None:
            self.module.load_state_dict(self._best["sel"][1])
        res[f"{prefix}n_params"] = sum(p.numel() for p in self.module.parameters())
        return res

    def save(self, path: Path):
        if self.module is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": self.module.state_dict(),
                        "cfg": dict(self.cfg) if hasattr(self.cfg, "keys") else None},
                       path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location="cpu")
        if self.module is None:
            self.module = self.build_module()
        self.module.load_state_dict(ckpt["state_dict"])


__all__ = ["MeritModel"]
