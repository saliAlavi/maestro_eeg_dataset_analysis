"""Concrete AAD runner: protocol sweep, per-split fit/eval, aggregation, logging."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.paths import ProjectPaths
from ..common.wandb_utils import WandbLogger
from ..data.aad_compat import bootstrap_ci
from ..data.factory import build_datamodule
from ..models.base import RunContext
from ..models.factory import build_model
from .base import AbstractRunner

log = logging.getLogger("runner")


def _train_val_split(train_view, val_frac):
    """Carve the chronological tail (val_frac) of a train view as validation.

    Windows are ordered by (trial recording order, window start) so the held-out
    val block is the most-recent training trials -- no look-ahead, and val precedes
    the test block in time. Returns (train_view, val_view_or_None).
    """
    from ..data.base import AADView

    if not val_frac or val_frac <= 0 or len(train_view) < 10:
        return train_view, None
    recs, idxs, spec = train_view.records, train_view.indices, train_view.spec
    order = sorted(range(len(idxs)),
                   key=lambda i: (recs[idxs[i].rec_ptr].trial_k, idxs[i].start))
    n_val = max(1, int(round(val_frac * len(order))))
    val_ids = set(order[-n_val:])
    tr = [idxs[i] for i in order if i not in val_ids]
    va = [idxs[i] for i in order if i in val_ids]
    name = getattr(train_view, "name", "")
    return (AADView(recs, tr, spec, f"{name}_tr"),
            AADView(recs, va, spec, f"{name}_val"))


class AADRunner(AbstractRunner):
    def __init__(self, cfg, paths: ProjectPaths, wandb: WandbLogger, device: str):
        self.cfg = cfg
        self.paths = paths
        self.wandb = wandb
        self.device = device
        self.run_name = wandb.name
        self.model_name = cfg.model.name

    def _device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def run(self) -> pd.DataFrame:
        dm = build_datamodule(self.cfg.data)
        log.info("preparing data for subjects %s ...", list(self.cfg.data.subjects))
        dm.prepare()
        fd = dm.feature_dims()
        log.info("feature dims: %s", fd)
        device = self._device()
        protocols = list(self.cfg.runner.get("protocols", ["within"]))

        model_dir = self.paths.model_dir(self.run_name)
        gstep = [0]
        rows = []
        detail = []

        max_folds = self.cfg.runner.get("max_folds", None)
        val_frac = float(self.cfg.runner.get("val_frac", 0.0))

        for protocol in protocols:
            for desc, train_view, test_view in dm.splits(protocol):
                if max_folds is not None and desc.fold is not None \
                        and desc.fold >= int(max_folds):
                    continue
                ctx = RunContext(device=device, wandb=self.wandb,
                                 model_dir=model_dir, split_name=desc.name,
                                 global_step=gstep)
                model = build_model(self.cfg.model, fd)
                # carve a chronological validation tail from train for best-model
                # selection / early stopping (no leakage: val precedes test in time).
                fit_train, fit_val = _train_val_split(train_view, val_frac)
                log.info("=== %s | %s | model=%s | train=%d val=%d test=%d ===",
                         protocol, desc.name, self.model_name,
                         len(fit_train), len(fit_val) if fit_val else 0, len(test_view))
                try:
                    model.fit(fit_train, fit_val, ctx)
                    metrics = model.evaluate(test_view, ctx, prefix="test/")
                    model.save(model_dir / f"{desc.name}.pt")
                except Exception as exc:
                    log.exception("split %s failed: %r", desc.name, exc)
                    continue
                row = dict(model=self.model_name, protocol=protocol,
                           split=desc.name, test_subject=desc.test_subject,
                           fold=desc.fold, n=metrics.get("test/n", 0),
                           chance=metrics.get("test/chance", 0.25),
                           acc=metrics.get("test/acc", float("nan")),
                           acc_hemisphere=metrics.get("test/acc_hemisphere", float("nan")),
                           acc_inner_outer=metrics.get("test/acc_inner_outer", float("nan")))
                rows.append(row)
                detail.append(dict(split=desc.name, protocol=protocol,
                                   test_subject=desc.test_subject, **metrics))
                self.wandb.log({f"{self.model_name}/{desc.name}/{k}": v
                                for k, v in metrics.items()
                                if isinstance(v, (int, float))})
                log.info("[%s] %s -> acc=%.3f (chance=%.2f, n=%d)",
                         self.model_name, desc.name, row["acc"], row["chance"], row["n"])

        df = pd.DataFrame(rows)
        agg = self._aggregate(df)
        self._save(df, agg, detail)
        return df

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = []
        tasks = ["acc", "acc_hemisphere", "acc_inner_outer"]
        for protocol in df["protocol"].unique():
            sub = df[df["protocol"] == protocol]
            chance_primary = float(sub["chance"].iloc[0]) if "chance" in sub else 0.25
            for task in tasks:
                if task not in sub or sub[task].isna().all():
                    continue
                # per-subject mean first (within-subject has folds), then CI across subjects
                per_sub = sub.groupby("test_subject")[task].mean().values
                per_sub = per_sub[np.isfinite(per_sub)]
                if len(per_sub) == 0:
                    continue
                m, lo, hi = bootstrap_ci(per_sub)
                chance = chance_primary if task == "acc" else 0.5
                out.append(dict(model=self.model_name, protocol=protocol, task=task,
                                mean_acc=m, ci_lo=lo, ci_hi=hi,
                                n_subjects=len(per_sub), chance=chance))
        return pd.DataFrame(out)

    def _save(self, df, agg, detail):
        run_dir = self.paths.run_dir(self.run_name)
        df.to_parquet(run_dir / "per_split.parquet")
        if not agg.empty:
            agg.to_parquet(run_dir / "aggregate.parquet")
        with open(run_dir / "detail.json", "w") as f:
            json.dump(detail, f, indent=2, default=float)
        log.info("saved results -> %s", run_dir)
        if not agg.empty:
            log.info("AGGREGATE:\n%s", agg.to_string(index=False))
            self.wandb.summary({f"agg/{r.protocol}/{r.task}": r.mean_acc
                                for r in agg.itertuples()})
