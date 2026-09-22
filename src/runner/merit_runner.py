"""Runner for MERIT: certify the candidates, sweep the protocols, report credit.

The runner is the only layer that knows about both the data and the model, and it
talks to each through its constant interface.  Beyond the usual sweep it does two
things specific to this paper:

  * it runs the **audio-only acceptance probe** on the assembled candidate bank
    *before* any training, and records it next to every number the run produces --
    a result is only interpretable against the floor of its own candidate set;
  * it aggregates the **per-stream contribution and Shapley credit** alongside
    accuracy, and reports the sign test / Wilcoxon over folds, so "the model uses
    stream m" is a claim with a p-value rather than an impression.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.paths import ProjectPaths
from ..common.wandb_utils import WandbLogger
from ..models.base import RunContext
from ..models.factory import build_model
from .base import AbstractRunner

log = logging.getLogger("runner.merit")


def _build_merit_data(cfg):
    from ..data.merit_data import MeritDataModule
    d = cfg.data
    if str(d.get("name", "merit")) == "external":
        # A public two-talker corpus, presented through the same interface.
        from ..data.external_data import ExternalDataModule
        return ExternalDataModule(
            corpus=str(d.get("corpus", "nju")),
            cache_root=d.get("cache_root", "/fs/scratch/PAS2301/alialavi/cache"),
            window_sec=float(d.get("window_sec", 10.0)),
            hop_sec=float(d.get("hop_sec", 5.0)),
            cand_mode=str(d.get("cand_mode", "qmatch")),
            n_folds=int(d.get("n_folds", 5)),
            val_frac=float(d.get("val_frac", 0.2)),
            content_holdout=float(d.get("content_holdout", 0.2)),
            seed=int(d.get("seed", 0)))
    subs = d.get("subjects", "all")
    subs = "all" if subs in ("all", None) else list(subs)
    return MeritDataModule(
        subjects=subs,
        cache_root=d.get("cache_root", "/fs/scratch/PAS2301/alialavi/cache"),
        dataset_path=d.get("dataset_path", "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset"),
        vjepa_dir=d.get("vjepa_dir", "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa"),
        modalities=tuple(d.get("modalities", ["eeg", "gaze", "imu", "video"])),
        static_mods=tuple(d.get("static_mods", ["fovea"])),
        window_sec=float(d.get("window_sec", 10.0)),
        hop_sec=float(d.get("hop_sec", 5.0)),
        cand_mode=str(d.get("cand_mode", "qmatch")),
        K=int(d.get("K", 4)),
        n_folds=int(d.get("n_folds", 5)),
        val_frac=float(d.get("val_frac", 0.2)),
        content_holdout=float(d.get("content_holdout", 0.2)),
        seed=int(d.get("seed", 0)),
        crop_len=d.get("crop_len", None),
        n_train_crops=int(d.get("n_train_crops", 5)),
        n_eval_crops=int(d.get("n_eval_crops", 5)),
        euclidean_align=bool(d.get("euclidean_align", False)),
        orient_feats=bool(d.get("orient_feats", False)),
        audio_feats=str(d.get("audio_feats", "env")),
        limit_subjects=d.get("limit_subjects", None),
        limit_content=d.get("limit_content", None))


class MeritRunner(AbstractRunner):
    def __init__(self, cfg, paths: ProjectPaths, wandb: WandbLogger, device: str):
        self.cfg = cfg
        self.paths = paths
        self.wandb = wandb
        self.device = device
        self.run_name = wandb.name
        self.out_dir = paths.run_dir(self.run_name)

    def _device(self):
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        cfg = self.cfg
        dev = self._device()
        dm = _build_merit_data(cfg).prepare()
        probe = dm.certify() if cfg.runner.get("certify", True) else float("nan")
        log.info("candidate construction '%s' K=%d | audio-only probe %.4f "
                 "(chance %.4f)", dm.cand_mode, dm.K, probe, 1 / dm.K)
        self.wandb.log({"candidates/audio_only_probe": probe,
                        "candidates/chance": 1 / dm.K})

        max_folds = cfg.runner.get("max_folds", None)
        folds = cfg.runner.get("folds", None)
        folds = None if folds is None else {int(f) for f in folds}
        variants = self._variants()
        self._check_streams(dm, variants)
        rows = []
        ctx_step = [0]
        # Splits are materialised once and reused by every variant, so all
        # variants are compared on byte-identical windows and folds.
        for protocol in cfg.runner.get("protocols", ["within"]):
            splits = [s for i, s in enumerate(dm.splits(protocol))
                      if (folds is None or i in folds)
                      and (max_folds is None or i < int(max_folds))]
            for vname, mcfg in variants:
                for name, tr, vl, te in splits:
                    tag = f"{vname}/{name}"
                    log.info("=== %s | train %d / val %d / test %d windows ===",
                             tag, len(tr), len(vl), len(te))
                    from omegaconf import OmegaConf, open_dict
                    mc = mcfg.copy()
                    with open_dict(mc):
                        mc.n_subjects = int(len(np.unique(dm.store.subject))
                                            if mcfg.get("subject_adapt", False) else 0)
                        mc.audio_channels = 2 if dm.audio_feats == "env_onset" else 1
                        # montage differs by corpus: 32 here, 64 on KUL/DTU
                        mc.eeg_channels = int(getattr(dm, "eeg_channels", 0)) or None
                    model = build_model(mc, feature_dims={"n_candidates": dm.K})
                    ctx = RunContext(device=dev, wandb=self.wandb,
                                     model_dir=self.paths.model_dir(self.run_name),
                                     split_name=tag, global_step=ctx_step)
                    fit_info = model.fit(tr, vl, ctx)
                    res = model.evaluate(te, ctx)
                    row = {"tag": str(cfg.model.get("tag", "")).lstrip("_"),
                           "run_name": self.run_name,
                           "variant": vname, "protocol": protocol, "split": name,
                           "audio_only_probe": probe, "cand_mode": dm.cand_mode,
                           "K": dm.K, "window_sec": dm.window_sec,
                           "n_train": len(tr), "n_val": len(vl), "n_test": len(te),
                           **fit_info, **res}
                    rows.append(row)
                    self.wandb.log({f"split/{tag}/{k}": v for k, v in row.items()
                                    if isinstance(v, (int, float))})
                    if cfg.runner.get("save_models", False):
                        model.save(self.paths.model_dir(self.run_name) /
                                   f"{vname}__{name}.pt")
                    self._dump(rows)

        df = pd.DataFrame(rows)
        self._dump(rows)
        self._summarise(df)
        return df

    @staticmethod
    def _check_streams(dm, variants):
        """Refuse to train a model on a stream the data layer never built.

        The model skips a declared stream that is absent from the batch, so a
        mismatch between ``model.static_mods`` and ``data.static_mods`` used to
        train silently without that stream -- which is how the per-trial
        orienting descriptors were left out of every v2 run.
        """
        have_t = set((dm.store.arrays or {}).keys())
        have_s = set((dm.store.static or {}).keys())
        for vname, mc in variants:
            miss = ([m for m in mc.get("time_mods", []) if m not in have_t] +
                    [m for m in mc.get("static_mods", []) if m not in have_s])
            if miss:
                raise ValueError(
                    f"variant {vname!r} declares streams {miss} that the data layer "
                    f"did not build (time: {sorted(have_t)}, static: {sorted(have_s)}); "
                    f"add them to data.modalities / data.static_mods")

    def _variants(self):
        """``runner.variants`` is a list of ``{name, model: {...overrides}}``.

        All variants share one materialised dataset and one set of splits, so a
        difference between two rows is a difference between two models and
        nothing else.  With no list configured the single ``model`` config runs.
        """
        from omegaconf import OmegaConf
        spec = self.cfg.runner.get("variants", None)
        if not spec:
            return [(self.cfg.model.get("tag", "") or self.cfg.model.name,
                     self.cfg.model)]
        out = []
        for v in spec:
            over = v.get("model", {}) or {}
            out.append((str(v["name"]), OmegaConf.merge(self.cfg.model, over)))
        return out

    # -- reporting ---------------------------------------------------------
    def _dump(self, rows):
        p = Path(self.out_dir) / "results.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)

    def _summarise(self, df):
        if df.empty:
            return
        from scipy import stats as sps
        lines, summary = [], {}
        keys = [k for k in ("variant", "protocol") if k in df.columns]
        for gkey, g in df.groupby(keys):
            protocol = "/".join(str(x) for x in
                                (gkey if isinstance(gkey, tuple) else (gkey,)))
            pref = "test/selected/"
            acc, con = g.get(pref + "acc"), g.get(pref + "contribution")
            if acc is None:
                continue
            lines.append(f"[{protocol}] n_folds={len(g)}  acc={acc.mean():.4f}"
                         f"+-{acc.std():.4f}  contribution={con.mean():+.4f}"
                         f"  folds_positive={(con > 0).sum()}/{len(con)}")
            summary[f"{protocol}/acc"] = float(acc.mean())
            summary[f"{protocol}/contribution"] = float(con.mean())
            if len(con) > 1:
                try:
                    summary[f"{protocol}/wilcoxon_p"] = float(
                        sps.wilcoxon(con, alternative="greater").pvalue)
                except ValueError:
                    pass
            for col in g.columns:
                if col.startswith(pref + "shapley_") or col.startswith(pref + "contrib_"):
                    stream = col.split("_", 1)[-1]
                    v = g[col].dropna()          # a stream absent from this variant
                    if v.empty:
                        continue
                    kind = "shapley" if "shapley_" in col else "contrib"
                    summary[f"{protocol}/{kind}/{stream}"] = float(v.mean())
                    lines.append(f"    {kind:8s} {stream:6s} {v.mean():+.4f}"
                                 f" (positive in {(v > 0).sum()}/{len(v)} folds)")
        txt = "\n".join(lines)
        log.info("\n%s", txt)
        (Path(self.out_dir) / "summary.txt").write_text(txt + "\n")
        (Path(self.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
        self.wandb.log({f"summary/{k}": v for k, v in summary.items()})


__all__ = ["MeritRunner"]
