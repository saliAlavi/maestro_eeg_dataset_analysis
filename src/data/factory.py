"""Data-layer factory: build an AbstractDataModule from the Hydra config."""
from __future__ import annotations

from ..common.registry import Registry
from .aad_datamodule import AADDataModule
from .synthetic import SyntheticDataModule
from .windows import WindowSpec

DATA_REGISTRY = Registry("datamodule")
DATA_REGISTRY.register("aad")(AADDataModule)
DATA_REGISTRY.register("synthetic")(SyntheticDataModule)


def build_datamodule(cfg) -> AADDataModule:
    """``cfg`` is the resolved ``data`` config group."""
    if cfg.name == "merit":
        # MERIT owns its own window/candidate/split machinery (src/data/merit_data.py)
        from ..runner.merit_runner import _build_merit_data
        from omegaconf import OmegaConf
        return _build_merit_data(OmegaConf.create({"data": cfg}))
    spec = WindowSpec(
        win_s=float(cfg.window.win_s),
        sr=float(cfg.window.sr),
        n_bands=int(cfg.window.n_bands),
        overlap=float(cfg.window.get("overlap", 0.5)),
        audio_level_norm=bool(cfg.window.get("audio_level_norm", True)),
        audio_norm=str(cfg.window.get("audio_norm", "rms")),
        lp_hz=float(cfg.window.get("lp_hz", 0.0)),
        hp_hz=float(cfg.window.get("hp_hz", 1.0)),
        eeg_lp_hz=float(cfg.window.get("eeg_lp_hz", -1.0)),
        gaze_traj_len=int(cfg.window.get("gaze_traj_len", 16)),
        perfect_align=bool(cfg.window.get("perfect_align", True)),
        norm_cand=bool(cfg.window.get("norm_cand", True)),
        norm_eeg=bool(cfg.window.get("norm_eeg", True)),
        mm_task=str(cfg.window.get("mm_task", "speaker")),
        n_neg=int(cfg.window.get("n_neg", 1)),
        min_shift_s=float(cfg.window.get("min_shift_s", 3.0)),
        audio_feats=bool(cfg.window.get("audio_feats", False)),
        w2v_pca_dim=int(cfg.window.get("w2v_pca_dim", 64)),
    )
    cls = DATA_REGISTRY.get(cfg.name)
    return cls(
        subjects=list(cfg.subjects),
        cache_dir=cfg.cache_dir,
        spec=spec,
        kind=cfg.get("kind", "main"),
        n_folds=int(cfg.get("n_folds", 5)),
        overwrite_cache=bool(cfg.get("overwrite_cache", False)),
        seed=int(cfg.get("seed", 0)),
        split_mode=cfg.get("split_mode", "chrono_forward"),
    )


__init_exports__ = ["build_datamodule", "DATA_REGISTRY"]
