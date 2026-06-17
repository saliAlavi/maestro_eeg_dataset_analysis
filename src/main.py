"""CLI entry point. Hydra builds the config; factories build each layer.

Usage:
    python -m src.main mode=train  model=maestro data.subjects=[1,2,3]
    python -m src.main mode=prepare data.subjects=[1,2,3]     # build caches only
    python -m src.main mode=selftest                          # CPU stack smoke

Modes:
    prepare  -- build/cache the windowed dataset for the requested subjects.
    train    -- full protocol sweep for one model (logs to wandb, saves to scratch).
    selftest -- run the whole stack on synthetic data, wandb disabled.
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from .common.logging_utils import get_logger, setup_logging
from .common.paths import ProjectPaths
from .common.seed import seed_everything
from .common.wandb_utils import WandbLogger

log = logging.getLogger("main")


def _prepare(cfg: DictConfig) -> None:
    from .data.factory import build_datamodule
    dm = build_datamodule(cfg.data)
    dm.prepare()
    log.info("prepare done for subjects %s", list(cfg.data.subjects))


def _selftest(cfg: DictConfig) -> None:
    from .data.aad_compat import sanity_check
    log.info("aad_utils path sanity: %s", sanity_check())
    cfg = cfg.copy()
    OmegaConf.set_struct(cfg, False)
    cfg.data.name = "synthetic"
    cfg.wandb.mode = "disabled"
    if "train" in cfg.model:
        cfg.model.train.epochs = min(int(cfg.model.train.get("epochs", 3)), 3)
    cfg.runner.protocols = ["within", "loso"]
    _train(cfg)


def _train(cfg: DictConfig) -> None:
    from .runner.factory import build_runner
    paths = ProjectPaths(cfg.project).ensure()
    wandb = WandbLogger.init(
        project=cfg.project, model_version=cfg.model.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.get("mode", "online"),
        entity=cfg.wandb.get("entity", None),
        group=cfg.wandb.get("group", None),
    )
    log.info("run: %s  (wandb %s)", wandb.name, "on" if wandb.enabled else "off")
    runner = build_runner(cfg, paths, wandb, cfg.runner.get("device", "auto"))
    df = runner.run()
    log.info("done. %d split rows.", 0 if df is None else len(df))
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.get("log_level", "INFO"))
    seed_everything(int(cfg.get("seed", 0)))
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    mode = cfg.get("mode", "train")
    if mode == "prepare":
        _prepare(cfg)
    elif mode == "selftest":
        _selftest(cfg)
    elif mode == "train":
        _train(cfg)
    else:
        raise ValueError(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
