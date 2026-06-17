"""Runner-layer factory."""
from __future__ import annotations

from ..common.registry import Registry
from .aad_runner import AADRunner
from .transfer_runner import TransferSSLRunner

RUNNER_REGISTRY = Registry("runner")
RUNNER_REGISTRY.register("aad")(AADRunner)
RUNNER_REGISTRY.register("transfer_ssl")(TransferSSLRunner)


def build_runner(cfg, paths, wandb, device):
    cls = RUNNER_REGISTRY.get(cfg.runner.get("name", "aad"))
    return cls(cfg, paths, wandb, device)


__all__ = ["RUNNER_REGISTRY", "build_runner"]
