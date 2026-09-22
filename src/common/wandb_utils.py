"""Weights & Biases integration.

Run-name contract (requested by the project owner):

    {project_name}__{model_version}__{YYYYmmdd-HHMMSS}[-j{job}t{task}]

The optional suffix is the SLURM job and array-task id.  Without it, array tasks
launched in the same second receive the same name, hence the same run
directory, and overwrite one another's results -- which is how four groups of
the 2 s / 5 s modality sweep were lost.  It contains no double underscore, so
the tag is still recoverable from the directory name.

If wandb is unavailable or disabled we return a no-op shim so the rest of the
code never has to branch on ``if wandb_run is not None``.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict


def make_run_name(project: str, model_version: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job and task is not None:
        stamp += f"-j{job}t{task}"
    elif job:
        stamp += f"-j{job}"
    else:
        stamp += f"-p{os.getpid()}"       # two local runs in one second
    return f"{project}__{model_version}__{stamp}"


class _NoOpRun:
    """Drop-in replacement for a wandb run when logging is off."""

    def __init__(self, name: str):
        self.name = name
        self.url = "(wandb disabled)"

    def log(self, *a, **k):
        pass

    def summary_update(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


class WandbLogger:
    """Thin wrapper so the runner/models log through one constant interface."""

    def __init__(self, run, enabled: bool):
        self.run = run
        self.enabled = enabled
        self.name = getattr(run, "name", "noop")

    @classmethod
    def init(
        cls,
        *,
        project: str,
        model_version: str,
        config: Dict[str, Any],
        mode: str = "online",
        entity: str | None = None,
        group: str | None = None,
    ) -> "WandbLogger":
        run_name = make_run_name(project, model_version)
        if mode == "disabled":
            return cls(_NoOpRun(run_name), enabled=False)
        try:
            import wandb

            # Run files go to scratch, never into the repository.  WANDB_DIR
            # overrides the default for anyone running elsewhere.
            wdir = os.environ.get(
                "WANDB_DIR", "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__wandb")
            os.makedirs(wdir, exist_ok=True)
            run = wandb.init(
                project=project,
                name=run_name,
                dir=wdir,
                entity=entity,
                group=group,
                config=config,
                mode=mode,
                reinit=True,
            )
            return cls(run, enabled=True)
        except Exception as exc:  # never let logging kill a training job
            print(f"[wandb] init failed ({exc!r}); falling back to no-op", flush=True)
            return cls(_NoOpRun(run_name), enabled=False)

    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        try:
            if self.enabled:
                self.run.log(metrics, step=step)
        except Exception:
            pass

    def summary(self, metrics: Dict[str, Any]) -> None:
        try:
            if self.enabled and hasattr(self.run, "summary"):
                for k, v in metrics.items():
                    self.run.summary[k] = v
        except Exception:
            pass

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception:
            pass
