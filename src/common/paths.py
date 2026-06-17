"""Canonical scratch paths for the project.

All heavy artefacts live on scratch, never in the git repo:

    /fs/scratch/PAS2301/alialavi/projects/
        {project}/           <- logs, models, run outputs for this project
            logs/  models/  runs/
        cache/               <- preprocessed / cached datasets, shared across projects
        datasets/            <- raw-ish dataset materialisations

The project name is fixed by config (default ``multimodal_aad``). Everything is
created on demand so a fresh checkout / fresh node just works.
"""
from __future__ import annotations

from pathlib import Path

SCRATCH_PROJECTS = Path("/fs/scratch/PAS2301/alialavi/projects")
CACHE_DIR = SCRATCH_PROJECTS / "cache"
DATASETS_DIR = SCRATCH_PROJECTS / "datasets"


class ProjectPaths:
    """Resolve and create the per-project scratch layout."""

    def __init__(self, project: str = "multimodal_aad"):
        self.project = project
        self.root = SCRATCH_PROJECTS / project
        self.logs = self.root / "logs"
        self.models = self.root / "models"
        self.runs = self.root / "runs"
        self.cache = CACHE_DIR
        self.datasets = DATASETS_DIR

    def ensure(self) -> "ProjectPaths":
        for d in (self.logs, self.models, self.runs, self.cache, self.datasets):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, run_name: str) -> Path:
        # Per CLAUDE.md: run artefacts live at
        #   /fs/scratch/PAS2301/alialavi/projects/{run_name}
        # where run_name is the EXACT wandb run name ({project}__{model}__{stamp}),
        # so a wandb run maps 1:1 to a local directory.
        d = SCRATCH_PROJECTS / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_dir(self, run_name: str) -> Path:
        d = self.run_dir(run_name) / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def __repr__(self) -> str:
        return f"ProjectPaths(project={self.project!r}, root={self.root})"
