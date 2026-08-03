"""Shared CLI for the per-experiment scripts (train_hemisphere/eccentricity/
pooled/reconstruction). One flag, ``--data-method``, switches the whole
train/val methodology between our leakage-safe control and the github repo's.

  # our leakage-safe numbers (this run's table, 'ours' columns)
  python train_hemisphere.py --data-method proper --protocol within --subject 3
  python train_hemisphere.py --data-method proper --protocol loso   --subject 3

  # the github repo's methodology (reproduces the inflated 'gh (repo)' column)
  python train_hemisphere.py --data-method github --protocol pooled
  python train_pooled.py     --data-method github --protocol loso    --subject 3
"""
from __future__ import annotations

import argparse
import logging

import gh_core


def main(task: str, default_modes=("eeg", "gaze", "imu")):
    ap = argparse.ArgumentParser(description=f"n_gh_checks experiment: {task}")
    ap.add_argument("--data-method", choices=["proper", "github"], default="proper",
                    help="proper = our leakage-safe control; github = repo methodology")
    ap.add_argument("--protocol", default=None,
                    help="proper: within|loso (default within). github: pooled|loso|within (default pooled)")
    ap.add_argument("--subject", type=int, default=None,
                    help="within: subject id (1-16); loso: held-out TEST subject; pooled: ignored")
    ap.add_argument("--modes", nargs="*", default=list(default_modes),
                    help="modalities (classification): any of eeg gaze imu")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    method = a.data_method
    protocol = a.protocol or ("within" if method == "proper" else "pooled")
    if method == "proper" and protocol not in ("within", "loso"):
        raise SystemExit("proper method supports --protocol within|loso")
    if method == "github" and protocol not in ("pooled", "loso", "within"):
        raise SystemExit("github method supports --protocol pooled|loso|within")
    if protocol in ("within", "loso") and a.subject is None:
        raise SystemExit(f"--subject is required for --protocol {protocol}")

    gh_core.run_experiment(task, method, protocol, a.modes, subject=a.subject,
                           epochs=a.epochs, save_ckpt=not a.no_ckpt,
                           skip_existing=a.skip_existing)
