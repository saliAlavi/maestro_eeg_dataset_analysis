"""Snapshot the measured run outputs from scratch into the repository.

Scratch on OSC is purged, but every number in the ICLR paper is derived from
``results.parquet`` files that live there.  This copies them (a few MB) into
``docs/iclr2027/results/`` so the paper, its macros and its figures can be
rebuilt from the repository alone.  ``make_macros`` and ``make_figures`` read
the live scratch directories when they exist and fall back to this snapshot.

    python -m src.tools.snapshot_results
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(REPO, "docs", "iclr2027", "results")

SCRATCH_RUNS = ["/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__credit__*",
                "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__merit__*"]
SCRATCH_PROJ = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad"
PROJ_CSVS = ["credit_candidate_probes.csv", "credit_role_separation.csv"]
EXTERNAL = os.path.join(REPO, "analysis", "results", "external")


def snapshot(verbose=True):
    n_runs = n_csv = 0
    for pat in SCRATCH_RUNS:
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, "results.parquet")
            # smoke tests and timing probes are never results (make_macros drops
            # them by tag too; not copying them keeps the snapshot legible)
            if not os.path.exists(f) or "__smoke__" in d or "__timing__" in d:
                continue
            dst = os.path.join(SNAP, "runs", os.path.basename(d))
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(f, os.path.join(dst, "results.parquet"))
            n_runs += 1
    for name in PROJ_CSVS:
        f = os.path.join(SCRATCH_PROJ, name)
        if os.path.exists(f):
            os.makedirs(SNAP, exist_ok=True)
            shutil.copy2(f, os.path.join(SNAP, name))
            n_csv += 1
    if os.path.isdir(EXTERNAL):
        for f in sorted(glob.glob(os.path.join(EXTERNAL, "**", "*.csv"),
                                  recursive=True)):
            rel = os.path.relpath(f, EXTERNAL)
            dst = os.path.join(SNAP, "external", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(f, dst)
            n_csv += 1
    if verbose:
        print(f"{n_runs} run results + {n_csv} csv files -> {SNAP}")
    return n_runs, n_csv


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    snapshot()


if __name__ == "__main__":
    main()
