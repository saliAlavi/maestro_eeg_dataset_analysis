"""Aggregate all n_gh_checks result JSONs into a summary table + comparison.

Collapses per-fold / per-subject results into (protocol, task, mode) rows with
mean +/- std and a bootstrap 95% CI across subjects, and prints a side-by-side
comparison against the github repo's reported (optimistic, best-epoch-on-test)
headline numbers so the effect of the leakage fixes is visible at a glance.

Writes: results/summary_long.csv, results/summary.md (copied into the repo).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aggregate")

RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/n_gh_checks")
REPO_RESULTS = Path(__file__).resolve().parent / "results"

# github repo's reported single-modality headline numbers (verified against the
# repo's stored result JSONs). Classification = best-epoch val-acc on the fold
# used for early stopping (val == test) => optimistic. Kept only for comparison.
GH_REF = {
    ("speaker4", "eeg"): 0.586, ("speaker4", "gaze"): 0.620, ("speaker4", "imu"): 0.568,
    ("hemisphere", "eeg"): 0.781, ("hemisphere", "gaze"): 0.785, ("hemisphere", "imu"): 0.787,
    ("eccentricity", "eeg"): 0.705, ("eccentricity", "gaze"): 0.682, ("eccentricity", "imu"): 0.683,
    ("reconstruction", "eeg"): 0.0030, ("reconstruction", "gaze"): 0.0000,
    ("reconstruction", "imu"): 0.0131, ("reconstruction", "eeg_gaze_imu"): 0.0192,
}
GH_LOSO_REF = {  # github LOSO (train_loso_hot) single-modality, 4-class only
    ("speaker4", "eeg"): 0.569, ("speaker4", "gaze"): 0.602, ("speaker4", "imu"): 0.602,
}


def _metric_key(rows):
    return "test_r" if rows and "test_r" in rows[0] else "test_acc"


def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect():
    recs = []
    for pj in sorted(glob.glob(str(RUN_ROOT / "results" / "*" / "*.json"))):
        p = Path(pj)
        protocol = p.parent.name
        try:
            rows = json.load(open(pj))
        except Exception as e:
            log.warning("bad json %s: %s", pj, e); continue
        if not isinstance(rows, list) or not rows:
            continue
        mkey = _metric_key(rows)
        # group per (task, mode) — one JSON file already is one (subject|task|mode)
        # for singles/recon; fusion_*.json holds many (task,mode,split) rows.
        by = defaultdict(list)
        for r in rows:
            by[(r["task"], r["mode"])].append(r)
        for (task, mode), rs in by.items():
            # per-subject value = mean over that subject's folds (within) / the
            # single value (loso). test_subject identifies the subject.
            per_sub = defaultdict(list)
            for r in rs:
                per_sub[r.get("test_subject")].append(float(r[mkey]))
            subj_vals = [np.mean(v) for v in per_sub.values()]
            for r in rs:
                recs.append(dict(protocol=protocol, task=task, mode=mode,
                                 metric="pearson_r" if mkey == "test_r" else "acc",
                                 chance=r.get("chance", 0.0),
                                 test_subject=r.get("test_subject"), fold=r.get("fold"),
                                 value=float(r[mkey]), best_val=r.get("best_val")))
    return pd.DataFrame(recs)


def summarize(df):
    out = []
    for (protocol, task, mode, metric, chance), sub in df.groupby(
            ["protocol", "task", "mode", "metric", "chance"]):
        per_sub = sub.groupby("test_subject")["value"].mean().values
        per_sub = per_sub[np.isfinite(per_sub)]
        if not len(per_sub):
            continue
        m, lo, hi = _boot_ci(per_sub)
        out.append(dict(protocol=protocol, task=task, mode=mode, metric=metric,
                        mean=m, std=float(per_sub.std()), ci_lo=lo, ci_hi=hi,
                        n_subjects=len(per_sub), chance=chance))
    return pd.DataFrame(out).sort_values(["metric", "task", "protocol", "mode"])


def make_markdown(summ):
    lines = ["# n_gh_checks — leakage-safe MAESTRO reproduction (5 s / 0.5 overlap)\n",
             "Ours = held-out test of best-inner-val checkpoint; trial-level splits; "
             "loudness-matched envelopes. `gh` = repo's reported (optimistic) number.\n"]
    for task in ["speaker4", "hemisphere", "eccentricity", "reconstruction"]:
        t = summ[summ.task == task]
        if t.empty:
            continue
        chance = t["chance"].iloc[0]
        lines.append(f"\n## {task}  (chance={chance:g})\n")
        lines.append("| mode | protocol | ours mean | ±std | 95% CI | n | gh (repo) |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            ref = GH_LOSO_REF.get((task, r["mode"])) if r["protocol"] == "loso" else None
            ref = ref if ref is not None else GH_REF.get((task, r["mode"]))
            refs = f"{ref:.3f}" if ref is not None else "—"
            fmt = "{:.4f}" if r["metric"] == "pearson_r" else "{:.3f}"
            lines.append(f"| {r['mode']} | {r['protocol']} | {fmt.format(r['mean'])} | "
                         f"{fmt.format(r['std'])} | [{fmt.format(r['ci_lo'])}, "
                         f"{fmt.format(r['ci_hi'])}] | {r['n_subjects']} | {refs} |")
    return "\n".join(lines) + "\n"


def main():
    df = collect()
    if df.empty:
        log.warning("no results found under %s", RUN_ROOT / "results"); return
    summ = summarize(df)
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPO_RESULTS / "results_long.csv", index=False)
    summ.to_csv(REPO_RESULTS / "summary_long.csv", index=False)
    md = make_markdown(summ)
    (REPO_RESULTS / "summary.md").write_text(md)
    (RUN_ROOT / "results" / "summary.md").write_text(md)
    print(md)
    log.info("wrote %s and %s", REPO_RESULTS / "summary.md", REPO_RESULTS / "summary_long.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    main()
