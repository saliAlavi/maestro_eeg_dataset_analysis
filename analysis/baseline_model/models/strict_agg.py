"""Aggregate strict_improve results: per config, real vs EEG-shuffle null (paired), and the
paired margin difference between configs (e.g. onset vs env)."""
import glob, json, sys
from collections import defaultdict
import numpy as np
from scipy import stats

RR = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad/results"


def load(proto):
    by = defaultdict(dict)                       # subject -> config -> row
    for pj in glob.glob(f"{RR}/strict_improve_{proto}/s*.json"):
        s = int(pj.split("/s")[-1].split(".")[0])
        for r in json.load(open(pj)):
            by[s][r["config"]] = r
    return by


def main():
    for proto in ("within", "loso"):
        by = load(proto)
        if not by:
            continue
        cfgs = sorted({c for s in by for c in by[s]})
        print(f"\n=== {proto.upper()} ({len(by)} subj) ===")
        print(f"{'config':16s} {'real':>7} {'null':>7} {'margin':>8} {'t(vs null)':>11} {'p':>9}")
        marg = {}
        for c in cfgs:
            subs = [s for s in by if c in by[s]]
            R = np.array([by[s][c]["real"] for s in subs]); N = np.array([by[s][c]["null"] for s in subs])
            marg[c] = {s: by[s][c]["real"] - by[s][c]["null"] for s in subs}
            t, p = stats.ttest_rel(R, N); p1 = p / 2 if t > 0 else 1 - p / 2
            print(f"{c:16s} {R.mean():7.3f} {N.mean():7.3f} {R.mean()-N.mean():+8.3f} {t:11.2f} {p1:9.1e}")
        # paired margin comparison vs 'env'
        if "env" in cfgs:
            print("  paired margin vs env (one-sided t, +=better):")
            for c in cfgs:
                if c == "env":
                    continue
                subs = [s for s in by if c in marg and s in marg[c] and s in marg["env"]]
                a = np.array([marg[c][s] for s in subs]); b = np.array([marg["env"][s] for s in subs])
                t, p = stats.ttest_rel(a, b); p1 = p / 2 if t > 0 else 1 - p / 2
                print(f"    {c:14s} Δmargin={a.mean()-b.mean():+.3f}  t={t:.2f} p1={p1:.3f}")


if __name__ == "__main__":
    main()
