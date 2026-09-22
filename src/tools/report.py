"""Quick console tables from the MERIT runs.

    python -m src.tools.report            # everything, one block per experiment
    python -m src.tools.report --tag hinge --streams
"""
from __future__ import annotations

import argparse

import pandas as pd

from .make_macros import PREF, STREAMS, load

CORE = [("acc", "acc"), ("null_mean", "null"), ("contribution", "contrib"),
        ("p_perm", "p"), ("flip_rate", "flip"), ("emb_cos_centered", "collapse"),
        ("zeros_all", "zeros")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--streams", action="store_true",
                    help="show per-stream Shapley credit instead of the core battery")
    a = ap.parse_args()
    df = load()
    if df.empty:
        print("no results yet")
        return
    if a.tag:
        df = df[df["tag"].str.contains(a.tag)]
    pd.set_option("display.width", 220)
    for (tag, protocol), g in df.groupby(["tag", "protocol"]):
        cols, names = [], []
        if a.streams:
            for m in STREAMS:
                c = PREF + f"shapley_{m}"
                if c in g and g[c].notna().any():
                    cols.append(c); names.append(f"phi/{m}")
            cols.append(PREF + "contribution"); names.append("total")
        else:
            for c, n in CORE:
                if PREF + c in g:
                    cols.append(PREF + c); names.append(n)
        if not cols:
            continue
        t = g.groupby("variant")[cols].agg(["mean", "std", "count"])
        t.columns = [f"{names[cols.index(c)]}.{s[:3]}" for c, s in t.columns]
        keep = [c for c in t.columns if not c.endswith(".cou")] + \
               [[c for c in t.columns if c.endswith(".cou")][0]]
        print(f"\n### {tag or '(untagged)'} / {protocol}  "
              f"(window {g['window_sec'].iloc[0]:g}s, {g['cand_mode'].iloc[0]}, "
              f"K={int(g['K'].iloc[0])}, probe={g['audio_only_probe'].iloc[0]:.4f})")
        print(t[keep].round(4).to_string())


if __name__ == "__main__":
    main()
