"""Turn measured results into LaTeX macros, so no number in the paper is typed.

Every result cell in the manuscript is written ``\\R{key}``.  This script scans
the run directories on scratch, aggregates over folds, and emits one
``\\defres{key}{value}`` per cell.  A key with no measurement renders as a red
``??`` in the compiled PDF, which is the point: an unfilled cell is visible.

    python -m src.tools.make_macros --out docs/iclr2027/results_macros.tex
"""
from __future__ import annotations

import argparse
import glob
import re
import os

import numpy as np
import pandas as pd

# The model was called CREDIT before it was renamed MERIT; runs made under either
# name are the same model and are read together.  Scratch is purged, so the same
# files are also read from the repository snapshot kept by
# src.tools.snapshot_results; duplicated (tag, variant, protocol, split) rows
# carry the same run_name and collapse in load().
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(REPO, "docs", "iclr2027", "results")
RUNS = ["/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__credit__*/results.parquet",
        "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__merit__*/results.parquet",
        os.path.join(SNAP, "runs", "*", "results.parquet")]
PROJ = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad"


def find_aux(name):
    """A per-experiment csv: the live scratch copy when present, else the snapshot."""
    for d in (PROJ, SNAP):
        f = os.path.join(d, name)
        if os.path.exists(f):
            return f
    return None

PREF = "test/selected/"
PREF_ACC = "test/sel_acc/"
STREAMS = ["eeg", "gaze", "imu", "video", "fovea", "orient"]


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return f"{x:.{nd}f}"


def _signed(x, nd=4):
    f = _fmt(x, nd)
    return None if f is None else (f"+{f}" if x >= 0 else f)


def _tag_from_path(f):
    """Recover the experiment tag from the run directory.

    Run directories are named ``{project}__{model}__{tag}__{stamp}`` (the wandb
    run name), so the tag is recoverable even for runs whose result rows predate
    the in-row ``tag`` column.  Without it, two experiments that share variant
    names -- the ladder on raw and on matched candidates, or the lag control at
    K=4 and K=2 -- would silently merge.
    """
    d = os.path.basename(os.path.dirname(f))
    p = re.sub(r"^multimodal_aad__(credit|merit)__", "", d)
    parts = p.rsplit("__", 1)
    return parts[0] if len(parts) == 2 else ""


def load(pattern=RUNS):
    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    fs = sorted(f for pat in pats for f in glob.glob(pat))
    if not fs:
        return pd.DataFrame()
    parts = []
    for f in fs:
        d = pd.read_parquet(f)
        if d.empty:
            continue
        t = _tag_from_path(f)
        if "tag" not in d.columns:
            d["tag"] = t
        else:
            d["tag"] = d["tag"].fillna("").astype(str).replace("", t)
        d["_src"] = f
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    for c, d in (("tag", ""), ("variant", "credit"), ("run_name", ""),
                 ("cand_mode", "qmatch"), ("K", 4), ("window_sec", 10.0)):
        if c not in df.columns:
            df[c] = d
    # LOSO runs one array task per held-out listener (tags loso_f0..loso_f15);
    # fold those back into a single "loso" group so folds aggregate.
    df["tag"] = (df["tag"].fillna("").astype(str)
                 .str.replace(r"_f\d+$", "", regex=True))
    # Every published run sets model.tag; an untagged run directory is a smoke
    # test or an interactive probe, never a result.
    df = df[~df["tag"].isin(["smoke", "timing", ""])]
    if df.empty:
        return df
    # a tag+variant+protocol+split may be re-run; keep the most recent run_name
    if "run_name" in df.columns:
        df = df.sort_values("run_name").drop_duplicates(
            ["tag", "variant", "protocol", "split", "window_sec", "cand_mode", "K"],
            keep="last")
    return df


def emit(df, probes=None, sep=None):
    out = []

    def put(key, val):
        if val is not None:
            out.append(f"\\defres{{{key}}}{{{val}}}")

    if not df.empty:
        gcols = ["tag", "variant", "protocol"]
        # A group still being written by a running job has fewer folds than its
        # siblings.  Emitting it would print a partial average as if complete, so
        # it is skipped and its cells render as ?? until the run finishes.
        full = df.groupby(["tag", "protocol"])["split"].nunique()
        for keys, g in df.groupby(gcols):
            tag, variant, protocol = [str(k) for k in keys]
            base = f"{tag}/{variant}/{protocol}"
            n = len(g)
            expect = 5 if protocol == "within" else int(full.loc[(tag, protocol)])
            if n < expect:
                print(f"[incomplete] {base}: {n}/{expect} folds -- not emitted")
                continue
            put(f"{base}/nfolds", str(n))
            put(f"{base}/window", _fmt(g["window_sec"].iloc[0], 0))
            put(f"{base}/cand", str(g["cand_mode"].iloc[0]))
            put(f"{base}/K", str(int(g["K"].iloc[0])))
            put(f"{base}/probe", _fmt(g["audio_only_probe"].iloc[0]))
            # the parameter count is a property of the architecture, not of the
            # checkpoint selection, so it carries no selection prefix
            for c in ("n_params", "test/n_params"):
                if c in g and g[c].notna().any():
                    put(f"{base}/params", f"{int(g[c].dropna().iloc[0])}")
                    break
            for pref, sfx in ((PREF, ""), (PREF_ACC, "/accsel")):
                for col, key, fn in (("acc", "acc", _fmt), ("null_mean", "null", _fmt),
                                     ("contribution", "contrib", _signed),
                                     ("p_perm", "p", _fmt),
                                     ("flip_rate", "flip", _fmt),
                                     ("zeros_all", "zeros", _fmt),
                                     ("emb_cos_centered", "collapse", _fmt),
                                     ("chance", "chance", _fmt),
                                     ("contribution_trial", "contribtrial", _signed),
                                     ("contribution_pos", "contribpos", _signed)):
                    c = pref + col
                    if c in g and g[c].notna().any():
                        put(f"{base}/{key}{sfx}", fn(float(g[c].dropna().mean())))
                # two-decimal copies for prose (abstract, introduction), still
                # generated from the run outputs rather than rounded by hand
                for col, key, fn in (("acc", "acc2", lambda x: _fmt(x, 2)),
                                     ("contribution", "contrib2", lambda x: _signed(x, 2))):
                    c = pref + col
                    if c in g and g[c].notna().any():
                        put(f"{base}/{key}{sfx}", fn(float(g[c].dropna().mean())))
                c = pref + "acc"
                if c in g and g[c].notna().any():
                    put(f"{base}/accsd{sfx}", _fmt(float(g[c].dropna().std(ddof=1))
                                                   if n > 1 else 0.0))
                c = pref + "contribution"
                if c in g and g[c].notna().any():
                    v = g[c].dropna()
                    put(f"{base}/contribsd{sfx}", _fmt(float(v.std(ddof=1))
                                                       if n > 1 else 0.0))
                    put(f"{base}/pos{sfx}", f"{int((v > 0).sum())}/{len(v)}")
            for m in STREAMS:
                for col, key, fn in ((f"shapley_{m}", f"shap/{m}", _signed),
                                     (f"contrib_{m}", f"contrib/{m}", _signed),
                                     (f"lomo_{m}", f"lomo/{m}", _signed)):
                    c = PREF + col
                    if c in g and g[c].notna().any():
                        v = g[c].dropna()
                        put(f"{base}/{key}", fn(float(v.mean())))
                        put(f"{base}/{key}/pos", f"{int((v > 0).sum())}/{len(v)}")
    if probes is not None:
        for _, r in probes.iterrows():
            k = f"probe/{r['construction']}/K{int(r['K'])}"
            put(k, _fmt(r["audio_only_probe"]))
            put(k + "/chance", _fmt(r["chance"]))
            put(k + "/excess", _signed(r["excess"]))
    if sep is not None:
        for _, r in sep.iterrows():
            k = "sep/" + str(r["statistic"]).replace(" ", "").replace(".", "").replace(
                "-", "").replace("(", "").replace(")", "")
            put(k + "/auc", _fmt(r["auc"], 3))
            put(k + "/d", _signed(r["cohen_d"], 2))
            put(k + "/target", _fmt(r["target_mean"], 2))
            put(k + "/masker", _fmt(r["masker_mean"], 2))
    return out


HEADER = r"""% GENERATED by src/tools/make_macros.py -- do not edit by hand.
% Every number in the manuscript is written \R{key} and resolved here.
% A key with no measurement renders as a red ?? in the PDF.
\makeatletter
\providecommand{\defres}[2]{\expandafter\gdef\csname res@#1\endcsname{#2}}
\providecommand{\R}[1]{\ifcsname res@#1\endcsname\csname res@#1\endcsname
  \else\textcolor{red}{\texttt{??}}\fi}
\makeatother
"""



EXT = os.path.join(REPO, "analysis", "results", "external")


def _find_ext(name):
    for d in (EXT, os.path.join(SNAP, "external")):
        f = os.path.join(d, name)
        if os.path.exists(f):
            return f
    return None


def emit_external():
    """Audio-probe results and attended-role priors on the public corpora.

    These come from ``src.tools.external_probe`` and ``src.tools.external_priors``
    rather than from a training run, but they are measurements all the same and
    go through the same macro path as everything else.
    """
    out = []
    f = _find_ext("external_probe_sweep.csv")
    if f:
        for _, r in pd.read_csv(f).iterrows():
            k = f"extprobe/{r['corpus']}/w{float(r['window_sec']):g}"
            out += [f"\\defres{{{k}/probe}}{{{_fmt(r['probe_raw'])}}}",
                    f"\\defres{{{k}/excess}}{{{_signed(r['excess_raw'])}}}",
                    f"\\defres{{{k}/qmatch}}{{{_fmt(r['probe_qmatch'])}}}",
                    f"\\defres{{{k}/qmexcess}}{{{_signed(r['excess_qmatch'])}}}",
                    f"\\defres{{{k}/nwin}}{{{int(r['n_windows']):,}}}".replace(",", "{,}"),
                    f"\\defres{{{k}/nlisteners}}{{{int(r['n_listeners'])}}}"]
    f = _find_ext("external_priors.csv")
    if f:
        for _, r in pd.read_csv(f).iterrows():
            k = f"extprior/{r['corpus']}"
            out += [f"\\defres{{{k}/prior}}{{{_fmt(r['prior'], 3)}}}",
                    f"\\defres{{{k}/role}}{{{r['role']}}}",
                    f"\\defres{{{k}/nrole}}{{{int(r['n_role'])}}}",
                    f"\\defres{{{k}/ntrials}}{{{int(r['n_trials'])}}}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=RUNS)
    ap.add_argument("--out", default="docs/iclr2027/results_macros.tex")
    a = ap.parse_args()
    df = load(a.pattern)
    probes = sep = None
    f = find_aux("credit_candidate_probes.csv")
    if f:
        probes = pd.read_csv(f)
    f = find_aux("credit_role_separation.csv")
    if f:
        sep = pd.read_csv(f)
    lines = emit(df, probes, sep)
    lines += emit_external()
    with open(a.out, "w") as fh:
        fh.write(HEADER + "\n".join(sorted(lines)) + "\n")
    print(f"{len(lines)} macros -> {a.out}  ({len(df)} result rows)")
    if not df.empty:
        print(df.groupby(["tag", "variant", "protocol"]).size().to_string())


if __name__ == "__main__":
    main()
