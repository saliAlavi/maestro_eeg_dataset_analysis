"""Run the audio-only acceptance probe on the public two-talker AAD corpora.

This is the model-independent half of \\merit{} applied to data it was not
designed on.  A candidate construction free of acoustic confounding must land at
1/K; anything above that is accuracy a decoder can take without ever consulting
the recording.  Running it on corpora built by other groups, with other talkers,
other rooms and other presentation hardware, is what turns the probe from an
observation about one dataset into an instrument.

  python -m src.tools.external_probe --window 10 --out analysis/results/external
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from ..data.merit_data import audio_only_probe, quantile_match
from ..data.external_aad import FS, LOADERS

log = logging.getLogger("tools.external_probe")


def _zs(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def windows(trials, win_sec, hop_sec, max_per_trial=None, seed=0):
    """(N,2,T) candidates, labels, content groups, subject ids.

    Slot order is randomised per window from a fixed seed, so slot index carries
    no information and every corpus is scored on the same convention as the main
    experiments.
    """
    W, H = int(win_sec * FS), int(hop_sec * FS)
    rng = np.random.default_rng(seed)
    C, y, g, s = [], [], [], []
    for t in trials:
        n = min(len(t.att), len(t.unatt))
        offs = list(range(0, n - W + 1, H))
        if max_per_trial and len(offs) > max_per_trial:
            offs = list(rng.choice(offs, max_per_trial, replace=False))
        for o in offs:
            a, u = t.att[o:o + W], t.unatt[o:o + W]
            k = int(rng.integers(0, 2))                  # which slot holds the target
            pair = (a, u) if k == 0 else (u, a)
            C.append(np.stack(pair))
            y.append(k)
            g.append(t.content)
            s.append(f"{t.corpus}/{t.subject}")
    if not C:
        return None
    C = np.stack(C).astype(np.float32)
    # A window in which either candidate is silent has no shape to measure and
    # standardising it produces non-finite statistics; drop it rather than let
    # it decide a fold.
    ok = (C.std(-1) > 1e-6).all(1)
    C, y, g, s = C[ok], np.array(y)[ok], list(np.array(g)[ok]), np.array(s)[ok]
    C = _zs(C)
    ok2 = np.isfinite(C).all((1, 2))
    C, y, g, s = C[ok2], y[ok2], list(np.array(g)[ok2]), s[ok2]
    if len(y) == 0:
        return None
    gi = pd.factorize(pd.Series(g))[0]
    return C, y, gi, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=["kul", "dtu", "nju"])
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--max-per-trial", type=int, default=40)
    ap.add_argument("--out", default="analysis/results/external")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for name in a.corpora:
        log.info("loading %s ...", name)
        trials = list(LOADERS[name]())
        log.info("%s: %d trials, %d listeners", name, len(trials),
                 len({t.subject for t in trials}))
        w = windows(trials, a.window, a.hop, a.max_per_trial)
        if w is None:
            log.warning("%s: no windows", name)
            continue
        C, y, g, s = w
        p_raw = audio_only_probe(C, y, g)
        p_qm = audio_only_probe(quantile_match(C), y, g)
        rows.append(dict(corpus=name, n_windows=len(y),
                         n_listeners=len(set(s)), n_content=len(set(g)),
                         window_sec=a.window, K=2, chance=0.5,
                         probe_raw=p_raw, excess_raw=p_raw - 0.5,
                         probe_qmatch=p_qm, excess_qmatch=p_qm - 0.5))
        log.info("%-4s  windows=%-6d probe=%.4f (excess %+.4f)  qmatch=%.4f",
                 name, len(y), p_raw, p_raw - 0.5, p_qm)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "external_probes.csv"), index=False)
    df.to_parquet(os.path.join(a.out, "external_probes.parquet"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
