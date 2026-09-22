"""The attended-role prior of each public corpus: how often each talker role is
the attended one.  A corpus with a balanced prior cannot leak the answer through
talker identity; one with an imbalanced prior can, and the audio-only probe then
reads the imbalance.  Written to CSV so the paper's numbers are generated, not
typed.

  python -m src.tools.external_priors --out analysis/results/external
"""
from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import pandas as pd

ROOT = "/fs/ess/PAS2301/Data/EEG"


def kul():
    import scipy.io as sio
    c = collections.Counter()
    for i in range(1, 17):
        p = f"{ROOT}/kuleuven/S{i}.mat"
        if os.path.exists(p):
            for t in np.atleast_1d(sio.loadmat(p, squeeze_me=True,
                                               struct_as_record=False)["trials"]):
                c[int(t.attended_track)] += 1
    return "track 1", c[1], sum(c.values())


def dtu():
    import scipy.io as sio
    c = collections.Counter()
    for i in range(1, 19):
        p = f"{ROOT}/dtu/metadata/S{i}_metadata.mat"
        if os.path.exists(p):
            m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
            for ns, mf in zip(np.atleast_1d(m["n_speakers"]), np.atleast_1d(m["attend_mf"])):
                if int(ns) == 2:
                    c[int(mf)] += 1
    return "male", c[1], sum(c.values())


def nju():
    import scipy.io as sio
    info = sio.loadmat(f"{ROOT}/nju/NJUNCA_preprocessed_arte_removed/"
                       "expinfomat_python_readable.mat", squeeze_me=True,
                       struct_as_record=False)["expinfomat_struct"]
    c = collections.Counter()
    for i in range(len(info)):
        r = np.atleast_1d(info[i])
        if r.size and hasattr(r[0], "_fieldnames"):
            for x in r:
                v = str(x.attended_lr).lower()
                if v in ("left", "right"):
                    c[v] += 1
    return "left", c["left"], c["left"] + c["right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/results/external")
    a = ap.parse_args()
    rows = []
    for name, fn in (("kul", kul), ("dtu", dtu), ("nju", nju)):
        role, k, n = fn()
        rows.append(dict(corpus=name, role=role, n_role=k, n_trials=n, prior=k / n))
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "external_priors.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
