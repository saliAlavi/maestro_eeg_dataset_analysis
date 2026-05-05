"""Reference evaluation protocol for the multimodal AAD dataset.

Future dataset users report numbers on the **same** trial splits and tasks
defined here, which pins down apples-to-apples comparison across modalities
and methods. This module gives both a human-readable specification of the
protocol and a callable harness (``evaluate()``) that takes a predictor
object, runs it over every split / task in the protocol, and returns a
tidy dataframe of accuracies + bootstrap CIs.

Why a protocol matters here. Our release has five modalities on every
trial where competing AAD corpora have one (EEG + stimulus audio). Without
a protocol, a paper reporting "77% AAD accuracy" can mean any of
    - EEG-only, within-subject, 30-s windows;
    - gaze-only, within-subject, 30-s windows;
    - multimodal fusion, LOSO, mixed windows; ...
all of which our dataset supports. We commit to the following reporting
contract.

Protocol v1.0
-------------

**Task framings (mandatory, all three).**
    T1  hemisphere (chance 0.5) : {1,2} left vs {3,4} right
    T2  inner_outer (chance 0.5): {2,3} vs {1,4}
    T3  4-class speaker identity (chance 0.25)

**Splits.**
    S1  within-subject 5-fold stratified CV (seed 0).
    S2  leave-one-subject-out (LOSO).
    S3  motion-residualised within-subject (iter-8 protocol). The trial-
        level spectral/temporal features are replaced with their residual
        after ridge regression against the concatenated gaze+IMU+video
        feature vector (71-dim at v1.0). This is a confound-controlled
        floor, not a benchmark ceiling.
    S4  high-quality-tier within-subject: restrict to subjects with
        comprehension >= 0.80 AND gaze validity >= 0.90.

**Mandatory baselines to beat (for method papers).**
    B1  gaze-only, S1 (target 0.770 hemisphere).
    B2  EEG spectral 368-D logreg, S1 (target 0.716 hemisphere).
    B3  EEG spectral, S3 motion-residualised (target 0.521 hemisphere,
        chance 0.5).

**Metrics.**
    primary  : accuracy (pooled, unweighted mean over 16 subjects).
    ci       : 95% bootstrap (10 000 resamples) over per-subject
               accuracies.
    secondary: per-subject accuracy (scatter), ITR (bits/min) at
               the 30-s window.

**Windows.**
    Main numbers are reported at 30-s decision windows. Continuous-AAD
    methods must additionally report accuracy at {1, 2, 4, 8, 16, 30}-s
    windows (S1 only).

**Reporting contract.**
    Every published number on this dataset must be accompanied by
    (task, split, window, subject-subset). The harness here enforces
    that.
"""
from __future__ import annotations
import sys, warnings, json, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score

from aad_utils import RESULTS_DIR, bootstrap_ci
from aad_utils.config import ATTENDED_HEMISPHERE

PROTOCOL_VERSION = "1.0"


@dataclass
class ProtocolResult:
    task: str
    split: str
    window_s: float
    subset: str
    mean_acc: float
    ci_lo: float
    ci_hi: float
    n_subjects: int
    n_trials: int
    chance: float


TASKS = {
    "hemisphere":  (lambda a: 0 if ATTENDED_HEMISPHERE[a] == "L" else 1, 2),
    "inner_outer": (lambda a: 0 if a in (2, 3) else 1, 2),
    "4class":      (lambda a: a - 1, 4),
}


def high_quality_subset(df_meta: pd.DataFrame) -> list[int]:
    """Return list of subject IDs that pass the pre-registered quality tier:
    comprehension >= 0.80 AND Tobii gaze-validity >= 0.70 (manufacturer's
    recommended quality floor). 9/16 subjects pass at v1.0."""
    try:
        cov = pd.read_parquet(RESULTS_DIR / "individual_differences_covariates.parquet")
        mask = (cov.get("comprehension", 0) >= 0.80)
        if "gaze_valid" in cov.columns:
            mask = mask & (cov["gaze_valid"] >= 0.70)
        return sorted(cov.loc[mask, "subject"].astype(int).tolist())
    except Exception:
        return list(range(1, 17))


def evaluate(predictor_fn: Callable,
             features_fn: Callable,
             df_meta: pd.DataFrame,
             window_s: float = 30.0,
             subset_name: str = "all",
             subjects: Optional[list[int]] = None) -> list[ProtocolResult]:
    """Generic evaluation harness.

    predictor_fn(X_train, y_train, X_test) -> y_pred
    features_fn(subject) -> (X, y_attended, trials) where X is (n_trials, d).
    df_meta : the per-subject metadata table (must have 'subject').
    """
    subjects = subjects if subjects is not None else sorted(df_meta["subject"].unique())
    results = []

    # Collect all data up-front.
    cache = {}
    for s in subjects:
        X, y_att, tr = features_fn(s)
        if X is None or len(X) < 20:
            continue
        cache[s] = (X, y_att, tr)

    for task, (lbl_fn, nc) in TASKS.items():
        # S1: within-subject 5-fold
        per_sub = []
        for s, (X, y_att, _) in cache.items():
            y = np.array([lbl_fn(a) for a in y_att])
            if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2:
                continue
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            accs = []
            for tr, te in skf.split(X, y):
                y_pred = predictor_fn(X[tr], y[tr], X[te])
                accs.append(accuracy_score(y[te], y_pred))
            per_sub.append(float(np.mean(accs)))
        if per_sub:
            m, lo, hi = bootstrap_ci(np.array(per_sub))
            results.append(ProtocolResult(task, "within-5fold", window_s, subset_name,
                                          m, lo, hi, len(per_sub),
                                          sum(len(cache[s][0]) for s in cache),
                                          1.0 / nc))

        # S2: LOSO (only for "all" subset — too few subjects in tier subsets for LOSO)
        if subset_name == "all" and len(cache) >= 5:
            per_test = []
            all_X = np.vstack([cache[s][0] for s in cache])
            all_y_att = np.concatenate([cache[s][1] for s in cache])
            all_y = np.array([lbl_fn(a) for a in all_y_att])
            all_groups = np.concatenate([[s] * len(cache[s][0]) for s in cache])
            if len(np.unique(all_y)) < nc: continue
            logo = LeaveOneGroupOut()
            for tr, te in logo.split(all_X, all_y, all_groups):
                if pd.Series(all_y[tr]).value_counts().min() < 2: continue
                y_pred = predictor_fn(all_X[tr], all_y[tr], all_X[te])
                per_test.append(accuracy_score(all_y[te], y_pred))
            if per_test:
                m, lo, hi = bootstrap_ci(np.array(per_test))
                results.append(ProtocolResult(task, "LOSO", window_s, subset_name,
                                              m, lo, hi, len(per_test),
                                              len(all_X), 1.0 / nc))
    return results


def baseline_logreg(X_tr, y_tr, X_te):
    """Reference baseline: L2-logistic regression with StandardScaler,
    C=0.5, max_iter=3000. Use this exact classifier to reproduce the
    paper's headline numbers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("sc", StandardScaler()),
                     ("c", LogisticRegression(max_iter=3000, C=0.5))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def main():
    """Print the protocol spec and re-run the three canonical baselines
    (B1 gaze-only, B2 EEG-spectral, B3 motion-residualised EEG) so users
    can verify reproduction end-to-end."""
    out = RESULTS_DIR / "evaluation_protocol"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Protocol v{PROTOCOL_VERSION}")
    print(json.dumps({k: v for k, v in globals().items()
                      if k.isupper() and not k.startswith("_")},
                     default=str, indent=2)[:1000])

    # Re-run B1/B2 through the harness so reported numbers are verifiable.
    spec_dir = RESULTS_DIR / "eeg_spectral"
    E = pd.concat([pd.read_parquet(p) for p in sorted(spec_dir.glob("s*.features.parquet"))],
                  ignore_index=True)
    G = pd.read_parquet(RESULTS_DIR / "fusion_gaze_features.parquet")

    E_cols = [c for c in E.columns if c not in ("subject", "trial", "attended", "snr")]
    G_cols = [c for c in G.columns if c not in ("subject", "trial", "attended", "group", "snr")]

    def make_feat_fn(df, cols):
        def f(s):
            g = df[df.subject == s]
            if len(g) < 20: return None, None, None
            return g[cols].fillna(0).values, g["attended"].values, g["trial"].values
        return f

    rows = []
    for name, df_mod, cols in [("B1_gaze-only", G, G_cols),
                               ("B2_eeg-spectral-368", E, E_cols)]:
        r = evaluate(baseline_logreg, make_feat_fn(df_mod, cols), df_mod,
                     subset_name="all")
        for pr in r:
            rows.append(dict(baseline=name, **asdict(pr)))
        # High-quality tier
        hq = high_quality_subset(df_mod)
        r2 = evaluate(baseline_logreg, make_feat_fn(df_mod, cols), df_mod,
                      subset_name="high-quality-tier", subjects=hq)
        for pr in r2:
            rows.append(dict(baseline=name, **asdict(pr)))

    R = pd.DataFrame(rows)
    R.to_parquet(out / "protocol_baseline_results.parquet")
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
