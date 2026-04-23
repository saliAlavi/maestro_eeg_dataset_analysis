"""Per-trial correlation between behavioural comprehension correctness
and decoder correctness.

For every (subject, trial) pair, join:
    - behavioural correctness (02_behavioral_records.parquet, Correct in
      {0, 1})
    - per-trial decoder correctness for each model:
        * aad_v4_4class/s*.parquet   (correct_trial / correct_hemisphere /
                                      correct_inner_outer at window=='full')
        * loso_cca_mel28/s*.parquet  (correct_trial)
        * gaze_residualised/s*.parquet (correct_trial, both conditions)

Emit:
    results/behaviour_decoding.parquet  — long form per (subject, trial,
        model, task, decoder_correct, behav_correct)
    results/behaviour_decoding_pooled.parquet — pooled accuracy conditional
        on behav_correct for each model.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _t2i(x: str) -> int | None:
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None


def load_behaviour() -> pd.DataFrame:
    br = pd.read_parquet(RESULTS / "02_behavioral_records.parquet")
    br = br[(br.is_training == False) & br["Correct"].notna()].copy()
    br["trial"] = br["Trial No."].apply(_t2i)
    br = br.dropna(subset=["trial"]).copy()
    br["trial"] = br["trial"].astype(int)
    br["subject"] = br["subject"].astype(int)
    br["behav_correct"] = br["Correct"].astype(int)
    return br[["subject", "trial", "behav_correct"]]


def load_decoder_rows() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    # 1) aad_v4_4class: 3 tasks at window='full'
    for fp in sorted(glob.glob(str(RESULTS / "aad_v4_4class/s*.parquet"))):
        df = pd.read_parquet(fp)
        full = df[df["window_s"] == "full"].copy()
        if full.empty:
            continue
        full["subject"] = full["subject"].astype(int)
        full["trial"] = full["trial"].astype(int)
        for col, task in [
            ("correct_trial", "4class"),
            ("correct_hemisphere", "hemisphere"),
            ("correct_inner_outer", "inner_outer"),
        ]:
            if col in full.columns:
                tmp = full[["subject", "trial", col]].rename(
                    columns={col: "decoder_correct"}
                )
                tmp["model"] = "cca-4class"
                tmp["task"] = task
                rows.append(tmp)

    # 2) LOSO CCA mel-28
    for fp in sorted(glob.glob(str(RESULTS / "loso_cca_mel28/s*.parquet"))):
        df = pd.read_parquet(fp)
        df["subject"] = df["test_subject"].astype(int)
        df["trial"] = df["trial"].astype(int)
        tmp = df[["subject", "trial", "correct_trial"]].rename(
            columns={"correct_trial": "decoder_correct"}
        )
        tmp["model"] = "loso-cca-mel-28"
        tmp["task"] = "envelope-attendance"
        rows.append(tmp)

    # 3) aad_v3 backbones (per-trial per-fold)
    for fp in sorted(glob.glob(str(RESULTS / "aad_v3/s*_*_derivative.parquet"))):
        df = pd.read_parquet(fp)
        if "window_s" in df.columns:
            df = df[df["window_s"] == "full"].copy()
        if df.empty:
            continue
        df["subject"] = df["subject"].astype(int)
        df["trial"] = df["trial"].astype(int)
        df["envelope_margin"] = df["rho_att"] - df["rho_una"]
        # Collapse 5 folds per (subject, trial) to a mean correctness
        agg = df.groupby(["subject", "trial", "features"], as_index=False).agg(
            decoder_correct=("correct_trial", "mean"),
            envelope_margin=("envelope_margin", "mean"),
        )
        for feat, sub in agg.groupby("features"):
            tmp = sub[["subject", "trial", "decoder_correct", "envelope_margin"]].copy()
            tmp["model"] = f"backward-trf-{feat}"
            tmp["task"] = "envelope-attendance"
            rows.append(tmp)

    # 4) gaze_residualised (within-subject 5-fold, two conditions)
    for fp in sorted(glob.glob(str(RESULTS / "gaze_residualised/s*.parquet"))):
        df = pd.read_parquet(fp)
        df["subject"] = df["subject"].astype(int)
        df["trial"] = df["trial"].astype(int)
        # average correct_trial over folds per (subject,trial) to collapse 5-fold
        agg = df.groupby(["subject", "trial", "condition"], as_index=False)[
            "correct_trial"].mean()
        for cond, sub in agg.groupby("condition"):
            tmp = sub[["subject", "trial", "correct_trial"]].rename(
                columns={"correct_trial": "decoder_correct"}
            )
            tmp["model"] = f"backward-trf-cca-mel-{cond}"
            tmp["task"] = "envelope-attendance"
            rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    return out


def pool(df: pd.DataFrame) -> pd.DataFrame:
    """Conditional accuracy by behav_correct bucket."""
    groups: list[dict] = []
    for (model, task), g in df.groupby(["model", "task"]):
        for bc, sub in g.groupby("behav_correct"):
            acc = float(sub["decoder_correct"].mean())
            n = int(len(sub))
            # bootstrap CI
            rng = np.random.default_rng(42)
            if n >= 2:
                idx = rng.integers(0, n, size=(2000, n))
                boots = sub["decoder_correct"].to_numpy()[idx].mean(axis=1)
                lo, hi = float(np.quantile(boots, 0.025)), float(
                    np.quantile(boots, 0.975))
            else:
                lo = hi = acc
            groups.append({
                "model": model, "task": task, "behav_correct": int(bc),
                "decoder_acc": acc, "ci_lo": lo, "ci_hi": hi, "n": n,
            })
    return pd.DataFrame(groups)


def envelope_margin_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """For models that carry a continuous envelope_margin, compare the
    per-trial rho_att - rho_una between behaviour-correct and
    behaviour-wrong trials."""
    if "envelope_margin" not in df.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, task), g in df.groupby(["model", "task"]):
        sub = g.dropna(subset=["envelope_margin"])
        if sub.empty:
            continue
        m1 = sub[sub.behav_correct == 1]["envelope_margin"].to_numpy()
        m0 = sub[sub.behav_correct == 0]["envelope_margin"].to_numpy()
        if len(m0) < 5 or len(m1) < 5:
            continue
        t, p = stats.ttest_ind(m1, m0, equal_var=False)
        u, pu = stats.mannwhitneyu(m1, m0, alternative="two-sided")
        rows.append({
            "model": model, "task": task,
            "margin_behav1": float(m1.mean()),
            "margin_behav0": float(m0.mean()),
            "delta_margin": float(m1.mean() - m0.mean()),
            "t": float(t), "p_t": float(p),
            "U": float(u), "p_mwu": float(pu),
            "n1": int(len(m1)), "n0": int(len(m0)),
        })
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Trial-level McNemar / chi-square tests: is decoder accuracy
    different when the subject answered the Q correctly?"""
    rows: list[dict] = []
    for (model, task), g in df.groupby(["model", "task"]):
        # 2x2 contingency
        correct_when_behav1 = int(((g["behav_correct"] == 1) &
                                   (g["decoder_correct"] == 1)).sum())
        wrong_when_behav1   = int(((g["behav_correct"] == 1) &
                                   (g["decoder_correct"] == 0)).sum())
        correct_when_behav0 = int(((g["behav_correct"] == 0) &
                                   (g["decoder_correct"] == 1)).sum())
        wrong_when_behav0   = int(((g["behav_correct"] == 0) &
                                   (g["decoder_correct"] == 0)).sum())
        table = np.array([[correct_when_behav1, wrong_when_behav1],
                          [correct_when_behav0, wrong_when_behav0]])
        if table.sum() == 0 or (table.sum(axis=1) == 0).any():
            chi2 = p_chi = np.nan
        else:
            chi2, p_chi, _, _ = stats.chi2_contingency(table)
        # Also: Spearman on continuous decoder_correct (can be 0..1 after fold avg)
        rho, p_sp = stats.spearmanr(g["behav_correct"], g["decoder_correct"])
        rows.append({
            "model": model, "task": task,
            "acc_behav1": correct_when_behav1 / max(1, correct_when_behav1 + wrong_when_behav1),
            "acc_behav0": correct_when_behav0 / max(1, correct_when_behav0 + wrong_when_behav0),
            "delta": (correct_when_behav1 / max(1, correct_when_behav1 + wrong_when_behav1)) -
                     (correct_when_behav0 / max(1, correct_when_behav0 + wrong_when_behav0)),
            "n_behav1": correct_when_behav1 + wrong_when_behav1,
            "n_behav0": correct_when_behav0 + wrong_when_behav0,
            "chi2": float(chi2) if not np.isnan(chi2) else np.nan,
            "p_chi": float(p_chi) if not np.isnan(p_chi) else np.nan,
            "spearman_r": float(rho),
            "p_spearman": float(p_sp),
        })
    return pd.DataFrame(rows)


def within_subject_paired(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, task): per-subject delta (acc_behav1 -
    acc_behav0), then Wilcoxon signed-rank across subjects. This is
    the subject-level test, immune to between-subject confounds in the
    pooled chi-square above."""
    rows: list[dict] = []
    for (model, task), g in df.groupby(["model", "task"]):
        deltas = []
        for subj, sub in g.groupby("subject"):
            bc1 = sub[sub.behav_correct == 1]["decoder_correct"]
            bc0 = sub[sub.behav_correct == 0]["decoder_correct"]
            if len(bc1) < 3 or len(bc0) < 3:
                continue
            deltas.append(float(bc1.mean() - bc0.mean()))
        if len(deltas) < 5:
            continue
        deltas = np.array(deltas)
        t, p_t = stats.wilcoxon(deltas, zero_method="wilcox",
                                alternative="two-sided")
        rows.append({
            "model": model, "task": task,
            "n_subjects_eligible": len(deltas),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "n_pos": int((deltas > 0).sum()),
            "n_neg": int((deltas < 0).sum()),
            "wilcoxon_W": float(t),
            "p_wilcoxon": float(p_t),
        })
    return pd.DataFrame(rows)


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, model, task): decoder accuracy on behav-correct vs
    behav-wrong trials; subject-level Spearman across their trial pairs."""
    rows: list[dict] = []
    for (subj, model, task), g in df.groupby(["subject", "model", "task"]):
        bc1 = g[g.behav_correct == 1]
        bc0 = g[g.behav_correct == 0]
        if len(bc1) >= 5 and len(bc0) >= 5:
            rho, p = stats.spearmanr(g["behav_correct"], g["decoder_correct"])
        else:
            rho, p = np.nan, np.nan
        rows.append({
            "subject": int(subj), "model": model, "task": task,
            "acc_behav1": float(bc1["decoder_correct"].mean()) if len(bc1) else np.nan,
            "acc_behav0": float(bc0["decoder_correct"].mean()) if len(bc0) else np.nan,
            "n_behav1": int(len(bc1)), "n_behav0": int(len(bc0)),
            "spearman_r": float(rho) if not np.isnan(rho) else np.nan,
            "p_spearman": float(p) if not np.isnan(p) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    beh = load_behaviour()
    dec = load_decoder_rows()
    joined = dec.merge(beh, on=["subject", "trial"], how="inner")
    print(f"joined rows: {len(joined)}")
    print(f"unique models: {joined['model'].nunique()}")

    joined.to_parquet(RESULTS / "behaviour_decoding.parquet", index=False)

    pooled = pool(joined)
    pooled.to_parquet(RESULTS / "behaviour_decoding_pooled.parquet",
                      index=False)
    print("\nPooled accuracy conditioned on behavioural correctness:")
    print(pooled.to_string(index=False))

    tests = paired_tests(joined)
    tests.to_parquet(RESULTS / "behaviour_decoding_tests.parquet",
                     index=False)
    print("\nChi-square + Spearman:")
    print(tests.to_string(index=False))

    margin = envelope_margin_contrast(joined)
    if not margin.empty:
        margin.to_parquet(RESULTS / "behaviour_decoding_margin.parquet",
                          index=False)
        print("\nEnvelope margin behav=1 vs behav=0:")
        print(margin.to_string(index=False))

    ps = per_subject(joined)
    ps.to_parquet(RESULTS / "behaviour_decoding_per_subject.parquet",
                  index=False)
    print(f"\nPer-subject rows: {len(ps)}")

    ws = within_subject_paired(joined)
    ws.to_parquet(RESULTS / "behaviour_decoding_within_subject.parquet",
                  index=False)
    print("\nWithin-subject paired Wilcoxon across (n eligible) subjects:")
    print(ws.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
