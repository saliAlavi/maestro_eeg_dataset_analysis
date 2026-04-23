"""Actually-run analyses to tell what signal is present in the EEG.

Tests:
    1) AAD backward-model accuracy (within-subject ridge, 5-fold), 30-s window.
    2) Alpha-lateralization: log(right/left parietal α) vs attended azimuth.
    3) Inter-subject correlation at Cz for identical trials.
    4) Cerebro-acoustic coherence (attended envelope vs each channel, 1-8 Hz).
    5) Task-related alpha desynchronization vs pre-trial baseline.
    6) Gaze-only AAD accuracy (sanity baseline).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import welch, coherence
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold

from aad_utils import (
    EEG_CHANNELS, EEG_SFREQ, RESULTS_DIR, list_subjects, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg, audio_envelope,
    load_audio_file,
)
from aad_utils.config import ATTENDED_SPEAKER_MAP


def get_trial(s, k, l=1.0, h=9.0, sr_out=64.0):
    try:
        eeg, ts = load_eeg_trial(s, k); em = load_eeg_time(s, k)
        g2 = load_gaze_trial_2d(s, k); at = load_audio_timestamps(s, k)
    except (FileNotFoundError, EOFError, ValueError, OSError):
        return None
    try:
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(raw, l_freq=l, h_freq=h, reference="auto")
        raw.resample(sr_out, verbose="ERROR")
        return raw, ali
    except Exception:
        return None


def get_attended_envelope(k, sr_out=64.0):
    tno = f"Training-{k}" if k <= 5 else f"Trial-{k-5}"
    tr = load_trials_csv()
    row = tr[tr["Trial No."] == tno]
    if not len(row): return None
    row = row.iloc[0]
    att_dev = "Device-1" if int(row["Attended Speaker"]) in (1, 2) else "Device-2"
    una_dev = "Device-2" if att_dev == "Device-1" else "Device-1"
    att = audio_envelope(*load_audio_file(row[att_dev]), sr_out=sr_out)
    una = audio_envelope(*load_audio_file(row[una_dev]), sr_out=sr_out)
    az = ATTENDED_SPEAKER_MAP[int(row["Attended Speaker"])][2]
    return att, una, az, float(row["SNR"])


# --------------------------------------------------------------------------- #
def aad_backward(subject, n_trials=40, lags_ms=(0, 50, 100, 150, 200, 250)):
    """Within-subject stimulus reconstruction."""
    SR = 64.0
    lags = [int(round(ms * SR / 1000)) for ms in lags_ms]
    trials = []
    for k in range(6, 6 + n_trials):
        r = get_trial(subject, k, 1, 9, SR)
        if r is None: continue
        raw, _ = r
        env_pkg = get_attended_envelope(k, SR)
        if env_pkg is None: continue
        att, una, az, snr = env_pkg
        E = raw.get_data().T
        L = min(len(E), len(att), len(una))
        if L < 10 * int(SR): continue
        trials.append(dict(E=E[:L], att=att[:L], una=una[:L], az=az, snr=snr))
    if len(trials) < 5: return None

    def lagged(X, lags):
        T = X.shape[0]; out = []
        for lag in lags:
            if lag >= 0: out.append(np.vstack([np.zeros((lag, X.shape[1])), X[:T-lag]]))
            else: out.append(np.vstack([X[-lag:], np.zeros((-lag, X.shape[1]))]))
        return np.concatenate(out, axis=1)

    rows = []
    kf = KFold(5, shuffle=True, random_state=0)
    for tr_i, te_i in kf.split(trials):
        Xs, ys = [], []
        for i in tr_i:
            Xs.append(lagged(trials[i]["E"], lags)); ys.append(trials[i]["att"])
        X = np.vstack(Xs); y = np.concatenate(ys)
        m = Ridge(alpha=1e3).fit(X, y)
        for i in te_i:
            pred = m.predict(lagged(trials[i]["E"], lags))
            ra = np.corrcoef(pred, trials[i]["att"])[0, 1]
            ru = np.corrcoef(pred, trials[i]["una"])[0, 1]
            rows.append(dict(subject=subject, trial_idx=i, rho_att=ra, rho_una=ru,
                             correct=int(ra > ru), az=trials[i]["az"], snr=trials[i]["snr"]))
    return pd.DataFrame(rows)


def alpha_lateralization(subjects, trials_per_subject=30):
    """log(R/L parietal alpha) ~ attended azimuth."""
    rows = []
    for s in subjects:
        for k in range(6, 6 + trials_per_subject):
            r = get_trial(s, k, 1, 40, 200)
            if r is None: continue
            raw, _ = r
            env_pkg = get_attended_envelope(k, 200)
            if env_pkg is None: continue
            *_, az, snr = env_pkg
            d = raw.get_data()
            f, P = welch(d, fs=200, nperseg=400)
            m = (f >= 8) & (f <= 13)
            lp = P[[EEG_CHANNELS.index(c) for c in ["P3", "P7"]]][:, m].mean()
            rp = P[[EEG_CHANNELS.index(c) for c in ["P4", "P8"]]][:, m].mean()
            rows.append(dict(subject=s, trial=k, ALI=np.log(rp / lp), az=az, snr=snr))
    return pd.DataFrame(rows)


def inter_subject_corr(subjects, trials=range(6, 16), channel="Cz"):
    rows = []
    for k in trials:
        sigs = []
        for s in subjects:
            r = get_trial(s, k, 1, 20, 64)
            if r is None: continue
            raw, _ = r
            sigs.append(raw.get_data()[EEG_CHANNELS.index(channel)])
        if len(sigs) < 3: continue
        T = min(len(x) for x in sigs); sigs = np.stack([x[:T] for x in sigs])
        # Leave-one-out ISC.
        isc = [np.corrcoef(sigs[i], np.delete(sigs, i, 0).mean(0))[0, 1] for i in range(len(sigs))]
        rows.append(dict(trial=k, isc_mean=np.mean(isc), n_subjects=len(sigs)))
    return pd.DataFrame(rows)


def cerebro_acoustic(subject, trials=range(6, 26), bands={"delta(1-4)": (1, 4), "theta(4-8)": (4, 8)}):
    SR = 64.0
    acc = {b: [] for b in bands}
    for k in trials:
        r = get_trial(subject, k, 1, 20, SR)
        if r is None: continue
        raw, _ = r
        env_pkg = get_attended_envelope(k, SR)
        if env_pkg is None: continue
        att, _, _, _ = env_pkg
        E = raw.get_data()
        L = min(E.shape[1], len(att))
        for b, (lo, hi) in bands.items():
            ch_vals = []
            for c in range(E.shape[0]):
                f, C = coherence(E[c, :L], att[:L], fs=SR, nperseg=256)
                mask = (f >= lo) & (f <= hi)
                ch_vals.append(C[mask].mean())
            acc[b].append(np.array(ch_vals))
    summary = {b: np.mean(acc[b], axis=0) if acc[b] else np.full(32, np.nan) for b in bands}
    return pd.DataFrame(summary, index=EEG_CHANNELS)


def gaze_aad(subjects, trials_per_subject=40):
    rows = []
    for s in subjects:
        for k in range(6, 6 + trials_per_subject):
            try:
                g2 = load_gaze_trial_2d(s, k)
            except (FileNotFoundError, EOFError, ValueError, OSError): continue
            if len(g2) < 20: continue
            tno = f"Trial-{k-5}"
            tr = load_trials_csv()
            r = tr[tr["Trial No."] == tno]
            if not len(r): continue
            row = r.iloc[0]
            az = ATTENDED_SPEAKER_MAP[int(row["Attended Speaker"])][2]
            rows.append(dict(
                subject=s, trial=k, att=int(row["Attended Speaker"]), az=az,
                gx_mean=np.nanmean(g2["gaze_x"]), gy_mean=np.nanmean(g2["gaze_y"]),
                gx_std=np.nanstd(g2["gaze_x"]), gy_std=np.nanstd(g2["gaze_y"]),
            ))
    G = pd.DataFrame(rows)
    # Within-subject 4-class accuracy.
    accs = []
    for s in G["subject"].unique():
        idx = G["subject"] == s
        if idx.sum() < 20: continue
        Xs = G.loc[idx, ["gx_mean", "gy_mean", "gx_std", "gy_std"]].fillna(0).values
        ys = G.loc[idx, "att"].values
        fold_acc = []
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys):
            m = Pipeline([("sc", StandardScaler()), ("cl", LogisticRegression(max_iter=2000))])
            m.fit(Xs[tr_i], ys[tr_i])
            fold_acc.append(m.score(Xs[te_i], ys[te_i]))
        accs.append(dict(subject=s, acc=np.mean(fold_acc)))
    return G, pd.DataFrame(accs)


def main():
    SUBJECTS = list_subjects()

    # Representative subset: S1 (known worst mastoids), S3, S5, S7, S9, S11, S13, S15.
    SUBSET = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Running signal audit on subjects {SUBSET}\n")

    print("[1/6] AAD backward-model ...")
    t0 = time.time(); dfs = []
    for s in SUBSET:
        d = aad_backward(s, n_trials=25)
        if d is not None: dfs.append(d)
        print(f"    Subject {s:2d}: done ({time.time()-t0:.1f}s elapsed)")
    aad = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    aad.to_parquet(RESULTS_DIR / "audit_aad.parquet")

    print("\n[2/6] Alpha lateralization ...")
    t0 = time.time()
    ali = alpha_lateralization(SUBSET, trials_per_subject=20)
    print(f"    {len(ali)} trials processed ({time.time()-t0:.1f}s)")
    ali.to_parquet(RESULTS_DIR / "audit_alpha_lateralization.parquet")

    print("\n[3/6] Inter-subject correlation ...")
    t0 = time.time()
    isc = inter_subject_corr(SUBSET, trials=range(6, 16), channel="Cz")
    print(f"    {len(isc)} trials ({time.time()-t0:.1f}s)")
    isc.to_parquet(RESULTS_DIR / "audit_isc.parquet")

    print("\n[4/6] Cerebro-acoustic coherence (Subject 3) ...")
    t0 = time.time()
    cac = cerebro_acoustic(3, trials=range(6, 26))
    print(f"    ({time.time()-t0:.1f}s)")
    cac.to_parquet(RESULTS_DIR / "audit_cac.parquet")

    print("\n[5/6] Gaze-only AAD baseline ...")
    t0 = time.time()
    _, gaze_acc = gaze_aad(SUBSET, trials_per_subject=40)
    print(f"    ({time.time()-t0:.1f}s)")
    gaze_acc.to_parquet(RESULTS_DIR / "audit_gaze_acc.parquet")

    # --------- Report ----------
    print("\n\n" + "=" * 72)
    print("SIGNAL AUDIT REPORT")
    print("=" * 72)

    if len(aad):
        per = aad.groupby("subject")["correct"].agg(["mean", "count"])
        overall = aad["correct"].mean()
        r_diff = (aad["rho_att"] - aad["rho_una"]).mean()
        print(f"\n[1] AAD backward-model accuracy (chance=0.5):")
        print(per.to_string())
        print(f"    Pooled:            {overall:.3f}  (mean ρ_att-ρ_una = {r_diff:+.4f})")
        by_snr = aad.groupby(pd.cut(aad["snr"], bins=[0, 6, 10, 14, 20]))["correct"].mean()
        print(f"    By SNR bin:\n{by_snr.to_string()}")

    if len(ali):
        rho, p = spearmanr(ali["az"], ali["ALI"], nan_policy="omit")
        print(f"\n[2] Alpha lateralization (log R/L vs attended az):")
        print(f"    Spearman ρ = {rho:+.3f}  p = {p:.3g}  (n={len(ali)} trials)")
        per_s = ali.groupby("subject").apply(lambda d: spearmanr(d["az"], d["ALI"], nan_policy="omit")[0])
        print(f"    Per-subject ρ:\n{per_s.to_string()}")

    if len(isc):
        print(f"\n[3] Inter-subject correlation at Cz:")
        print(isc.to_string(index=False))
        print(f"    Mean ISC across trials: {isc['isc_mean'].mean():.4f}")

    if len(cac):
        print(f"\n[4] Cerebro-acoustic coherence (Subject 3, attended envelope):")
        for b in cac.columns:
            top5 = cac[b].sort_values(ascending=False).head(5)
            print(f"    Top channels in {b}:")
            for ch, v in top5.items():
                print(f"      {ch:<4}  {v:.4f}")

    if len(gaze_acc):
        print(f"\n[5] Gaze-only 4-class AAD (chance=0.25):")
        print(gaze_acc.to_string(index=False))
        print(f"    Mean accuracy: {gaze_acc['acc'].mean():.3f} (± {gaze_acc['acc'].std():.3f})")

    # Bad-channel summary from the earlier scan.
    bc = RESULTS_DIR / "bad_channels_manifest.parquet"
    if bc.exists():
        m = pd.read_parquet(bc)
        print(f"\n[6] Signal-quality context (from cohort scan):")
        print(f"    Trials with ≥1 bad channel: {(m['n_bad'] >= 1).mean()*100:.1f}%")
        print(f"    Trials with ≥4 bad channels: {(m['n_bad'] >= 4).mean()*100:.1f}%")
        print(f"    M2 saturation prevalence: {m['m2_sat'].mean()*100:.1f}%")

    print("\n" + "=" * 72)
    print("Results cached to analysis/results/audit_*.parquet")


if __name__ == "__main__":
    main()
