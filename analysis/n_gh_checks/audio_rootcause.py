"""Root-cause: why does the audio-only encoder pick the attended talker at ~0.50?

Metadata already showed (a) NO train/test audio overlap (every talker a unique
voice, folds voice-disjoint) and (b) attended == loudest of the 4 in 100/100
trials, median +15 dB. But extract_envelope z-scores each candidate, removing
scalar loudness. This script tests whether loudness leaks into z-scored envelope
SHAPE that transfers across content:

  * LOUDNESS ORACLE  : argmax raw pre-zscore RMS  -> upper bound (the confound).
  * SHAPE PROBE      : logistic 'attended vs not' on z-scored-envelope shape
                       features ONLY (no level), content-disjoint 5-fold,
                       4-way argmax per trial -> what the model can actually read.
  * per-feature separability + correlation of each shape feature with raw loudness
    -> identifies the leak path (loud -> which shape change).

Envelopes are built exactly like the repo's extract_envelope (target_rms scale
then |hilbert|->20Hz LP->64Hz->z-score). Content-level (subject-independent):
the marking is a property of which talker was the target, not the listener.
"""
import os, sys, json, glob
import numpy as np
import soundfile as sf
from scipy.signal import hilbert, sosfiltfilt, butter, welch
from scipy.stats import kurtosis, skew

UP = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
LOCAL = "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset"
sys.path.insert(0, os.path.join(UP, "scripts"))
import dataloader as dl  # noqa: E402
import csv

FS = 64  # TARGET_FS


def env_zscored(wav, sr, target_rms):
    """Exactly repo extract_envelope, but also return the pre-zscore envelope."""
    x = wav.astype(np.float64)
    x = x * (target_rms / (np.sqrt(np.mean(x ** 2)) + 1e-8))
    e = np.abs(hilbert(x)).astype(np.float32)
    sos = butter(4, 20.0 / (sr / 2), btype="low", output="sos")
    e = sosfiltfilt(sos, e).astype(np.float32)
    e = dl._resample(e, sr, FS)
    ez = (e - e.mean()) / (e.std() + 1e-8)
    return ez, e   # z-scored (what model sees), raw envelope (for loudness ref)


def shape_features(ez):
    """Level-free shape features of a z-scored envelope (mean=0,std=1)."""
    p = np.percentile(ez, [5, 10, 50, 90, 95])
    f, P = welch(ez, fs=FS, nperseg=min(256, len(ez)))
    tot = P.sum() + 1e-12
    def band(lo, hi): return P[(f >= lo) & (f < hi)].sum() / tot
    # temporal sparsity (Gini of |env|)
    a = np.sort(np.abs(ez)); n = len(a)
    gini = (2 * np.sum((np.arange(1, n + 1)) * a) / (n * a.sum() + 1e-12)) - (n + 1) / n
    return {
        "kurtosis": float(kurtosis(ez)),
        "skew": float(skew(ez)),
        "dyn_range_p95_p5": float(p[4] - p[0]),
        "frac_silence_lt_-0.5": float(np.mean(ez < -0.5)),
        "mod_0.5_4Hz": float(band(0.5, 4)),
        "mod_4_8Hz": float(band(4, 8)),
        "hf_8_20Hz": float(band(8, 20)),
        "gini_sparsity": float(gini),
    }


def build():
    man = json.load(open(os.path.join(LOCAL, "metadata", "audio_manifest.json")))
    rows = [r for r in csv.DictReader(open(os.path.join(LOCAL, "metadata", "trials.csv")))
            if r["kind"] == "main"]
    feats, loud, att_flag, trial_of = [], [], [], []
    fnames = None
    for ti, r in enumerate(rows):
        tid = r["trial_id"]; att = int(r["attended_speaker"])
        adir = os.path.join(LOCAL, "media", "audio", tid)
        wavs = []
        ok = True
        for spk in [1, 2, 3, 4]:
            m = sorted(glob.glob(os.path.join(adir, f"speaker{spk}_*.flac")))
            if not m:
                ok = False; break
            w, sr = sf.read(m[0], dtype="float32", always_2d=False)
            wavs.append((w, sr))
        if not ok:
            print("skip", tid); continue
        trms = float(np.mean([np.sqrt(np.mean(w.astype(np.float64) ** 2)) for w, _ in wavs]))
        for spk in [1, 2, 3, 4]:
            w, sr = wavs[spk - 1]
            ez, e_raw = env_zscored(w, sr, trms)
            fd = shape_features(ez)
            if fnames is None:
                fnames = list(fd.keys())
            feats.append([fd[k] for k in fnames])
            loud.append(float(np.sqrt(np.mean(w.astype(np.float64) ** 2))))  # true RMS
            att_flag.append(1 if spk == att else 0)
            trial_of.append(ti)
        if (ti + 1) % 20 == 0:
            print(f"  processed {ti+1} trials")
    return (np.array(feats), np.array(loud), np.array(att_flag),
            np.array(trial_of), fnames)


def four_way_acc(score, att_flag, trial_of):
    """argmax of a per-candidate score within each trial -> 4-way accuracy."""
    correct = 0; n = 0
    for t in np.unique(trial_of):
        idx = np.where(trial_of == t)[0]
        if len(idx) != 4:
            continue
        pick = idx[np.argmax(score[idx])]
        correct += int(att_flag[pick] == 1); n += 1
    return correct / n, n


def main():
    X, loud, y, trial, fnames = build()
    print(f"\ncandidates: {len(y)}  ({len(np.unique(trial))} trials x4)  features: {fnames}")

    # 1) LOUDNESS ORACLE (raw RMS) -- the confound in raw form
    acc, n = four_way_acc(loud, y, trial)
    print(f"\n[LOUDNESS ORACLE] 4-way by argmax raw RMS: {acc:.4f}  (n={n} trials)")

    # 2) per-feature separability (attended vs not) + single-feature 4-way
    from sklearn.metrics import roc_auc_score
    print("\n[PER-FEATURE]  feature: AUC(att vs not) | 4-way argmax | corr-with-loudness")
    order = []
    for j, fn in enumerate(fnames):
        auc = roc_auc_score(y, X[:, j])
        a4, _ = four_way_acc(X[:, j], y, trial)
        a4n, _ = four_way_acc(-X[:, j], y, trial)
        best4 = max(a4, a4n)
        r = np.corrcoef(X[:, j], np.log(loud + 1e-12))[0, 1]
        order.append((abs(auc - 0.5), fn, auc, best4, r))
        print(f"  {fn:22s}: AUC={auc:.3f} | 4way={best4:.3f} | corr(loud)={r:+.3f}")

    # 3) SHAPE PROBE: logistic on shape features ONLY, content-disjoint 5-fold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups=trial):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    acc_shape, n = four_way_acc(oof, y, trial)
    print(f"\n[SHAPE PROBE] logistic on z-scored-envelope SHAPE only, "
          f"content-disjoint 5-fold, 4-way: {acc_shape:.4f}  (n={n})")
    print("           (compare: the deep model's audio-only floor ~0.50; chance 0.25)")

    out = dict(loudness_oracle=acc, shape_probe=acc_shape, chance=0.25,
               features=fnames,
               per_feature={fn: dict(auc=float(a), four_way=float(b), corr_loud=float(r))
                            for _, fn, a, b, r in
                            [(o[0], o[1], o[2], o[3], o[4]) for o in order]})
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "rootcause.json")
    os.makedirs(os.path.dirname(od), exist_ok=True)
    json.dump(out, open(od, "w"), indent=2)
    print(f"\nsaved -> {od}")


if __name__ == "__main__":
    main()
