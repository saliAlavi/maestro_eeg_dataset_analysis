"""Iter-6 · Scene-video features + AAD control.

Extracts motion-energy and optical-flow summaries from the Tobii scene
video per trial, then tests whether the video features alone can decode
the attended speaker. Egocentric video of a quiet listening room should
mostly reflect head/body motion; this is a CPU-only analysis, so we use
lightweight features (dense optical flow on downsampled frames + frame
differencing).

CLI:
    python video_features.py --subject 3 --out results/video_aad/s3.parquet
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from aad_utils import RESULTS_DIR, load_trials_csv, video_trial_dir, trial_name
from aad_utils.config import ATTENDED_HEMISPHERE

DS = (160, 90)        # downsample resolution
FRAME_STRIDE = 2      # process every 2nd frame for speed
FLOW_N = 30           # subsample frames for dense-flow stats


def video_feature_vector(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, prev = cap.read()
    if not ok: cap.release(); return None
    prev_g = cv2.cvtColor(cv2.resize(prev, DS), cv2.COLOR_BGR2GRAY)
    energies = []; brightness = []
    flow_mag_samples = []
    flow_dir_samples = []
    flow_std_samples = []
    i = 0
    flow_every = max(1, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / FLOW_N)))
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % FRAME_STRIDE != 0:
            i += 1; continue
        g = cv2.cvtColor(cv2.resize(frame, DS), cv2.COLOR_BGR2GRAY)
        energies.append(float(cv2.absdiff(g, prev_g).mean()))
        brightness.append(float(g.mean()))
        if i % flow_every == 0:
            try:
                flow = cv2.calcOpticalFlowFarneback(prev_g, g, None,
                        0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_mag_samples.append(float(mag.mean()))
                flow_std_samples.append(float(mag.std()))
                flow_dir_samples.append((float(np.mean(np.cos(ang))),
                                          float(np.mean(np.sin(ang)))))
            except Exception: pass
        prev_g = g; i += 1
    cap.release()
    if not energies: return None
    energies = np.array(energies); brightness = np.array(brightness)
    f = dict(
        motion_energy_mean=float(np.mean(energies)),
        motion_energy_std=float(np.std(energies)),
        motion_energy_peak=float(np.percentile(energies, 95)),
        motion_energy_min=float(np.min(energies)),
        brightness_mean=float(np.mean(brightness)),
        brightness_std=float(np.std(brightness)),
        fps=float(fps), n_frames=int(len(energies)),
    )
    if flow_mag_samples:
        f["flow_mag_mean"] = float(np.mean(flow_mag_samples))
        f["flow_mag_std"]  = float(np.std(flow_mag_samples))
        f["flow_mag_peak"] = float(np.max(flow_mag_samples))
        f["flow_std_mean"] = float(np.mean(flow_std_samples))
        cs = np.array([c for c, _ in flow_dir_samples])
        ss = np.array([s for _, s in flow_dir_samples])
        f["flow_dir_cx"] = float(np.mean(cs))
        f["flow_dir_cy"] = float(np.mean(ss))
        f["flow_dir_consistency"] = float(np.sqrt(cs.mean()**2 + ss.mean()**2))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    tr_csv = load_trials_csv()
    rows = []
    for k in range(1, 101):
        vd = video_trial_dir(a.subject, k, kind="main")
        if vd is None: continue
        vp = vd / "scenevideo.mp4"
        if not vp.exists(): continue
        try:
            f = video_feature_vector(vp)
        except Exception: continue
        if f is None: continue
        tno = trial_name(k, "main")
        tr = tr_csv[tr_csv["Trial No."] == tno]
        if not len(tr): continue
        f.update(subject=a.subject, trial=k, attended=int(tr.iloc[0]["Attended Speaker"]),
                 snr=float(tr.iloc[0]["SNR"]))
        rows.append(f)
        if len(rows) % 10 == 0:
            print(f"[S{a.subject}] processed {len(rows)} trials  ({time.time()-t0:.0f}s)", flush=True)
    if len(rows) < 10:
        print(f"[S{a.subject}] too few trials"); return
    F = pd.DataFrame(rows)
    feat_cols = [c for c in F.columns if c not in ("subject","trial","attended","snr","fps","n_frames")]
    X = F[feat_cols].fillna(0).values
    y_full = F["attended"].values
    print(f"[S{a.subject}] {len(F)} trials × {len(feat_cols)} video features", flush=True)

    def eval_task(lbl, task, nc):
        y = np.array([lbl(a_) for a_ in y_full])
        if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2:
            return []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        lr_accs, gb_accs = [], []
        for tr_i, te_i in skf.split(X, y):
            lr = Pipeline([("sc",StandardScaler()),
                           ("cl", LogisticRegression(max_iter=3000, C=0.5))]).fit(X[tr_i], y[tr_i])
            lr_accs.append(accuracy_score(y[te_i], lr.predict(X[te_i])))
            gb = lgb.LGBMClassifier(n_estimators=200, verbosity=-1).fit(X[tr_i], y[tr_i])
            gb_accs.append(accuracy_score(y[te_i], gb.predict(X[te_i])))
        return [
            dict(subject=a.subject, task=task, classifier="logreg",
                 chance=1/nc, acc=float(np.mean(lr_accs))),
            dict(subject=a.subject, task=task, classifier="lightgbm",
                 chance=1/nc, acc=float(np.mean(gb_accs))),
        ]

    results = []
    results += eval_task(lambda a_: 0 if ATTENDED_HEMISPHERE[a_]=="L" else 1, "hemisphere", 2)
    results += eval_task(lambda a_: 0 if a_ in (2,3) else 1, "inner_outer", 2)
    results += eval_task(lambda a_: a_-1, "4class", 4)
    R = pd.DataFrame(results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(a.out.with_suffix(".features.parquet"))
    R.to_parquet(a.out)
    print(f"[S{a.subject}] done in {time.time()-t0:.0f}s", flush=True)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
