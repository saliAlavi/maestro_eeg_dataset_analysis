"""Ridge backward-model (mTRF) baseline on the shortcut-free tasks.

The canonical non-deep AAD decoder: reconstruct the speech envelope from lagged
EEG with ridge regression, then score each candidate by the Pearson correlation
between the reconstruction and that candidate, and take the argmax.

    s_hat(t) = sum_{c, tau} g(c, tau) * r(c, t + tau),   tau in [0, 390] ms

Positive lags: the neural response to audio at time t appears in EEG at
t + 100..300 ms, so the reconstruction reads FUTURE EEG relative to the stimulus
sample it predicts.

This is the honest floor the deep model has to clear.  It is also structurally
immune to the audio shortcut for the same reason `CouplingHead` is: Pearson r is
invariant to affine rescaling of the candidate, and a constant reconstruction
correlates 0 with everything, so a brain-independent solution scores chance.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

UP = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
sys.path.insert(0, os.path.join(UP, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import (build_dataset, load_official_splits,             # noqa: E402
                        get_official_split_windows, carve_inner_val,
                        carve_inner_val_content, compute_global_content_holdout)
from data_v2 import (make_audio_bank, content_per_window,                # noqa: E402
                     position_in_trial)
from scipy.signal import butter, sosfiltfilt                             # noqa: E402


def bandpass(eeg, lo, hi, fs=64):
    """Zero-phase band-pass over the window axis. The benchmark ships EEG at
    1-40 Hz; envelope tracking lives in delta-theta, so a backward model given
    the full 1-40 Hz band is fitting mostly noise. Giving ridge its own band is
    what makes the comparison fair."""
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, eeg, axis=1), dtype=np.float32)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LAGS = 26                       # 0..390 ms at 64 Hz
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def lagged(eeg):
    """(B,T,C) -> (B,T,C*N_LAGS) with x[b,t,(c,l)] = eeg[b, t+l, c]."""
    B, T, C = eeg.shape
    pad = torch.cat([eeg, eeg.new_zeros(B, N_LAGS - 1, C)], dim=1)
    return torch.stack([pad[:, l:l + T] for l in range(N_LAGS)],
                       dim=-1).reshape(B, T, C * N_LAGS)


def fit_ridge(eeg, env, batch=32):
    """Accumulate normal equations over windows (never materialises the full
    design matrix) and solve for every lambda."""
    D = eeg.shape[2] * N_LAGS
    XtX = torch.zeros(D, D, device=DEVICE, dtype=torch.float64)
    Xty = torch.zeros(D, device=DEVICE, dtype=torch.float64)
    for s in range(0, len(eeg), batch):
        X = lagged(eeg[s:s + batch].to(DEVICE)).reshape(-1, D).double()
        y = env[s:s + batch].to(DEVICE).reshape(-1).double()
        XtX += X.T @ X
        Xty += X.T @ y
    scale = torch.diagonal(XtX).mean()
    eye = torch.eye(D, device=DEVICE, dtype=torch.float64)
    return {lam: torch.linalg.solve(XtX + lam * scale * eye, Xty) for lam in LAMBDAS}


@torch.no_grad()
def reconstruct(eeg, g, batch=32):
    """(N,T,C) -> (N,T) predicted envelope."""
    D = eeg.shape[2] * N_LAGS
    out = []
    for s in range(0, len(eeg), batch):
        X = lagged(eeg[s:s + batch].to(DEVICE)).reshape(-1, D).double()
        out.append((X @ g).reshape(-1, eeg.shape[1]).float().cpu())
    return torch.cat(out)


def corr_scores(pred, cands):
    """pred (N,T), cands (N,K,T) -> (N,K) Pearson r over time."""
    p = pred - pred.mean(-1, keepdim=True)
    p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
    c = cands - cands.mean(-1, keepdim=True)
    c = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.einsum("nt,nkt->nk", p, c)


def evaluate(pred, cands, labels, strata=None, n_shuffle=20, seed=1000):
    strata = strata or {}
    acc = float((corr_scores(pred, cands).argmax(1) == labels).float().mean())
    # reconstruction quality itself: r between prediction and the TRUE envelope
    recon_r = float(corr_scores(pred, cands[torch.arange(len(labels)), labels]
                                .unsqueeze(1)).mean())
    nulls = []
    snulls = {k: [] for k in strata}
    for k in range(n_shuffle):
        rng = np.random.default_rng(seed + k)
        nulls.append(float((corr_scores(pred[rng.permutation(len(pred))], cands)
                            .argmax(1) == labels).float().mean()))
        for nm, st in strata.items():
            q = np.arange(len(pred))
            r2 = np.random.default_rng(5000 + k)
            for gv in np.unique(st):
                m = np.where(st == gv)[0]
                q[m] = m[r2.permutation(len(m))]
            snulls[nm].append(float((corr_scores(pred[q], cands).argmax(1) == labels)
                                    .float().mean()))
    return dict(acc=acc, null_mean=float(np.mean(nulls)), recon_r=recon_r,
                margin=float(acc - np.mean(nulls)),
                p_perm=float((np.sum(np.array(nulls) >= acc) + 1) / (n_shuffle + 1)),
                **{f"null_{nm}": float(np.mean(v)) for nm, v in snulls.items()},
                **{f"margin_{nm}": float(acc - np.mean(v)) for nm, v in snulls.items()},
                n=int(len(pred)))


def prep(data, bank, idx):
    eeg = torch.from_numpy(data["eeg"][idx])
    A = torch.from_numpy(bank["A"][idx])                       # (n,K,T)
    pos = torch.from_numpy(bank["pos"][idx].astype(np.int64))
    # the attended candidate is the regression target
    env = A[torch.arange(len(idx)), pos]
    return eeg, A, pos, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_path", default="/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset")
    ap.add_argument("--cache_root", default="/fs/scratch/PAS2301/alialavi/cache")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--hop_sec", type=float, default=5.0)
    ap.add_argument("--split_setting", default="within", choices=["within", "loso"])
    ap.add_argument("--cands", default="qmatch:4,shifted_qm:2,shifted_qm:3,raw:4")
    ap.add_argument("--band", default="none",
                    help="lo,hi band-pass for the EEG, or none")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dc = os.path.join(args.cache_root, f"n_gh_fixed_data__eeg"
                      f"_w{args.window_sec:g}_h{args.hop_sec:g}_all.npz")
    if os.path.exists(dc):
        print(f"[build] <- cached {dc}", flush=True)
        z = np.load(dc)
        data = {k: z[k] for k in z.files if not k.startswith("audio_")}
        data["audio"] = [z[f"audio_{i}"] for i in range(4)]
    else:
        data = build_dataset(local_path=args.local_path, mode="eeg", subjects="all",
                             cache_dir=os.path.join(args.cache_root, "n_gh_newrepo__eeg"),
                             window_sec=args.window_sec, hop_sec=args.hop_sec)

    if args.band != "none":
        lo, hi = (float(v) for v in args.band.split(","))
        data["eeg"] = bandpass(data["eeg"], lo, hi)
        print(f"[band] EEG band-passed to {lo}-{hi} Hz", flush=True)

    folds = load_official_splits(os.path.join(args.local_path, "splits"),
                                 args.split_setting)
    if args.split_setting == "loso":
        tr_c, ho_c = compute_global_content_holdout(data, 0.2, SEED)
    wc = content_per_window(data)
    pos_all = position_in_trial(data)

    results = {}
    for spec in args.cands.split(","):
        cm, K = spec.split(":"); K = int(K)
        bank = make_audio_bank(data, cm, args.window_sec, args.hop_sec, n_cand=K)
        per_fold = {}
        for fi in folds:
            tr_idx, te_idx = get_official_split_windows(data, fi)
            if args.split_setting == "loso":
                tr_idx = tr_idx[np.isin(wc[tr_idx], list(tr_c))]
                te_idx = te_idx[np.isin(wc[te_idx], list(ho_c))]
                itr, ivl = carve_inner_val(data, tr_idx, 0.2, SEED + fi["fold"])
            else:
                itr, ivl = carve_inner_val_content(data, tr_idx, 0.2, SEED + fi["fold"])
            if len(te_idx) < 5 or len(itr) < 20:
                continue

            eeg_tr, _, _, env_tr = prep(data, bank, itr)
            gs = fit_ridge(eeg_tr, env_tr)

            eeg_vl, A_vl, pos_vl, _ = prep(data, bank, ivl)
            best_lam, best = None, -1.0
            for lam, g in gs.items():
                a = float((corr_scores(reconstruct(eeg_vl, g), A_vl).argmax(1)
                           == pos_vl).float().mean())
                if a > best:
                    best, best_lam = a, lam

            eeg_te, A_te, pos_te, _ = prep(data, bank, te_idx)
            r = evaluate(reconstruct(eeg_te, gs[best_lam]), A_te, pos_te,
                         strata={"pos": pos_all[te_idx],
                                 "trial": data["trial_ids"][te_idx]})
            r["lambda"] = best_lam
            r["inner_val_acc"] = best
            per_fold[fi["fold"]] = r
            print(f"  {cm} K={K} fold {fi['fold']}: acc={r['acc']:.4f} "
                  f"null={r['null_mean']:.4f} margin={r['margin']:+.4f} "
                  f"recon_r={r['recon_r']:.4f} lam={best_lam:g} n={r['n']}", flush=True)

        keys = [k for k in next(iter(per_fold.values())) if k != "lambda"]
        mean = {k: [float(np.mean([v[k] for v in per_fold.values()])),
                    float(np.std([v[k] for v in per_fold.values()]))] for k in keys}
        results[f"{cm}_K{K}"] = {"folds": per_fold, "mean": mean, "chance": 1.0 / K}
        print(f"##### RIDGE {cm} K={K}: acc={mean['acc'][0]:.4f}+-{mean['acc'][1]:.4f} "
              f"null={mean['null_mean'][0]:.4f} margin={mean['margin'][0]:+.4f} "
              f"recon_r={mean['recon_r'][0]:.4f} (chance {1.0/K:.4f})", flush=True)

        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "results", "fixed",
                                       f"ridge_{args.split_setting}"
                                       f"_w{args.window_sec:g}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(results, open(out, "w"), indent=2, default=float)
        print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
