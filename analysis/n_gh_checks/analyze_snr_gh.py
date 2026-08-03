"""SNR-stratified 4-class accuracy (github analyze_snr.py), leakage-safe folds.

Reloads the trained speaker4 single-modality checkpoints, recomputes per-window
correctness on each within-fold's held-out TEST windows, tags each window with
its trial's SNR (from trials.csv), and reports accuracy in equal-count SNR bins
pooled across all subjects/folds. Writes JSON + a PNG.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import gh_data as D
from gh_models import AADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snr_gh")
RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/n_gh_checks")


def _snr_by_trial():
    """{trial_k (1..100): snr_db} from the main trials.csv rows."""
    import src.data.aad_compat as C
    csv = C.load_trials_csv()
    col = next((c for c in csv.columns if c.strip().lower() in ("snr", "snr_db")), None)
    if col is None:
        raise RuntimeError(f"no SNR column in trials.csv; have {list(csv.columns)}")
    out = {}
    for k in range(1, 101):
        row = csv[csv["Trial No."] == C.trial_name(k, "main")]
        if len(row):
            out[k] = float(row.iloc[0][col])
    return out


@torch.no_grad()
def _correct_and_meta(model, records, task, modality, device):
    """Per-window (correct bool, trial_k) on a record list for a single model."""
    data = D.materialize_classif(records, task, [modality])
    meta = D._window_indices(records, D.SPEC)
    labels = data["labels"]
    preds = []
    n = len(labels)
    for i in range(0, n, 256):
        b = np.arange(i, min(i + 256, n))
        eeg = torch.from_numpy(data["eeg"][b]).to(device) if modality == "eeg" else None
        gaze = torch.from_numpy(data["gaze"][b]).to(device) if modality == "gaze" else None
        imu = torch.from_numpy(data["imu"][b]).to(device) if modality == "imu" else None
        audio = [torch.from_numpy(a[b]).to(device) for a in data["audio"]]
        preds.append(model(eeg, None, gaze, imu, audio).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    correct = (preds == labels)
    trial_ks = np.array([records[p].trial_k for (p, s) in meta], np.int64)
    return correct, trial_ks


def run(modes=("eeg", "gaze", "imu"), n_bins=4, protocol="within"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snr_map = _snr_by_trial()
    by = D.load_subjects(D.SUBJECTS)
    ck_root = RUN_ROOT / "ckpt" / protocol / "speaker4"
    results = {}
    all_snr = []  # collect once for global bin edges
    per_mode = {m: {"snr": [], "correct": []} for m in modes}
    for s in D.SUBJECTS:
        splits = list(D.within_splits(by, s)) if protocol == "within" else [D.loso_split(by, s)]
        for sp in splits:
            for m in modes:
                ck = ck_root / f"{sp.name}_{m}.pt"
                if not ck.exists():
                    continue
                model = AADModel([m], n_speakers=4).to(device)
                model.load_state_dict(torch.load(ck, map_location=device))
                correct, tks = _correct_and_meta(model, sp.test, "speaker4", m, device)
                snr = np.array([snr_map.get(int(k), np.nan) for k in tks])
                good = np.isfinite(snr)
                per_mode[m]["snr"].append(snr[good])
                per_mode[m]["correct"].append(correct[good])
                if m == modes[0]:
                    all_snr.append(snr[good])
    edges = np.quantile(np.concatenate(all_snr), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for m in modes:
        snr = np.concatenate(per_mode[m]["snr"]); cor = np.concatenate(per_mode[m]["correct"])
        bins = np.digitize(snr, edges[1:-1])
        rows = []
        for bi in range(n_bins):
            sel = bins == bi
            rows.append(dict(bin=bi, n=int(sel.sum()),
                             snr_lo=float(edges[bi + 1] if bi else np.nanmin(snr)),
                             acc=float(cor[sel].mean()) if sel.sum() else float("nan")))
        results[m] = rows
        log.info("SNR %s: %s", m, [round(r["acc"], 3) for r in rows])
    out = RUN_ROOT / "results" / f"snr_analysis_{protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(chance=0.25, n_bins=n_bins, edges=[float(e) for e in edges],
                   by_mode=results), open(out, "w"), indent=2, default=float)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for m in modes:
            plt.plot([r["bin"] for r in results[m]], [r["acc"] for r in results[m]],
                     "o-", label=m)
        plt.axhline(0.25, ls="--", c="gray", label="chance")
        plt.xlabel(f"SNR quantile bin (low->high, {n_bins} bins)"); plt.ylabel("4-class acc")
        plt.title(f"SNR-stratified accuracy ({protocol}, leakage-safe)"); plt.legend()
        plt.tight_layout(); plt.savefig(out.with_suffix(".png"), dpi=120)
        log.info("wrote %s", out.with_suffix(".png"))
    except Exception as e:
        log.warning("plot skipped: %s", e)
    log.info("wrote %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="within", choices=["within", "loso"])
    ap.add_argument("--n_bins", type=int, default=4)
    a = ap.parse_args()
    run(n_bins=a.n_bins, protocol=a.protocol)
