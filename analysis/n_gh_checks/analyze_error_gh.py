"""Pairwise error complementarity (github analyze_error.py), leakage-safe folds.

For the 4-class task, reloads the eeg/gaze/imu single-modality checkpoints,
records per-window correctness on each fold's held-out TEST windows (aligned
across modalities), pools across subjects/folds, and for every modality pair
computes the 2x2 agreement table, complementary rate, and a McNemar test.
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import gh_data as D
from gh_models import AADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("err_gh")
RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/n_gh_checks")


@torch.no_grad()
def _correct(model, data, modality, device):
    labels = data["labels"]; preds = []
    for i in range(0, len(labels), 256):
        b = np.arange(i, min(i + 256, len(labels)))
        eeg = torch.from_numpy(data["eeg"][b]).to(device) if modality == "eeg" else None
        gaze = torch.from_numpy(data["gaze"][b]).to(device) if modality == "gaze" else None
        imu = torch.from_numpy(data["imu"][b]).to(device) if modality == "imu" else None
        audio = [torch.from_numpy(a[b]).to(device) for a in data["audio"]]
        preds.append(model(eeg, None, gaze, imu, audio).argmax(1).cpu().numpy())
    return (np.concatenate(preds) == labels)


def _mcnemar(b, c):
    """b, c = discordant counts. Exact binomial if b+c<25 else chi2 w/ continuity."""
    from scipy.stats import binomtest, chi2
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    stat = (abs(b - c) - 1) ** 2 / n
    return float(chi2.sf(stat, 1))


def run(modes=("eeg", "gaze", "imu"), protocol="within"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    by = D.load_subjects(D.SUBJECTS)
    ck_root = RUN_ROOT / "ckpt" / protocol / "speaker4"
    cor = {m: [] for m in modes}
    for s in D.SUBJECTS:
        splits = list(D.within_splits(by, s)) if protocol == "within" else [D.loso_split(by, s)]
        for sp in splits:
            data = D.materialize_classif(sp.test, "speaker4", list(modes))
            have = {}
            for m in modes:
                ck = ck_root / f"{sp.name}_{m}.pt"
                if not ck.exists():
                    break
                model = AADModel([m], n_speakers=4).to(device)
                model.load_state_dict(torch.load(ck, map_location=device))
                have[m] = _correct(model, data, m, device)
            if len(have) == len(modes):
                for m in modes:
                    cor[m].append(have[m])
    cor = {m: np.concatenate(v) for m, v in cor.items() if v}
    pairs = {}
    for a, b in combinations(modes, 2):
        if a not in cor or b not in cor:
            continue
        ca, cb = cor[a], cor[b]
        both = int((ca & cb).sum()); nither = int((~ca & ~cb).sum())
        a_only = int((ca & ~cb).sum()); b_only = int((~ca & cb).sum())
        n = len(ca)
        pairs[f"{a}|{b}"] = dict(
            n=n, both_correct=both, neither=nither, a_only=a_only, b_only=b_only,
            complementary_rate=(a_only + b_only) / n,
            acc_a=float(ca.mean()), acc_b=float(cb.mean()),
            union_acc=float((ca | cb).mean()), mcnemar_p=_mcnemar(a_only, b_only))
        log.info("%s|%s: compl=%.3f union=%.3f p=%.3g",
                 a, b, pairs[f"{a}|{b}"]["complementary_rate"],
                 pairs[f"{a}|{b}"]["union_acc"], pairs[f"{a}|{b}"]["mcnemar_p"])
    out = RUN_ROOT / "results" / f"error_complementarity_{protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(chance=0.25, pairs=pairs), open(out, "w"), indent=2, default=float)
    log.info("wrote %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="within", choices=["within", "loso"])
    a = ap.parse_args()
    run(protocol=a.protocol)
