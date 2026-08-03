"""Leakage-safe late fusion — github late_fusion.py, honest protocol.

Loads the FROZEN single-modality classification checkpoints trained by
train_gh.py, and for each multimodal combo trains only the LateFusionCombiner
(one softmax weight per modality) on THIS split's train windows, selecting on the
inner-val split and reporting a single held-out-test evaluation. Single-modality
probabilities are frozen, so they are precomputed once per split.

Run after train_gh.py has produced ckpt/{protocol}/{task}/{split}_{mode}.pt:
  python late_fusion_gh.py --protocol within
  python late_fusion_gh.py --protocol loso
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import gh_data as D
from gh_models import AADModel, LateFusionCombiner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fusion_gh")

RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/n_gh_checks")
COMBOS = [("eeg", "gaze"), ("eeg", "imu"), ("gaze", "imu"), ("eeg", "gaze", "imu")]


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _model_probs(model, data, modality, device, bs=256):
    """Frozen single-modality probs over all windows -> (N, n_spk)."""
    model.eval()
    n = len(data["labels"])
    out = []
    for i in range(0, n, bs):
        b = np.arange(i, min(i + bs, n))
        eeg = torch.from_numpy(data["eeg"][b]).to(device) if modality == "eeg" else None
        gaze = torch.from_numpy(data["gaze"][b]).to(device) if modality == "gaze" else None
        imu = torch.from_numpy(data["imu"][b]).to(device) if modality == "imu" else None
        audio = [torch.from_numpy(a[b]).to(device) for a in data["audio"]]
        out.append(model(eeg, None, gaze, imu, audio).cpu().numpy())
    return np.concatenate(out, 0)


def _load_single(task, split_name, modality, device):
    n_spk = D.N_SPEAKERS[task]
    ck = RUN_ROOT / "ckpt" / _proto_of(split_name) / task / f"{split_name}_{modality}.pt"
    model = AADModel([modality], n_speakers=n_spk).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    return model


def _proto_of(split_name):
    return "within" if split_name.startswith("within") else "loso"


def _fuse_split(task, sp, combo, device, epochs=30, lr=1e-2, bs=32, seed=42):
    """Precompute frozen single probs; train combiner on train, select on val,
    evaluate on test. Returns dict."""
    mods = list(combo)
    data = {ph: D.materialize_classif(getattr(sp, ph), task, mods)
            for ph in ("train", "val", "test")}
    # frozen per-modality probs for each phase
    probs = {ph: {} for ph in data}
    for ph in data:
        for m in mods:
            model = _load_single(task, sp.name, m, device)
            probs[ph][m] = _model_probs(model, data[ph], m, device)
    labels = {ph: data[ph]["labels"] for ph in data}

    def acc(combiner, ph):
        with torch.no_grad():
            pl = [torch.from_numpy(probs[ph][m]).to(device) for m in mods]
            comb = combiner(pl).cpu().numpy()
        return float((comb.argmax(1) == labels[ph]).mean())

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    combiner = LateFusionCombiner(len(mods)).to(device)
    opt = torch.optim.Adam(combiner.parameters(), lr=lr)
    ytr = labels["train"]; n_spk = D.N_SPEAKERS[task]
    pl_tr = {m: torch.from_numpy(probs["train"][m]).to(device) for m in mods}
    y_tr_t = torch.from_numpy(np.eye(n_spk, dtype=np.float32)[ytr]).to(device)
    best_val, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        combiner.train()
        idx = rng.permutation(len(ytr))
        for i in range(0, len(ytr), bs):
            b = idx[i:i + bs]
            comb = combiner([pl_tr[m][b] for m in mods])
            loss = -(y_tr_t[b] * torch.log(comb + 1e-8)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = acc(combiner, "val")
        if v > best_val:
            best_val = v
            best_state = copy.deepcopy(combiner.state_dict())
    combiner.load_state_dict(best_state)
    w = F.softmax(combiner.logits, 0).detach().cpu().numpy().round(3).tolist()
    return dict(task=task, mode="_".join(mods), protocol=sp.protocol,
                split=sp.name, test_subject=sp.test_subject, fold=sp.fold,
                best_val=best_val, test_acc=acc(combiner, "test"),
                n_test=len(labels["test"]), chance=1.0 / n_spk, weights=w)


def run(protocol, subjects=None, epochs=30):
    device = _device()
    subjects = subjects or D.SUBJECTS
    by = D.load_subjects(D.SUBJECTS if protocol == "loso" else subjects)
    res_dir = RUN_ROOT / "results" / protocol
    res_dir.mkdir(parents=True, exist_ok=True)
    for task in D.CLASSIF_TASKS:
        rows = []
        for s in subjects:
            splits = (list(D.within_splits(by, s)) if protocol == "within"
                      else [D.loso_split(by, s)])
            for sp in splits:
                for combo in COMBOS:
                    try:
                        r = _fuse_split(task, sp, combo, device, epochs=epochs)
                    except FileNotFoundError as e:
                        log.warning("skip %s %s %s: %s", task, sp.name, combo, e)
                        continue
                    rows.append(r)
                    log.info("[fuse|%s|%s|%s|%s] test=%.4f (val=%.4f w=%s)",
                             protocol, task, r["mode"], sp.name, r["test_acc"],
                             r["best_val"], r["weights"])
        with open(res_dir / f"fusion_{task}.json", "w") as f:
            json.dump(rows, f, indent=2, default=float)
    log.info("fusion done protocol=%s -> %s", protocol, res_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["within", "loso"], required=True)
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    run(a.protocol, subjects=a.subjects, epochs=a.epochs)
