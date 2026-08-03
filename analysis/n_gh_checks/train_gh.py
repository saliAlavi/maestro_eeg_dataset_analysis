"""Unified leakage-safe trainer for the github MAESTRO reproduction.

One array task = one (protocol, subject). It trains the whole task x mode matrix
for that subject and writes a per-config JSON + best checkpoint.

Faithful github recipe: Adam(1e-4), ReduceLROnPlateau(max, .5, patience 5),
grad-clip 1.0, manual label-smoothing (0.1) CE on the softmax probs, batch 32,
<=50 epochs, early-stop patience 10 — EXCEPT the monitored metric is the
inner-val split (carved from train), not the test split, and the reported number
is a single evaluation of the best-val checkpoint on the held-out test.

Usage:
  python train_gh.py --protocol within --subject 1
  python train_gh.py --protocol loso   --subject 1     # subject = held-out test subject
  python train_gh.py --smoke                            # tiny CPU end-to-end check
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import gh_data as D
from gh_models import AADModel, LinearModel, n_channels_for, pearson_r, pearson_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_gh")

RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/n_gh_checks")

CLASSIF_CONFIGS = [(t, (m,)) for t in D.CLASSIF_TASKS for m in ("eeg", "gaze", "imu")]
RECON_CONFIGS = [("reconstruction", mods) for mods in (
    ("eeg",), ("gaze",), ("imu",), ("eeg", "gaze"), ("eeg", "imu"),
    ("gaze", "imu"), ("eeg", "gaze", "imu"))]
ALL_CONFIGS = CLASSIF_CONFIGS + RECON_CONFIGS


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batches(n, bs, shuffle, rng):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _ls_ce(probs, labels, k, eps=0.1):
    onehot = F.one_hot(labels, k).float()
    smooth = onehot * (1.0 - eps) + eps / k
    return -(smooth * torch.log(probs + 1e-8)).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# Classification (hemisphere / eccentricity / speaker4)
# --------------------------------------------------------------------------- #
def _classif_forward(model, data, bidx, modalities, device):
    def g(key):
        return torch.from_numpy(data[key][bidx]).to(device) if key in data else None
    eeg = g("eeg") if "eeg" in modalities else None
    gaze = g("gaze") if "gaze" in modalities else None
    imu = g("imu") if "imu" in modalities else None
    audio = [torch.from_numpy(a[bidx]).to(device) for a in data["audio"]]
    return model(eeg, None, gaze, imu, audio)


@torch.no_grad()
def _classif_acc(model, data, modalities, device, bs=256):
    model.eval()
    labels = data["labels"]
    correct = 0
    for b in _batches(len(labels), bs, False, None):
        probs = _classif_forward(model, data, b, modalities, device)
        pred = probs.argmax(1).cpu().numpy()
        correct += int((pred == labels[b]).sum())
    return correct / max(1, len(labels))


def train_classif(task, modalities, split, device, epochs=50, bs=32,
                  lr=1e-4, patience=10, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_spk = D.N_SPEAKERS[task]
    model = AADModel(modalities, n_speakers=n_spk).to(device)
    tr = D.materialize_classif(split.train, task, modalities)
    va = D.materialize_classif(split.val, task, modalities)
    te = D.materialize_classif(split.test, task, modalities)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=5, min_lr=1e-6)
    best_val, best_state, best_epoch, bad = -1.0, None, 0, 0
    labels_tr = tr["labels"]
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(labels_tr), bs, True, rng):
            probs = _classif_forward(model, tr, b, modalities, device)
            y = torch.from_numpy(labels_tr[b]).to(device)
            loss = _ls_ce(probs, y, n_spk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vacc = _classif_acc(model, va, modalities, device)
        sched.step(vacc)
        if vacc > best_val:
            best_val, best_epoch, bad = vacc, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    test_acc = _classif_acc(model, te, modalities, device)
    return dict(task=task, mode="_".join(modalities), best_val=best_val,
                test_acc=test_acc, best_epoch=best_epoch, n_test=len(te["labels"]),
                chance=1.0 / n_spk), model


# --------------------------------------------------------------------------- #
# Reconstruction (linear backward, Pearson r)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _recon_r(model, data, device, bs=256):
    model.eval()
    X, y = data["X"], data["y"]
    rs = []
    for b in _batches(len(X), bs, False, None):
        pred = model(torch.from_numpy(X[b]).to(device))
        rs.append(pearson_r(torch.from_numpy(y[b]).to(device), pred).cpu().numpy())
    return float(np.concatenate(rs).mean()) if rs else 0.0


def train_recon(modalities, split, device, epochs=50, bs=32, lr=1e-4,
                weight_decay=1e-3, patience=10, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = LinearModel(n_in_channels=n_channels_for(modalities)).to(device)
    tr = D.materialize_recon(split.train, modalities)
    va = D.materialize_recon(split.val, modalities)
    te = D.materialize_recon(split.test, modalities)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=5, min_lr=1e-6)
    best_val, best_state, best_epoch, bad = -1e9, None, 0, 0
    Xtr, ytr = tr["X"], tr["y"]
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(Xtr), bs, True, rng):
            pred = model(torch.from_numpy(Xtr[b]).to(device))
            loss = pearson_loss(torch.from_numpy(ytr[b]).to(device), pred)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vr = _recon_r(model, va, device)
        sched.step(vr)
        if vr > best_val:
            best_val, best_epoch, bad = vr, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    test_r = _recon_r(model, te, device)
    return dict(task="reconstruction", mode="_".join(modalities), best_val=best_val,
                test_r=test_r, best_epoch=best_epoch, n_test=len(te["y"]),
                chance=0.0), model


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _splits_for(by_subject, protocol, subject):
    if protocol == "within":
        return list(D.within_splits(by_subject, subject))
    elif protocol == "loso":
        return [D.loso_split(by_subject, subject)]
    raise ValueError(protocol)


def run(protocol, subject, epochs=50, save_ckpt=True, device=None, skip_existing=False):
    device = device or _device()
    subjects = D.SUBJECTS if protocol == "loso" else [subject]
    res_dir = RUN_ROOT / "results" / protocol
    ckpt_dir = RUN_ROOT / "ckpt" / protocol
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # resilience for preemptible/reruns: skip configs whose JSON already exists
    todo = ALL_CONFIGS
    if skip_existing:
        todo = [(t, m) for (t, m) in ALL_CONFIGS
                if not (res_dir / f"s{subject}_{t}_{'_'.join(m)}.json").exists()]
        log.info("skip_existing: %d/%d configs remain", len(todo), len(ALL_CONFIGS))
        if not todo:
            log.info("all configs already done for %s S%s", protocol, subject); return

    log.info("loading %d subjects (protocol=%s, subject=%s, device=%s)",
             len(subjects), protocol, subject, device)
    by_subject = D.load_subjects(subjects)
    splits = _splits_for(by_subject, protocol, subject)

    for task, mods in tqdm(todo, desc=f"{protocol} S{subject}"):
        rows = []
        for sp in splits:
            t0 = time.time()
            if task == "reconstruction":
                r, model = train_recon(mods, sp, device, epochs=epochs)
                metric = r["test_r"]
            else:
                r, model = train_classif(task, mods, sp, device, epochs=epochs)
                metric = r["test_acc"]
            r.update(split=sp.name, protocol=protocol, test_subject=sp.test_subject,
                     fold=sp.fold, secs=round(time.time() - t0, 1))
            rows.append(r)
            if save_ckpt:
                cdir = ckpt_dir / task
                cdir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), cdir / f"{sp.name}_{'_'.join(mods)}.pt")
            log.info("[%s|%s|%s|%s] metric=%.4f (val=%.4f ep=%d n=%d %.1fs)",
                     protocol, task, "_".join(mods), sp.name, metric,
                     r["best_val"], r["best_epoch"], r["n_test"], r["secs"])
        out = res_dir / f"s{subject}_{task}_{'_'.join(mods)}.json"
        with open(out, "w") as f:
            json.dump(rows, f, indent=2, default=float)
    log.info("done protocol=%s subject=%s -> %s", protocol, subject, res_dir)


def smoke():
    """Tiny end-to-end check on real cache (CPU): 1 subject, few epochs."""
    device = "cpu"
    by = D.load_subjects([1])
    sp = next(D.within_splits(by, 1))
    log.info("smoke split: train=%d val=%d test=%d trials", len(sp.train), len(sp.val), len(sp.test))
    for task, mods in [("hemisphere", ("eeg",)), ("speaker4", ("gaze",)),
                       ("reconstruction", ("eeg", "imu"))]:
        if task == "reconstruction":
            r, _ = train_recon(mods, sp, device, epochs=2, patience=2)
        else:
            r, _ = train_classif(task, mods, sp, device, epochs=2, patience=2)
        log.info("SMOKE %s %s -> %s", task, mods,
                 {k: r[k] for k in ("test_acc", "test_r", "best_val", "n_test") if k in r})
    log.info("smoke ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["within", "loso"])
    ap.add_argument("--subject", type=int)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    else:
        assert a.protocol and a.subject, "need --protocol and --subject"
        run(a.protocol, a.subject, epochs=a.epochs, save_ckpt=not a.no_ckpt,
            skip_existing=a.skip_existing)
