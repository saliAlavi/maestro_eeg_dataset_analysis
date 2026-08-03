"""Shared training engine for the per-experiment scripts, with a selectable
DATA METHOD so the *same* code reproduces both columns of the comparison table.

Two methods, chosen per run via ``--data-method``:

  proper (default) — our leakage-safe control:
      * 5 s windows @ 0.5 overlap
      * trial-level splits: within-subject chrono-forward CV, or subject-disjoint LOSO
      * an inner-val split carved from TRAIN for early stopping / checkpoint selection
      * a single evaluation of the best-val checkpoint on a held-out TEST split

  github — the repo's methodology (reproduces the inflated numbers):
      * 30 s whole-trial windows (~1 window/trial, no overlap)
      * pooled StratifiedKFold(5, shuffle, seed 42) with subjects NOT held out
        (or a github-style subject-disjoint LOSO), and val == test
      * the reported number is MAX-over-epochs on the very fold used for early
        stopping / checkpoint selection (best-epoch-on-the-eval-fold)

Everything else (model architecture, optimizer, loss, envelopes) is identical
across methods, so the delta between the two columns is exactly the leakage.
The underlying DATA is our controlled cache in both cases (loudness-matched,
per-device aligned, our EEG preprocessing); the github method reproduces the
repo's evaluation methodology on that controlled data, isolating the
methodology leak. (Exactly matching the repo's absolute numbers would
additionally require the repo's raw-data preprocessing.)
"""
from __future__ import annotations

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

log = logging.getLogger("gh_core")
RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/n_gh_checks")

RECON_COMBOS = [("eeg",), ("gaze",), ("imu",), ("eeg", "gaze"), ("eeg", "imu"),
                ("gaze", "imu"), ("eeg", "gaze", "imu")]


# --------------------------------------------------------------------------- #
# Method / protocol plumbing
# --------------------------------------------------------------------------- #
def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def window_params(method: str, by_subject: dict):
    """(win_len, hop) for the chosen method."""
    if method == "github":
        recs = [r for s in by_subject for r in by_subject[s]]
        L = D.whole_trial_len(recs)               # ~30 s whole trial
        return L, L                               # 1 window/trial, no overlap
    return D.SPEC.win_len, D.SPEC.hop_len         # proper: 320 / 160 (5 s @ 0.5)


def result_tag(method: str, protocol: str) -> str:
    return protocol if method == "proper" else f"gh_{protocol}"


def subjects_needed(protocol: str, subject: int | None):
    if protocol == "within":
        return [subject]
    return D.SUBJECTS                              # loso / pooled need everyone


def get_splits(by_subject, task, method, protocol, subject):
    if method == "proper":
        if protocol == "within":
            return list(D.within_splits(by_subject, subject))
        if protocol == "loso":
            return [D.loso_split(by_subject, subject)]
        raise ValueError(f"proper method supports within|loso, not {protocol}")
    # github
    if protocol == "pooled":
        return list(D.github_pooled_splits(by_subject, task))
    if protocol == "loso":
        return list(D.github_loso_splits(by_subject, task))
    if protocol == "within":                       # per-subject StratifiedKFold, val==test
        from sklearn.model_selection import StratifiedKFold
        recs = by_subject[subject]
        labs = [D.task_label(r, task) for r in recs]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        out = []
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(recs)), labs)):
            train = [recs[i] for i in tr]; test = [recs[i] for i in te]
            out.append(D.Split(f"ghwithin_s{subject}_fold{f}", "within", subject, f,
                               train, test, test))
        return out
    raise ValueError(protocol)


# --------------------------------------------------------------------------- #
# Training internals (shared)
# --------------------------------------------------------------------------- #
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


def _cf_forward(model, data, b, modalities, device):
    def g(key):
        return torch.from_numpy(data[key][b]).to(device) if key in data else None
    eeg = g("eeg") if "eeg" in modalities else None
    gaze = g("gaze") if "gaze" in modalities else None
    imu = g("imu") if "imu" in modalities else None
    audio = [torch.from_numpy(a[b]).to(device) for a in data["audio"]]
    return model(eeg, None, gaze, imu, audio)


@torch.no_grad()
def _cf_acc(model, data, modalities, device, bs=256):
    model.eval()
    labels = data["labels"]; correct = 0
    for b in _batches(len(labels), bs, False, None):
        correct += int((_cf_forward(model, data, b, modalities, device)
                        .argmax(1).cpu().numpy() == labels[b]).sum())
    return correct / max(1, len(labels))


def train_classif(task, modalities, split, method, device, win_len, hop,
                  epochs=50, bs=32, lr=1e-4, patience=10, seed=42):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    n_spk = D.N_SPEAKERS[task]
    model = AADModel(modalities, n_speakers=n_spk).to(device)
    mk = lambda recs: D.materialize_classif(recs, task, modalities, win_len=win_len, hop=hop)
    tr = mk(split.train)
    github = method == "github"
    ev = mk(split.test)                            # for github this fold IS val==test
    va = ev if github else mk(split.val)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=5, min_lr=1e-6)
    ytr = tr["labels"]
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(ytr), bs, True, rng):
            probs = _cf_forward(model, tr, b, modalities, device)
            loss = _ls_ce(probs, torch.from_numpy(ytr[b]).to(device), n_spk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vacc = _cf_acc(model, va, modalities, device)
        sched.step(vacc)
        if vacc > best:
            best, best_ep, bad = vacc, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if not github and bad >= patience:     # github: run all epochs, take max
                break
    model.load_state_dict(best_state)
    if github:
        metric = best                              # max-over-epochs on val==test (the leak)
    else:
        metric = _cf_acc(model, ev, modalities, device)   # held-out test of best-val ckpt
    return dict(task=task, mode="_".join(modalities), method=method,
                metric=metric, best_val=best, best_epoch=best_ep,
                n_test=len(ev["labels"]), chance=1.0 / n_spk), model


@torch.no_grad()
def _recon_r(model, data, device, bs=256):
    model.eval(); X, y = data["X"], data["y"]; rs = []
    for b in _batches(len(X), bs, False, None):
        pred = model(torch.from_numpy(X[b]).to(device))
        rs.append(pearson_r(torch.from_numpy(y[b]).to(device), pred).cpu().numpy())
    return float(np.concatenate(rs).mean()) if rs else 0.0


def train_recon(modalities, split, method, device, win_len, hop,
                epochs=50, bs=32, lr=1e-4, weight_decay=1e-3, patience=10, seed=42):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = LinearModel(n_in_channels=n_channels_for(modalities)).to(device)
    mk = lambda recs: D.materialize_recon(recs, modalities, win_len=win_len, hop=hop)
    tr = mk(split.train)
    github = method == "github"
    ev = mk(split.test); va = ev if github else mk(split.val)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=5, min_lr=1e-6)
    Xtr, ytr = tr["X"], tr["y"]
    best, best_state, best_ep, bad = -1e9, None, 0, 0
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
        vr = _recon_r(model, va, device); sched.step(vr)
        if vr > best:
            best, best_ep, bad = vr, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if not github and bad >= patience:
                break
    model.load_state_dict(best_state)
    metric = best if github else _recon_r(model, ev, device)
    return dict(task="reconstruction", mode="_".join(modalities), method=method,
                metric=metric, best_val=best, best_epoch=best_ep,
                n_test=len(ev["y"]), chance=0.0), model


# --------------------------------------------------------------------------- #
# One experiment (task) end to end
# --------------------------------------------------------------------------- #
def run_experiment(task, method, protocol, modes, subject=None, epochs=50,
                   save_ckpt=True, skip_existing=False):
    """task in {hemisphere,eccentricity,speaker4,reconstruction}; modes is a list
    of modality tuples (classif) or ignored for reconstruction (uses RECON_COMBOS)."""
    device = _device()
    tag = result_tag(method, protocol)
    res_dir = RUN_ROOT / "results" / tag
    ckpt_dir = RUN_ROOT / "ckpt" / tag / task
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sid = subject if protocol == "within" else "all"

    configs = ([("reconstruction", m) for m in RECON_COMBOS] if task == "reconstruction"
               else [(task, tuple(m) if isinstance(m, (list, tuple)) else (m,)) for m in modes])

    by = D.load_subjects(subjects_needed(protocol, subject))
    win_len, hop = window_params(method, by)
    log.info("task=%s method=%s protocol=%s subj=%s win_len=%d hop=%d device=%s",
             task, method, protocol, sid, win_len, hop, device)

    for tk, mods in tqdm(configs, desc=f"{tag}:{task} s{sid}"):
        out = res_dir / f"s{sid}_{tk}_{'_'.join(mods)}.json"
        if skip_existing and out.exists():
            continue
        splits = get_splits(by, tk, method, protocol, subject)
        rows = []
        for sp in splits:
            t0 = time.time()
            if tk == "reconstruction":
                r, model = train_recon(mods, sp, method, device, win_len, hop, epochs=epochs)
            else:
                r, model = train_classif(tk, mods, sp, method, device, win_len, hop, epochs=epochs)
            # for pooled there is no held-out subject; use fold as the aggregation unit
            agg_unit = sp.test_subject if sp.test_subject is not None else sp.fold
            r.update(split=sp.name, protocol=protocol, method=method,
                     test_subject=agg_unit, fold=sp.fold, secs=round(time.time() - t0, 1))
            # keep the metric under both keys so aggregate.py picks it up
            r["test_acc" if tk != "reconstruction" else "test_r"] = r["metric"]
            rows.append(r)
            if save_ckpt:
                torch.save(model.state_dict(), ckpt_dir / f"{sp.name}_{'_'.join(mods)}.pt")
            log.info("[%s|%s|%s|%s] metric=%.4f (val=%.4f ep=%d n=%d %.1fs)",
                     tag, tk, "_".join(mods), sp.name, r["metric"], r["best_val"],
                     r["best_epoch"], r["n_test"], r["secs"])
        json.dump(rows, open(out, "w"), indent=2, default=float)
    log.info("done task=%s method=%s protocol=%s -> %s", task, method, protocol, res_dir)
