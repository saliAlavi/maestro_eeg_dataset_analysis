"""Truthful baseline for the GitHub AADModel: content-disjoint splits + EEG-shuffle null.

The strict-LOSO adjudication (ADJUDICATION.md) showed the AADModel's 0.356 four-way is
non-neural: its EEG-shuffle null equals its accuracy, because the learned audio encoder reads
the attended talker's audio -- and in strict LOSO the SAME 100 stimuli appear in train and test
(shared across subjects), so the model can memorize the attended track/voice. Here we remove
that path by making train/val/test **content-disjoint** (a trial_k in test never appears in
train or val), for BOTH protocols, and re-measure real accuracy and the EEG-shuffle null:

  within : per-subject 5-fold over trial_k (train/val/test share no trial_k).
  loso   : held-out subject AND held-out content -- test = the held-out subject's held-out
           trial_k; train/val = the other 15 subjects' remaining content (val trial_k disjoint
           from train and test).

If the elevated null was audio-track memorization, content-disjoint training should push the
null back to 0.25 (a four-choice problem). Writes results/truthful_{protocol}/s{S}.json.

  python truthful_gh.py --subject 1 --protocol loso
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import torch

import gh_data as D
from gh_data import Split, materialize_classif
from train_gh import train_classif, _classif_forward, _batches

RUN_ROOT = "/fs/scratch/PAS2301/alialavi/projects/n_gh_checks"
TASK = "speaker4"; MODS = ("eeg",)


def _content_split(by, seed=42, test_frac=0.30, val_frac=0.15):
    ref = by[sorted(by)[0]]
    by_att = defaultdict(list)
    for r in ref:
        by_att[(r.attended - 1)].append(r.trial_k)
    rng = np.random.default_rng(seed); test_c, val_c = set(), set()
    for a, ks in by_att.items():
        ks = sorted(set(ks)); rng.shuffle(ks)
        nt = max(1, int(round(test_frac * len(ks)))); nv = max(1, int(round(val_frac * len(ks))))
        test_c.update(ks[:nt]); val_c.update(ks[nt:nt + nv])
    return test_c, val_c


def loso_cd(by, test_s, seed=42):
    test_c, val_c = _content_split(by, seed)
    test = [r for r in by[test_s] if r.trial_k in test_c]
    tr, va = [], []
    for s in by:
        if s == test_s:
            continue
        for r in by[s]:
            if r.trial_k in test_c:
                continue
            (va if r.trial_k in val_c else tr).append(r)
    return [Split(f"cdloso_s{test_s}", "loso", test_s, None, tr, va, test)]


def within_cd(by, subject, seed=42, n_folds=5):
    recs = by[subject]
    by_att = defaultdict(list)
    for r in recs:
        by_att[(r.attended - 1)].append(r.trial_k)
    rng = np.random.default_rng(seed); folds = [set() for _ in range(n_folds)]
    for a, ks in by_att.items():
        ks = list(dict.fromkeys(ks)); rng.shuffle(ks)
        for i, k in enumerate(ks):
            folds[i % n_folds].add(k)
    all_tk = {r.trial_k for r in recs}
    out = []
    for f in range(n_folds):
        test_c = folds[f]
        rest = np.array(sorted(all_tk - test_c)); np.random.default_rng(100 + f).shuffle(rest)
        val_c = set(rest[:max(1, int(round(0.15 * len(rest))))].tolist())
        test = [r for r in recs if r.trial_k in test_c]
        va = [r for r in recs if r.trial_k in val_c]
        tr = [r for r in recs if r.trial_k not in test_c and r.trial_k not in val_c]
        out.append(Split(f"cdwithin_s{subject}_f{f}", "within", subject, f, tr, va, test))
    return out


@torch.no_grad()
def null_acc(model, te, device, n=20):
    labels = te["labels"]; accs = []; rng = np.random.default_rng(0)
    for _ in range(n):
        d = dict(te); d["eeg"] = te["eeg"][rng.permutation(len(labels))]   # break EEG<->trial
        correct = 0
        for b in _batches(len(labels), 256, False, None):
            probs = _classif_forward(model, d, b, MODS, device)
            correct += int((probs.argmax(1).cpu().numpy() == labels[b]).sum())
        accs.append(correct / max(1, len(labels)))
    return float(np.mean(accs))


def run(subject, protocol, device):
    by = D.load_subjects(D.SUBJECTS if protocol == "loso" else [subject])
    splits = loso_cd(by, subject) if protocol == "loso" else within_cd(by, subject)
    accs, nulls, ns = [], [], []
    for sp in splits:
        r, model = train_classif(TASK, MODS, sp, device)
        te = materialize_classif(sp.test, TASK, MODS)
        a, nl = r["test_acc"], null_acc(model, te, device)
        accs.append(a); nulls.append(nl); ns.append(len(te["labels"]))
        print(f"[truthful-gh|{protocol}|s{subject}|{sp.name}] real={a:.3f} null={nl:.3f} "
              f"val={r['best_val']:.3f} ep={r['best_epoch']} n={ns[-1]}", flush=True)
    n = sum(ns)
    real = sum(a * k for a, k in zip(accs, ns)) / n; null = sum(x * k for x, k in zip(nulls, ns)) / n
    row = dict(subject=subject, protocol=protocol, real=real, null=null, n=n, chance=0.25)
    out = f"{RUN_ROOT}/results/truthful_{protocol}"; os.makedirs(out, exist_ok=True)
    json.dump(row, open(f"{out}/s{subject}.json", "w"), default=float)
    print(f"[truthful-gh|{protocol}|s{subject}] REAL={real:.3f} NULL={null:.3f} "
          f"(real-null={real-null:+.3f}) n={n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--protocol", choices=["within", "loso"], required=True)
    a = ap.parse_args()
    run(a.subject, a.protocol, "cuda" if torch.cuda.is_available() else "cpu")
