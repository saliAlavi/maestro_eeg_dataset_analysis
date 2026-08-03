"""Late fusion (github late_fusion.py) with the same --data-method toggle.

Loads the FROZEN single-modality classification checkpoints trained by the
per-experiment scripts (for the chosen method/protocol), and for each multimodal
combo trains only the LateFusionCombiner (one softmax weight per modality),
selecting on the split's val and reporting its test. Single-modality probs are
frozen, so they are precomputed once per split.

Run AFTER train_hemisphere/eccentricity/pooled have produced checkpoints for the
same --data-method/--protocol:
  python late_fusion.py --data-method proper --protocol within --subject 3
  python late_fusion.py --data-method github --protocol pooled
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

import gh_core as C
import gh_data as D
from gh_models import AADModel, LateFusionCombiner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fusion")
RUN_ROOT = C.RUN_ROOT
COMBOS = [("eeg", "gaze"), ("eeg", "imu"), ("gaze", "imu"), ("eeg", "gaze", "imu")]
CLASSIF_TASKS = ("hemisphere", "eccentricity", "speaker4")


@torch.no_grad()
def _probs(model, data, modality, device, bs=256):
    model.eval(); n = len(data["labels"]); out = []
    for i in range(0, n, bs):
        b = np.arange(i, min(i + bs, n))
        eeg = torch.from_numpy(data["eeg"][b]).to(device) if modality == "eeg" else None
        gaze = torch.from_numpy(data["gaze"][b]).to(device) if modality == "gaze" else None
        imu = torch.from_numpy(data["imu"][b]).to(device) if modality == "imu" else None
        audio = [torch.from_numpy(a[b]).to(device) for a in data["audio"]]
        out.append(model(eeg, None, gaze, imu, audio).cpu().numpy())
    return np.concatenate(out, 0)


def _fuse_split(task, sp, combo, ck_root, win_len, hop, method, device, epochs=30, lr=1e-2, seed=42):
    mods = list(combo); n_spk = D.N_SPEAKERS[task]
    github = method == "github"
    mk = lambda recs: D.materialize_classif(recs, task, mods, win_len=win_len, hop=hop)
    phases = {"train": mk(sp.train), "test": mk(sp.test)}
    phases["val"] = phases["test"] if github else mk(sp.val)
    probs = {ph: {} for ph in phases}
    for ph in phases:
        for m in mods:
            ck = ck_root / task / f"{sp.name}_{m}.pt"
            if not ck.exists():
                raise FileNotFoundError(ck)
            model = AADModel([m], n_speakers=n_spk).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            probs[ph][m] = _probs(model, phases[ph], m, device)
    lab = {ph: phases[ph]["labels"] for ph in phases}

    def acc(comb, ph):
        with torch.no_grad():
            pl = [torch.from_numpy(probs[ph][m]).to(device) for m in mods]
            c = comb(pl).cpu().numpy()
        return float((c.argmax(1) == lab[ph]).mean())

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    comb = LateFusionCombiner(len(mods)).to(device)
    opt = torch.optim.Adam(comb.parameters(), lr=lr)
    ytr = lab["train"]; y_t = torch.from_numpy(np.eye(n_spk, dtype=np.float32)[ytr]).to(device)
    pl_tr = {m: torch.from_numpy(probs["train"][m]).to(device) for m in mods}
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        comb.train(); idx = rng.permutation(len(ytr))
        for i in range(0, len(ytr), 32):
            b = idx[i:i + 32]
            out = comb([pl_tr[m][b] for m in mods])
            loss = -(y_t[b] * torch.log(out + 1e-8)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = acc(comb, "val")
        if v > best:
            best = v; best_state = copy.deepcopy(comb.state_dict())
    comb.load_state_dict(best_state)
    w = F.softmax(comb.logits, 0).detach().cpu().numpy().round(3).tolist()
    agg = sp.test_subject if sp.test_subject is not None else sp.fold
    return dict(task=task, mode="_".join(mods), method=method, protocol=sp.protocol,
                split=sp.name, test_subject=agg, fold=sp.fold, best_val=best,
                test_acc=acc(comb, "test"), n_test=len(lab["test"]),
                chance=1.0 / n_spk, weights=w)


def run(method, protocol, subject=None, epochs=30):
    device = C._device()
    tag = C.result_tag(method, protocol)
    by = D.load_subjects(C.subjects_needed(protocol, subject))
    win_len, hop = C.window_params(method, by)
    ck_root = RUN_ROOT / "ckpt" / tag
    res_dir = RUN_ROOT / "results" / tag
    res_dir.mkdir(parents=True, exist_ok=True)
    subj_iter = ([subject] if protocol in ("within", "loso") and subject else
                 (D.SUBJECTS if protocol in ("within", "loso") else [None]))
    for task in CLASSIF_TASKS:
        rows = []
        for subj in subj_iter:
            for sp in C.get_splits(by, task, method, protocol, subj):
                for combo in COMBOS:
                    try:
                        r = _fuse_split(task, sp, combo, ck_root, win_len, hop, method, device, epochs)
                    except FileNotFoundError as e:
                        log.warning("skip %s %s %s: missing %s", task, sp.name, combo, e); continue
                    rows.append(r)
                    log.info("[fuse|%s|%s|%s|%s] test=%.4f (val=%.4f w=%s)",
                             tag, task, r["mode"], sp.name, r["test_acc"], r["best_val"], r["weights"])
        if rows:
            json.dump(rows, open(res_dir / f"fusion_{task}.json", "w"), indent=2, default=float)
    log.info("fusion done method=%s protocol=%s -> %s", method, protocol, res_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-method", choices=["proper", "github"], default="proper")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    method = a.data_method
    protocol = a.protocol or ("within" if method == "proper" else "pooled")
    run(method, protocol, subject=a.subject, epochs=a.epochs)
