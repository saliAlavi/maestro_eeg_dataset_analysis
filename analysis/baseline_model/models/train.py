"""Train + evaluate NeuroCLIP-AAD under the honest protocol, with null controls.

One array task = one (protocol, subject). within -> that subject's 5 chrono-forward
folds; loso -> that held-out test subject's fold. Runs N seeds, reports mean+-sd.

Reported number = a SINGLE evaluation of the best-on-inner-val checkpoint on the
untouched TEST split. Controls computed every run:
  * EEG-shuffle null (scramble EEG across windows, keep candidates) -> must ~0.25
    (proves the decision needs the EEG<->candidate pairing, not candidate structure).
  * CAUSALITY / lag curve (stimulus-bleed control): re-score at EEG<->audio lags in
    {-16,-8,0,8,16} samples (+-250 ms). Genuine cortical tracking is CAUSAL (audio
    LEADS EEG by ~100-250 ms), so accuracy should peak at POSITIVE lag; instantaneous
    acoustic/electrical bleed of the played audio into the electrodes would peak at
    lag 0 and be symmetric. causal_margin = mean(acc | lag>0) - mean(acc | lag<0) > 0
    is evidence the signal is neural, not stimulus artifact.
Trial-order (attended==((k-1)%4)+1) needs NO residualization control here: the model
input is only (eeg, cand) with per-window slot permutation, so trial index / schedule
cannot reach it (a linear detrend of trial_k is vacuous against a mod-4 comb anyway).

Usage:
  python train.py --protocol within --subject 1
  python train.py --protocol loso   --subject 1
  python train.py --smoke
"""
from __future__ import annotations

import argparse, json, logging, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D
from model import NeuroCLIPAAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("neuroclip")
RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad")


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_arrays(view):
    """Materialise a view once: eeg (N,32,320), cand (N,4,28,320) = 1 matched + 3
    same-talker time-shifted spoilers, attended (N,) = the permuted slot of the matched
    (aligned) candidate. Per-window match-mismatch decision; no trial pooling needed."""
    d = view.as_numpy()
    return dict(eeg=d["eeg"].astype(np.float32), cand=d["cand_env"].astype(np.float32),
                att=d["attended"].astype(np.int64))


# --------------------------------------------------------------------------- #
# augmentation (train only)
# --------------------------------------------------------------------------- #
def _augment(eeg, cand, gen):
    B, C, T = eeg.shape
    eeg = eeg + 0.1 * torch.randn(eeg.shape, generator=gen, device=eeg.device)
    ml = int(0.2 * T)
    starts = torch.randint(0, T - ml + 1, (B,), generator=gen, device=eeg.device)
    ar = torch.arange(T, device=eeg.device)
    span = (ar[None] >= starts[:, None]) & (ar[None] < (starts + ml)[:, None])
    gate = (torch.rand(B, generator=gen, device=eeg.device) < 0.5)[:, None]
    eeg = eeg * (~(span & gate))[:, None, :]
    ch = torch.rand(B, C, generator=gen, device=eeg.device).argsort(1)[:, :2]
    cmask = torch.ones(B, C, device=eeg.device).scatter_(1, ch, 0.0)
    cgate = (torch.rand(B, generator=gen, device=eeg.device) < 0.5)[:, None]
    eeg = eeg * torch.where(cgate, cmask, torch.ones_like(cmask))[:, :, None]
    cand = cand + 0.02 * torch.randn(cand.shape, generator=gen, device=cand.device)
    return eeg, cand


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _scores(model, arr, device, bs=512, eeg_shuffle=False):
    model.eval()
    eeg = arr["eeg"]; cand = arr["cand"]
    n = len(eeg)
    order = np.random.default_rng(0).permutation(n) if eeg_shuffle else None
    out = []
    for i in range(0, n, bs):
        j = slice(i, min(i + bs, n))
        e = eeg[order[j]] if eeg_shuffle else eeg[j]
        e = torch.from_numpy(np.ascontiguousarray(e)).to(device)
        c = torch.from_numpy(np.ascontiguousarray(cand[j])).to(device)
        out.append(model.predict_scores(e, c).cpu().numpy())
    return np.concatenate(out, 0)                       # (N,4) slot scores


@torch.no_grad()
def _scores_lag(model, arr, device, lag, bs=512):
    """Re-score with EEG shifted vs candidates by ``lag`` samples (64 Hz).
    lag>0 = audio leads EEG (causal/neural); lag<0 = EEG leads audio (anti-causal).
    Both streams cropped to the overlapping region before encoding."""
    model.eval()
    eeg, cand = arr["eeg"], arr["cand"]
    T = eeg.shape[-1]
    if lag > 0:
        eeg, cand = eeg[..., lag:], cand[..., :T - lag]
    elif lag < 0:
        eeg, cand = eeg[..., :T + lag], cand[..., -lag:]
    n = len(eeg); out = []
    for i in range(0, n, bs):
        j = slice(i, min(i + bs, n))
        e = torch.from_numpy(np.ascontiguousarray(eeg[j])).to(device)
        c = torch.from_numpy(np.ascontiguousarray(cand[j])).to(device)
        out.append(model.predict_scores(e, c).cpu().numpy())
    return np.concatenate(out, 0)


def _metrics(model, arr, device):
    """Per-window match-mismatch accuracy (headline) + controls. Each 5 s window is an
    independent decision: pick the matched (aligned) candidate among the same-talker
    time-shifts. Chance = 1/N_CAND."""
    att = arr["att"]
    acc = float((_scores(model, arr, device).argmax(1) == att).mean())
    # causality / lag curve: neural tracking is CAUSAL (audio leads EEG ~100-250 ms);
    # accuracy should peak at positive lag. (For shifted candidates there is no bleed
    # confound, but the lag curve still localises the neural tracking lag.)
    lags = [-16, -8, 0, 8, 16]
    lag_acc = {str(L): float((_scores_lag(model, arr, device, L).argmax(1) == att).mean())
               for L in lags}
    causal = float(np.mean([lag_acc[str(L)] for L in lags if L > 0]))
    anticausal = float(np.mean([lag_acc[str(L)] for L in lags if L < 0]))
    # EEG-shuffle null: scramble EEG across windows -> must collapse to chance
    null_acc = float((_scores(model, arr, device, eeg_shuffle=True).argmax(1) == att).mean())
    return dict(win_acc=acc, null_acc=null_acc, lag_acc=lag_acc, causal_acc=causal,
                anticausal_acc=anticausal, causal_margin=causal - anticausal,
                n_test=len(att))


# --------------------------------------------------------------------------- #
# one training run (single split, single seed)
# --------------------------------------------------------------------------- #
def train_split(train_view, val_view, test_view, device, seed=0,
                epochs=100, bs=256, lr=3e-4, wd=1e-2, patience=15, warmup=5):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    tr = _as_arrays(train_view); va = _as_arrays(val_view); te = _as_arrays(test_view)
    model = NeuroCLIPAAD().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_at(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    n = len(tr["eeg"]); rng = np.random.default_rng(seed)
    best_val, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train(); idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            eeg = torch.from_numpy(np.ascontiguousarray(tr["eeg"][b])).to(device)
            cand = torch.from_numpy(np.ascontiguousarray(tr["cand"][b])).to(device)
            att = torch.from_numpy(tr["att"][b]).to(device)
            eeg, cand = _augment(eeg, cand, gen)
            loss, _ = model.compute_loss(eeg, cand, att)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vacc = _metrics(model, va, device)["win_acc"]
        if vacc > best_val:
            best_val, best_ep, bad = vacc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    m = _metrics(model, te, device)
    m.update(best_val=best_val, best_epoch=best_ep, seed=seed,
             n_train=len(tr["eeg"]), n_test=len(te["eeg"]))
    return m


def run(protocol, subject, seeds=(0, 1, 2), epochs=100, save=True, skip_existing=False):
    out = RUN_ROOT / "results" / protocol / f"s{subject}.json"
    if skip_existing and out.exists():
        log.info("skip_existing: %s already done", out); return []
    device = _device()
    subs = D.SUBJECTS if protocol == "loso" else [subject]
    log.info("protocol=%s subject=%s device=%s loading %d subjects", protocol, subject, device, len(subs))
    dm = D.build_dm(subs)
    rows = []
    for desc, tr_v, va_v, te_v in D.splits(dm, protocol):
        if protocol == "loso" and desc.test_subject != subject:
            continue
        for seed in seeds:
            t0 = time.time()
            m = train_split(tr_v, va_v, te_v, device, seed=seed, epochs=epochs)
            m.update(split=desc.name, protocol=protocol, test_subject=desc.test_subject,
                     fold=desc.fold, secs=round(time.time() - t0, 1))
            rows.append(m)
            log.info("[%s|%s|seed%d] acc=%.3f null=%.3f causal_margin=%+.3f lags=%s (val=%.3f ep=%d %.0fs)",
                     protocol, desc.name, seed, m["win_acc"], m["null_acc"],
                     m["causal_margin"], m["lag_acc"], m["best_val"], m["best_epoch"], m["secs"])
    if save and rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out, "w"), indent=2, default=float)
        log.info("wrote %s", out)
    return rows


def smoke():
    """Tiny CPU end-to-end + permutation-equivariance + null sanity."""
    device = "cpu"
    # permutation-equivariance unit test
    m = NeuroCLIPAAD().eval()
    eeg = torch.randn(2, 32, 320); cand = torch.randn(2, 4, 28, 320)
    s0 = m.predict_scores(eeg, cand)
    perm = torch.tensor([2, 0, 3, 1])
    s1 = m.predict_scores(eeg, cand[:, perm])
    assert torch.allclose(s0[:, perm], s1, atol=1e-5), "NOT permutation-equivariant!"
    log.info("permutation-equivariance OK")
    dm = D.build_dm([1])
    desc, tr_v, va_v, te_v = next(D.splits(dm, "within"))
    log.info("smoke split %s train=%d val=%d test=%d windows", desc.name, len(tr_v), len(va_v), len(te_v))
    r = train_split(tr_v, va_v, te_v, device, seed=0, epochs=2, patience=2)
    log.info("SMOKE metrics: acc=%.3f null=%.3f causal_margin=%+.3f lags=%s (n_test=%d)",
             r["win_acc"], r["null_acc"], r["causal_margin"], r["lag_acc"], r["n_test"])
    log.info("smoke ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["within", "loso"])
    ap.add_argument("--subject", type=int)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    else:
        assert a.protocol and a.subject, "need --protocol and --subject"
        run(a.protocol, a.subject, seeds=tuple(a.seeds), epochs=a.epochs,
            skip_existing=a.skip_existing)
