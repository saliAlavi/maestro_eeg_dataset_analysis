"""Train + evaluate the backward (stimulus-reconstruction) AAD baselines.

The model reconstructs the attended talker's envelope from EEG (neg-Pearson loss).
The decision correlates the reconstruction against the four REAL co-present talkers'
envelopes (data.py, permuted slots) and reports BOTH:
  * 4-way accuracy   (argmax over the 4 real talkers; chance 0.25 at EVERY window)
  * binary accuracy  (attended vs each other talker; chance 0.5)
plus the EEG-shuffle null (must hit chance 0.25) and the causal lag curve.

Honest protocol: trial-disjoint train/val splits in both within and LOSO (data.py);
inner-val early stopping on binary MM accuracy; single held-out test eval of the
best-val checkpoint. The scale-free correlation + per-(subject,trial) candidate
permutation defuse loudness and the deterministic attended schedule.

Usage:
  python train_bwd.py --model vlaai  --protocol loso   --subject 1
  python train_bwd.py --model linear --protocol within --subject 1
  python train_bwd.py --model vlaai  --smoke
"""
from __future__ import annotations

import argparse, json, logging, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D
import backward as B

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bwd")
RUN_ROOT = Path("/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad")

# MINIMUM INNOVATION: discriminative-reconstruction match-mismatch margin.
# lambda>0 adds an InfoNCE/CE over the reconstruction's correlations with the 4
# candidates -> the reconstruction is pushed to correlate MORE with the matched
# candidate than with the same-talker time-shifted spoilers, aligning training with
# the decision metric. Confound-free (same talker => only temporal alignment helps);
# does not touch any reserved method-paper lever. Set via --mm-margin.
MM_MARGIN = 0.0
MM_SCALE = 12.0     # correlations live in [-1,1]; scale sharpens them for the CE
# MINIMUM INNOVATION 2: multi-band (spectrogram) reconstruction. BANDS=28 reconstructs
# the 28 gammatone bands and fuses per-band correlations (band-heterogeneous cortical
# tracking) instead of one broadband envelope. Set via --bands.
BANDS = 1
# Push-accuracy knobs: HIDDEN/BLOCKS = VLAAI capacity (data-rich LOSO); EARLY = inner-val
# early-stop metric (binary_acc|four_acc); ENSEMBLE = average scores over N seed-models.
HIDDEN = 128
BLOCKS = 4
EARLY = "binary"
ENSEMBLE = 1
# Data-efficiency levers for the data-starved within-subject regime (EEG-only; no reserved
# method-paper lever). AUG_* = train-time EEG augmentation; denser TRAIN windows via D.OVERLAP.
AUG_CHDROP = 0.0     # per-channel dropout probability (robust spatial reconstruction)
AUG_NOISE = 0.0      # additive Gaussian noise std (EEG is z-scored, so ~fraction of 1 SD)

OPT = {  # per-model training config
    "linear": dict(lr=1e-3, wd=1e-2, epochs=60, bs=256, patience=15, warmup=3),
    "vlaai":  dict(lr=1e-3, wd=1e-4, epochs=80, bs=128, patience=15, warmup=5),
    "vlaai2": dict(lr=1e-3, wd=1e-3, epochs=90, bs=128, patience=18, warmup=6),
}


def _augment(eeg):
    """Train-time EEG augmentation (channel dropout + additive noise). eeg: (B,32,T)."""
    if AUG_CHDROP > 0:
        keep = (torch.rand(eeg.shape[0], eeg.shape[1], 1, device=eeg.device) > AUG_CHDROP).float()
        eeg = eeg * keep / (1.0 - AUG_CHDROP)
    if AUG_NOISE > 0:
        eeg = eeg + AUG_NOISE * torch.randn_like(eeg)
    return eeg


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_arrays(view):
    """eeg (N,32,W); att (N,) matched slot. Broadband (BANDS=1): cand (N,4,W),
    target (N,W). Multi-band (BANDS=28): cand (N,4,28,W), target (N,28,W)."""
    d = view.as_numpy()
    eeg = d["eeg"].astype(np.float32)
    att = d["attended"].astype(np.int64)
    full = d["cand_env"].astype(np.float32)                     # (N,4,28,W)
    cand = full if BANDS > 1 else full.mean(2)                  # (N,4,28,W) or (N,4,W)
    target = cand[np.arange(len(att)), att]                     # (N,28,W) or (N,W)
    return dict(eeg=eeg, cand=cand, att=att, target=target)


def _batches(n, bs, shuffle, rng):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def _recon(model, eeg_np, device, bs=512):
    model.eval(); out = []
    for i in range(0, len(eeg_np), bs):
        e = torch.from_numpy(np.ascontiguousarray(eeg_np[i:i + bs])).to(device)
        out.append(model(e).cpu().numpy())
    return np.concatenate(out, 0)                               # (N,W)


def _acc_from_scores(scores, att):
    four = float((scores.argmax(1) == att).mean())
    s_match = scores[np.arange(len(att)), att][:, None]
    wins = (s_match > scores).sum(1)                           # spoilers beaten (att col is False)
    binary = float((wins / (scores.shape[1] - 1)).mean())     # matched vs each spoiler, chance 0.5
    return binary, four


def _score_matrix(model, arr, device, shuffle=False):
    """(N,K) correlation scores of the reconstruction with each candidate."""
    eeg = arr["eeg"]
    if shuffle:
        eeg = eeg[np.random.default_rng(0).permutation(len(eeg))]
    r = _recon(model, eeg, device)
    return B.mm_scores(torch.from_numpy(r), torch.from_numpy(arr["cand"])).numpy()


def _metrics_from(scores, null_scores, att):
    b, f = _acc_from_scores(scores, att); nb, nf = _acc_from_scores(null_scores, att)
    return dict(binary_acc=b, four_acc=f, null_binary=nb, null_four=nf, n_test=len(att))


def _metrics(model, arr, device):
    m = _metrics_from(_score_matrix(model, arr, device),
                      _score_matrix(model, arr, device, shuffle=True), arr["att"])
    T = arr["eeg"].shape[-1]; lags = [-16, -8, 0, 8, 16]; lag_bin = {}
    for L in lags:
        if L > 0:
            e, c = arr["eeg"][..., L:], arr["cand"][..., :T - L]
        elif L < 0:
            e, c = arr["eeg"][..., :T + L], arr["cand"][..., -L:]
        else:
            e, c = arr["eeg"], arr["cand"]
        r = _recon(model, e, device)
        lag_bin[str(L)] = _acc_from_scores(B.mm_scores(torch.from_numpy(r), torch.from_numpy(c)).numpy(),
                                           arr["att"])[0]
    causal = float(np.mean([lag_bin[str(L)] for L in lags if L > 0]))
    anti = float(np.mean([lag_bin[str(L)] for L in lags if L < 0]))
    m.update(lag_binary=lag_bin, causal_margin=causal - anti)
    return m


def _ensemble_metrics(models, arr, device):
    """Average the candidate-correlation scores across the seed-models."""
    s = np.mean([_score_matrix(m, arr, device) for m in models], 0)
    ns = np.mean([_score_matrix(m, arr, device, shuffle=True) for m in models], 0)
    return _metrics_from(s, ns, arr["att"])


def train_split(tr_v, va_v, te_v, model_name, device, seed=0):
    cfg = OPT[model_name]
    torch.manual_seed(seed); np.random.seed(seed)
    tr = _as_arrays(tr_v); va = _as_arrays(va_v); te = _as_arrays(te_v)
    kw = dict(n_out=BANDS)
    if model_name in ("vlaai", "vlaai2"):
        kw.update(hidden=HIDDEN, n_blocks=BLOCKS)
    model = B.build_backward(model_name, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    def lr_at(ep):
        if ep < cfg["warmup"]:
            return (ep + 1) / cfg["warmup"]
        p = (ep - cfg["warmup"]) / max(1, cfg["epochs"] - cfg["warmup"])
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    n = len(tr["eeg"]); rng = np.random.default_rng(seed)
    Xe, Yt, Xc, At = tr["eeg"], tr["target"], tr["cand"], tr["att"]
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for b in _batches(n, cfg["bs"], True, rng):
            eeg = torch.from_numpy(np.ascontiguousarray(Xe[b])).to(device)
            tgt = torch.from_numpy(np.ascontiguousarray(Yt[b])).to(device)
            r_hat = model(_augment(eeg))
            loss = B.neg_pearson_loss(r_hat, tgt)
            if MM_MARGIN > 0:                              # discriminative-reconstruction term
                cand = torch.from_numpy(np.ascontiguousarray(Xc[b])).to(device)
                att = torch.from_numpy(At[b]).to(device)
                scores = B.mm_scores(r_hat, cand)          # (B,4) correlations
                loss = loss + MM_MARGIN * F.cross_entropy(MM_SCALE * scores, att)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vmetric = _metrics(model, va, device)[EARLY + "_acc"]
        if vmetric > best:
            best, best_ep, bad = vmetric, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    m = _metrics(model, te, device)
    m.update(model=model_name, mm_margin=MM_MARGIN, best_val=best, best_epoch=best_ep,
             seed=seed, n_train=len(tr["eeg"]))
    return m, model, te


def run(model_name, protocol, subject, seeds=(0, 1, 2), skip_existing=False):
    variant = model_name + ("mb" if BANDS > 1 else "") + ("mm" if MM_MARGIN > 0 else "")
    if D.WIN_S != 5.0:
        variant += f"w{D.WIN_S:g}"
    if ENSEMBLE > 1:
        variant += f"e{ENSEMBLE}"
    if HIDDEN != 128:
        variant += f"h{HIDDEN}"
    if EARLY != "binary":
        variant += "f4"
    out = RUN_ROOT / "results" / f"{variant}_{protocol}" / f"s{subject}.json"
    if skip_existing and out.exists():
        log.info("skip %s", out); return []
    device = _device()
    subs = D.SUBJECTS if protocol == "loso" else [subject]
    log.info("model=%s protocol=%s subject=%s device=%s", model_name, protocol, subject, device)
    dm = D.build_dm(subs)
    rows = []
    for desc, tr_v, va_v, te_v in D.splits(dm, protocol):
        if protocol == "loso" and desc.test_subject != subject:
            continue
        if ENSEMBLE > 1:                                       # average scores over N seed-models
            t0 = time.time(); models = []; te = None
            for seed in range(ENSEMBLE):
                _, model, te = train_split(tr_v, va_v, te_v, model_name, device, seed)
                models.append(model)
            m = _ensemble_metrics(models, te, device)
            m.update(model=model_name, mm_margin=MM_MARGIN, ensemble=ENSEMBLE, win_s=D.WIN_S,
                     split=desc.name, protocol=protocol, test_subject=desc.test_subject,
                     fold=desc.fold, secs=round(time.time() - t0, 1))
            rows.append(m)
            log.info("[%s|%s|%s|ens%d] bin=%.3f(null %.3f) 4way=%.3f(null %.3f) (%.0fs)",
                     model_name, protocol, desc.name, ENSEMBLE, m["binary_acc"], m["null_binary"],
                     m["four_acc"], m["null_four"], m["secs"])
        else:
            for seed in seeds:
                t0 = time.time()
                m, _, _ = train_split(tr_v, va_v, te_v, model_name, device, seed)
                m.update(win_s=D.WIN_S, split=desc.name, protocol=protocol,
                         test_subject=desc.test_subject, fold=desc.fold, secs=round(time.time() - t0, 1))
                rows.append(m)
                log.info("[%s|%s|%s|seed%d] bin=%.3f(null %.3f) 4way=%.3f(null %.3f) causal=%+.3f (val=%.3f ep=%d %.0fs)",
                         model_name, protocol, desc.name, seed, m["binary_acc"], m["null_binary"],
                         m["four_acc"], m["null_four"], m["causal_margin"], m["best_val"], m["best_epoch"], m["secs"])
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out, "w"), indent=2, default=float)
        log.info("wrote %s", out)
    return rows


def _ens_recon(models, eeg_np, device):
    """z-score each model's reconstruction over time, then average across seeds."""
    rs = []
    for m in models:
        r = _recon(m, eeg_np, device)
        rs.append((r - r.mean(-1, keepdims=True)) / (r.std(-1, keepdims=True) + 1e-6))
    return np.mean(rs, 0)


def _candidate_only_acc(arr):
    """logreg on candidate envelope features -> matched slot; MUST be ~chance (honesty)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    c = arr["cand"]; bb = c.mean(2) if c.ndim == 4 else c            # (N,K,W)
    F = np.stack([bb.std(-1), np.abs(bb).mean(-1), (bb ** 2).mean(-1)], -1).reshape(len(bb), -1)
    try:
        return float(cross_val_score(LogisticRegression(max_iter=300), F, arr["att"], cv=3).mean())
    except Exception:
        return float("nan")


def _curve_windows_from_models(models, test_recs, windows, device):
    """Per window -> (r_bar, cand, att): recon-ensembled reconstruction + candidates + labels
    for one held-out chunk (a LOSO subject, or one within-subject fold)."""
    out = {}
    for w in windows:
        te = D.test_view(test_recs, w)
        if len(te) == 0:                                  # window longer than the ~30 s trial
            log.warning("window %g s: no windows fit the trial; skipping", w); continue
        arr = _as_arrays(te)
        out[w] = (_ens_recon(models, arr["eeg"], device), arr["cand"], arr["att"])
    return out


def _curve_rows(pooled, n_seeds, model_name, subject, protocol):
    """pooled: {win: [(r_bar, cand, att), ...chunks...]}. Concatenate chunks (folds), then
    per window compute real 4-way/binary + averaged EEG-shuffle null + candidate-only."""
    rows = []
    for w in sorted(pooled):
        chunks = pooled[w]
        r_bar = np.concatenate([c[0] for c in chunks], 0)
        cand = np.concatenate([c[1] for c in chunks], 0)
        att = np.concatenate([c[2] for c in chunks], 0)
        cand_t = torch.from_numpy(cand)
        b, f = _acc_from_scores(B.mm_scores(torch.from_numpy(r_bar), cand_t).numpy(), att)
        rng = np.random.default_rng(0); nbs, nfs = [], []       # EEG-shuffle == permute r_bar rows
        for _ in range(50):
            sn = B.mm_scores(torch.from_numpy(r_bar[rng.permutation(len(r_bar))]), cand_t).numpy()
            bn_, fn_ = _acc_from_scores(sn, att); nbs.append(bn_); nfs.append(fn_)
        nb, nf = float(np.mean(nbs)), float(np.mean(nfs))
        co = _candidate_only_acc(dict(cand=cand, att=att))
        rows.append(dict(win_s=w, four_acc=f, binary_acc=b, null_four=nf, null_binary=nb,
                         cand_only=co, n_test=len(att), n_seeds=n_seeds,
                         model=model_name, test_subject=subject, protocol=protocol))
        log.info("[curve|%s|s%d|w%g] 4way=%.3f (null %.3f, cand-only %.3f) bin=%.3f (null %.3f) n=%d",
                 protocol, subject, w, f, nf, co, b, nb, len(att))
    return rows


def run_curve(model_name, subject, windows=(5, 10, 15, 20, 30), n_seeds=5, skip_existing=False,
              protocol="loso"):
    """Train n_seeds at 5 s, then EVALUATE the decision-window curve (up to the whole trial) with
    reconstruction-ensembling. Candidates are the four real talkers, so the null is ~0.25 at every
    window. loso: train on the other 15 subjects, eval the held-out subject. within: per-subject
    5-fold, train on 4 folds, eval the held-out fold, POOL test windows across folds."""
    tag = f"curve_{model_name}" + ("mb" if BANDS > 1 else "") + ("mm" if MM_MARGIN > 0 else "")
    if AUG_CHDROP > 0 or AUG_NOISE > 0:
        tag += "aug"
    if D.OVERLAP != 0.5:
        tag += f"ov{int(round(D.OVERLAP * 100))}"
    if HIDDEN != 128:
        tag += f"h{HIDDEN}"
    out = RUN_ROOT / "results" / f"{tag}_{protocol}" / f"s{subject}.json"
    if skip_existing and out.exists():
        log.info("skip %s", out); return []
    device = _device()
    D.WIN_S = 5.0                                          # train/val at 5 s (data-rich); eval curve below
    from collections import defaultdict
    pooled = defaultdict(list)
    if protocol == "loso":
        dm = D.build_dm(D.SUBJECTS)
        _, tr_v, va_v, test_recs = D.loso_curve_split(dm, subject)
        models = []
        for sd in range(n_seeds):
            t0 = time.time()
            _, model, _ = train_split(tr_v, va_v, D.test_view(test_recs, 5.0), model_name, device, sd)
            models.append(model)
            log.info("  [s%d seed%d trained %.0fs]", subject, sd, time.time() - t0)
        for w, chunk in _curve_windows_from_models(models, test_recs, windows, device).items():
            pooled[w].append(chunk)
    else:                                                 # within: pool test windows over 5 folds
        dm = D.build_dm([subject])
        for fold, tr, va, te in D._intra_folds(dm.by_subject[subject], seed=42):
            tr_v, va_v = D._view(tr, D.WIN_S, "tr"), D._view(va, D.WIN_S, "val")
            t0 = time.time(); models = []
            for sd in range(n_seeds):
                _, model, _ = train_split(tr_v, va_v, D.test_view(te, 5.0), model_name, device, sd)
                models.append(model)
            log.info("  [s%d fold%d %dseeds trained %.0fs]", subject, fold, n_seeds, time.time() - t0)
            for w, chunk in _curve_windows_from_models(models, te, windows, device).items():
                pooled[w].append(chunk)
    rows = _curve_rows(pooled, n_seeds, model_name, subject, protocol)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2, default=float)
    log.info("wrote %s", out)
    return rows


def smoke():
    device = "cpu"
    for name in ("linear", "vlaai"):
        dm = D.build_dm([1])
        desc, tr_v, va_v, te_v = next(D.splits(dm, "within"))
        OPT[name] = {**OPT[name], "epochs": 3, "patience": 3}
        m, _, _ = train_split(tr_v, va_v, te_v, name, device, seed=0)
        log.info("SMOKE %s: bin=%.3f(null %.3f) 4way=%.3f(null %.3f) causal=%+.3f lags=%s",
                 name, m["binary_acc"], m["null_binary"], m["four_acc"], m["null_four"],
                 m["causal_margin"], {k: round(v, 2) for k, v in m["lag_binary"].items()})
    log.info("smoke ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear", "vlaai", "vlaai2"], default="vlaai")
    ap.add_argument("--protocol", choices=["within", "loso"])
    ap.add_argument("--subject", type=int)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=0, help="override the model's default epochs")
    ap.add_argument("--mm-margin", type=float, default=0.0,
                    help="weight of the discriminative-reconstruction match-mismatch CE term")
    ap.add_argument("--bands", type=int, default=1, help="1=broadband, 28=multi-band spectrogram")
    ap.add_argument("--win-s", type=float, default=5.0, help="decision-window length (s)")
    ap.add_argument("--hidden", type=int, default=128, help="VLAAI hidden width")
    ap.add_argument("--blocks", type=int, default=4, help="VLAAI residual blocks")
    ap.add_argument("--early", choices=["binary", "four"], default="binary",
                    help="inner-val early-stop metric")
    ap.add_argument("--ensemble", type=int, default=1, help="average scores over N seed-models")
    ap.add_argument("--curve", action="store_true",
                    help="L1+L2: train at 5s, eval the decision-window curve (recon-ensemble, circular spoilers)")
    ap.add_argument("--dec-wins", type=float, nargs="*", default=[5, 10, 15, 20, 30])
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--overlap", type=float, default=0.5, help="TRAIN-window overlap (eval fixed at 0.5)")
    ap.add_argument("--aug-chdrop", type=float, default=0.0, help="per-channel dropout prob (train aug)")
    ap.add_argument("--aug-noise", type=float, default=0.0, help="additive Gaussian noise std (train aug)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    MM_MARGIN = a.mm_margin
    BANDS = a.bands
    D.WIN_S = a.win_s
    D.OVERLAP = a.overlap
    HIDDEN = a.hidden
    BLOCKS = a.blocks
    EARLY = a.early
    ENSEMBLE = a.ensemble
    AUG_CHDROP = a.aug_chdrop
    AUG_NOISE = a.aug_noise
    if a.smoke:
        smoke()
    elif a.curve:
        assert a.subject, "need --subject"
        if a.epochs:
            OPT[a.model] = {**OPT[a.model], "epochs": a.epochs}
        run_curve(a.model, a.subject, windows=tuple(a.dec_wins), n_seeds=a.n_seeds,
                  skip_existing=a.skip_existing, protocol=a.protocol or "loso")
    else:
        assert a.protocol and a.subject, "need --protocol and --subject"
        if a.epochs:
            OPT[a.model] = {**OPT[a.model], "epochs": a.epochs}
        run(a.model, a.protocol, a.subject, seeds=tuple(a.seeds), skip_existing=a.skip_existing)
