"""Ablation: does a LEARNED similarity head close our baseline's gap to the GitHub model?

Our baseline decides by a FIXED, equal-weight Fisher-z average of the per-band
reconstruction<->candidate correlations. The GitHub AADModel instead uses a LEARNED
similarity head (a learned linear projection over the matched features). Here we add the
faithful analog to OUR reconstruction baseline: keep the same VLAAI 28-band content decoder,
but replace the fixed band average with a LEARNED linear head over the 28 per-band correlations,
fit discriminatively (cross-entropy over the four candidates) on the training fold only.

Strict LOSO, speaker4 (chance 0.25), 5 s. For each held-out subject we report, on the SAME
reconstructions: (a) FIXED readout = our baseline; (b) LEARNED readout = +learned head. The head
is fit on training windows only, so the comparison is admissible.

  python learned_head_ablation.py <test_subject>
"""
import glob, os, sys, json, numpy as np, torch
import torch.nn.functional as F
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
OUT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad/results/learned_head"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 64.0; W = 320
SUBJECT = int(sys.argv[1]); ALLSUB = list(range(1, 17))
NSEED = int(os.environ.get("NSEED", "3")); MARGIN = float(os.environ.get("MARGIN", "0.3"))
PROTOCOL = os.environ.get("PROTOCOL", "loso")


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def prep(s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64); env = z["env"][:, :4].astype(np.float32)
    att = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int)
    rng = np.random.default_rng(20260619 + s); perm = np.stack([rng.permutation(4) for _ in range(len(att))])
    T = eeg.shape[-1]; st = list(range(0, T - W + 1, W // 2)); tri = np.tile(np.arange(len(att)), len(st))
    eegw = np.concatenate([eeg[:, :, a:a + W] for a in st], 0)
    candp = _zt(np.concatenate([env[:, :, :, a:a + W] for a in st], 0).astype(np.float32), -1)
    permw = np.concatenate([perm for _ in st], 0); attw = att[tri]
    yslot = np.array([np.flatnonzero(permw[i] == attw[i])[0] for i in range(len(tri))])
    cand = np.stack([candp[i][permw[i]] for i in range(len(candp))])
    return dict(eeg_z=_zt(eegw, 2), tgt=candp[np.arange(len(tri)), attw], cand=cand,
                yslot=yslot, tk=tk[tri])


def train_content(tr, va, seed=0, epochs=60, patience=12):
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(tr["eeg_z"]); rng = np.random.default_rng(seed); best = -1; bstate = None; bad = 0
    for ep in range(epochs):
        m.train(); idx = rng.permutation(n)
        for i in range(0, n, 256):
            b = idx[i:i + 256]
            r = m(torch.from_numpy(tr["eeg_z"][b]).to(DEV))
            loss = B.neg_pearson_loss(r, torch.from_numpy(tr["tgt"][b]).to(DEV))
            if MARGIN > 0:
                sc = B.mm_scores(r, torch.from_numpy(tr["cand"][b]).to(DEV))
                loss = loss + MARGIN * F.cross_entropy(12.0 * sc, torch.from_numpy(tr["yslot"][b]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        vb = _val_bin(m, va)
        if vb > best:
            best = vb; bstate = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(bstate); return m


@torch.no_grad()
def _val_bin(m, d, bs=512):
    m.eval(); z = _perband_corr([m], d)  # (N,4,28) atanh-corr
    s = z.mean(-1); y = d["yslot"]
    sm = s[np.arange(len(y)), y][:, None]
    return float(((sm > s).sum(1) / 3).mean())


@torch.no_grad()
def _recon_and_cand(models, d, bs=512):
    """Ensembled (z-scored) reconstruction rbar (N,28,W) and the candidates (N,4,28,W)."""
    eeg = d["eeg_z"]; rs = []
    for m in models:
        m.eval(); out = []
        for i in range(0, len(eeg), bs):
            r = m(torch.from_numpy(eeg[i:i + bs]).to(DEV)).cpu()
            out.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
        rs.append(torch.cat(out, 0))
    return torch.stack(rs).mean(0).numpy(), d["cand"]


def _corr_from(rbar, cand):
    """Fisher-z per-band corr of rbar (N,28,W) with each candidate (N,4,28,W) -> (N,4,28)."""
    rho = B.pearson(torch.from_numpy(rbar).unsqueeze(1), torch.from_numpy(cand))
    return torch.atanh(rho.clamp(-0.999, 0.999)).numpy()


def _perband_corr(models, d):
    rbar, cand = _recon_and_cand(models, d)
    return _corr_from(rbar, cand)


def _acc(scores, y):
    return float((scores.argmax(1) == y).mean())


def _slice(d, tk_set, keep=True):
    m = np.isin(d["tk"], list(tk_set))
    sel = m if keep else ~m
    return {k: v[sel] for k, v in d.items()}


def _fit_head(Ztr, ytr):
    """Learned linear head over the 28 per-band correlations (CE over candidates, train-only)."""
    w = torch.zeros(28, requires_grad=True, device=DEV); b0 = torch.zeros(1, requires_grad=True, device=DEV)
    X = torch.from_numpy(Ztr).float().to(DEV); Y = torch.from_numpy(ytr).long().to(DEV)
    opt = torch.optim.Adam([w, b0], lr=0.05, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy((X * w).sum(-1) + b0, Y).backward(); opt.step()
    return w.detach().cpu().numpy()


def eval_one(tr, va, test):
    """Train content + learned head on (tr,va); eval fixed vs learned readout on test, each with
    its EEG-shuffle null. All fits are train-only, so both readouts are admissible."""
    models = [train_content(tr, va, seed=sd) for sd in range(NSEED)]
    w = _fit_head(_perband_corr(models, tr), tr["yslot"])
    rbar, cand = _recon_and_cand(models, test); y = test["yslot"]
    Zte = _corr_from(rbar, cand)
    fs = lambda Z: Z.mean(-1)
    ls = lambda Z: (Z * w).sum(-1)
    fixed, learned = _acc(fs(Zte), y), _acc(ls(Zte), y)
    rng = np.random.default_rng(0); fn, ln = [], []
    for _ in range(20):
        Zn = _corr_from(rbar[rng.permutation(len(y))], cand)
        fn.append(_acc(fs(Zn), y)); ln.append(_acc(ls(Zn), y))
    return dict(fixed=fixed, learned=learned, fixed_null=float(np.mean(fn)),
                learned_null=float(np.mean(ln)), n=len(y))


def run():
    if PROTOCOL == "loso":
        tr_subj = [s for s in ALLSUB if s != SUBJECT]
        Dd = {s: prep(s) for s in ALLSUB}
        # content-disjoint: hold out this subject's held-out CONTENT for test; train/val on the
        # other 15 subjects' remaining content (a val trial_k never appears in train OR test).
        all_tk = sorted({int(k) for s in ALLSUB for k in np.unique(Dd[s]["tk"])})
        rng = np.random.default_rng(42); tks = np.array(all_tk); strat = (tks - 1) % 4
        test_tk, val_tk = set(), set()
        for c in np.unique(strat):
            ks = tks[strat == c].copy(); rng.shuffle(ks)
            nt = max(1, int(round(0.30 * len(ks)))); nv = max(1, int(round(0.15 * len(ks))))
            test_tk.update(ks[:nt].tolist()); val_tk.update(ks[nt:nt + nv].tolist())
        test = _slice(Dd[SUBJECT], test_tk, keep=True)
        tr = {k: np.concatenate([_slice(Dd[s], test_tk | val_tk, keep=False)[k] for s in tr_subj]) for k in Dd[SUBJECT]}
        va = {k: np.concatenate([_slice(Dd[s], val_tk, keep=True)[k] for s in tr_subj]) for k in Dd[SUBJECT]}
        results = [eval_one(tr, va, test)]
    else:                                       # within: per-subject 5-fold content-disjoint
        d = prep(SUBJECT); tks = np.array(sorted(set(d["tk"].tolist()))); strat = (tks - 1) % 4
        rng = np.random.default_rng(42)
        order = {c: rng.permutation(tks[strat == c]) for c in np.unique(strat)}
        folds = [[] for _ in range(5)]
        for c, arr in order.items():
            for i, k in enumerate(arr):
                folds[i % 5].append(int(k))
        results = []
        for f in range(5):
            test_tk = set(folds[f])
            rest = [k for k in tks.tolist() if k not in test_tk]
            rrng = np.random.default_rng(100 + f); rest = np.array(rest); rrng.shuffle(rest)
            val_tk = set(rest[:max(1, int(round(0.15 * len(rest))))].tolist())
            tr = _slice(d, test_tk | val_tk, keep=False); va = _slice(d, val_tk, keep=True)
            test = _slice(d, test_tk, keep=True)
            results.append(eval_one(tr, va, test))
            print(f"  s{SUBJECT} fold{f} done", flush=True)

    n = sum(r["n"] for r in results)
    agg = {k: sum(r[k] * r["n"] for r in results) / n for k in ("fixed", "learned", "fixed_null", "learned_null")}
    row = dict(subject=SUBJECT, protocol=PROTOCOL, n=n, **{f"{k}_acc" if k in ("fixed", "learned") else k: v for k, v in agg.items()})
    out = f"{OUT}_{PROTOCOL}"; os.makedirs(out, exist_ok=True)
    json.dump(row, open(f"{out}/s{SUBJECT}.json", "w"), default=float)
    print(f"[learned-head|{PROTOCOL}|s{SUBJECT}] FIXED={agg['fixed']:.3f}(null {agg['fixed_null']:.3f})  "
          f"LEARNED={agg['learned']:.3f}(null {agg['learned_null']:.3f})  "
          f"Δacc={agg['learned']-agg['fixed']:+.3f}  n={n}", flush=True)


if __name__ == "__main__":
    run()
