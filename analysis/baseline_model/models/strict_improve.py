"""Push the strict-regime four-way accuracy (content-disjoint, EEG-shuffle null must stay ~0.25).

Sweeps improvement levers on the VLAAI backward content decoder, one (subject, protocol) per call,
looping configs internally (data loaded once). For each config reports test 4-way + EEG-shuffle
null (fixed Fisher-z correlation readout) at a 5 s operating point.

Levers (env vars set which are swept; default sweeps the round-1 set):
  * ONSET target      : env | onset (relu time-derivative) | both (env+onset, 56-band)
  * HARDNEG           : train the match-mismatch margin against ALL 6 speakers (speakers 5-6 are
                        always-distractor free hard negatives); test stays 4-way.

Content-disjoint splits: within = per-subject 5-fold over trial_k; loso = held-out subject AND
content. Writes results/strict_improve_{protocol}/s{S}.json.

  PROTOCOL=loso python strict_improve.py 1
"""
import glob, os, sys, json, numpy as np, torch
import torch.nn.functional as F
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
OUT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad/results/strict_improve"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 64.0; W = 320
SUBJECT = int(sys.argv[1]); ALLSUB = list(range(1, 17))
NSEED = int(os.environ.get("NSEED", "3")); MARGIN = float(os.environ.get("MARGIN", "0.3"))
PROTOCOL = os.environ.get("PROTOCOL", "loso")
# config sweep: (onset_mode, hardneg). Selectable via CONFIGS env (e.g. "env,onset,both+hardneg").
_CFG = os.environ.get("CONFIGS", "")
if _CFG:
    CONFIGS = [(c.replace("+hardneg", ""), c.endswith("+hardneg")) for c in _CFG.split(",")]
else:
    CONFIGS = [("env", False), ("onset", False), ("both", False), ("env", True), ("both", True)]


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def _onset(x):                                              # relu time-derivative, same length
    d = np.maximum(np.diff(x, axis=-1), 0.0)
    return np.concatenate([np.zeros_like(x[..., :1]), d], axis=-1)


def featize(env_bands, onset_mode):                        # (...,28,W) -> (...,F,W), z per band
    if onset_mode == "env":
        f = env_bands
    elif onset_mode == "onset":
        f = _onset(env_bands)
    else:
        f = np.concatenate([env_bands, _onset(env_bands)], axis=-2)
    return _zt(f, -1)


def prep(s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64); env6 = z["env"][:, :6].astype(np.float32)
    att = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(att)
    T = eeg.shape[-1]; st = list(range(0, T - W + 1, W // 2)); tri = np.tile(np.arange(N), len(st))
    eeg_z = _zt(np.concatenate([eeg[:, :, a:a + W] for a in st], 0), 2)
    env6w = np.concatenate([env6[:, :, :, a:a + W] for a in st], 0)          # (Nw,6,28,W) raw
    r4 = np.random.default_rng(20260619 + s); perm4 = np.stack([r4.permutation(4) for _ in range(N)])
    r6 = np.random.default_rng(777 + s); perm6 = np.stack([r6.permutation(6) for _ in range(N)])
    return dict(eeg_z=eeg_z, env6=env6w, perm4=perm4[tri], perm6=perm6[tri], att=att[tri], tk=tk[tri])


def build(d, onset_mode):
    feat6 = featize(d["env6"], onset_mode)                                   # (Nw,6,F,W)
    att = d["att"]; Nw = len(att)
    tgt = feat6[np.arange(Nw), att]                                          # (Nw,F,W)
    cand4 = np.take_along_axis(feat6, d["perm4"][:, :, None, None], axis=1)  # (Nw,4,F,W)
    cand6 = np.take_along_axis(feat6, d["perm6"][:, :, None, None], axis=1)  # (Nw,6,F,W)
    y4 = np.argmax(d["perm4"] == att[:, None], axis=1)
    y6 = np.argmax(d["perm6"] == att[:, None], axis=1)
    return dict(eeg_z=d["eeg_z"], tgt=tgt, cand4=cand4, y4=y4, cand6=cand6, y6=y6, tk=d["tk"])


def _slice(a, tk_set, keep=True):
    m = np.isin(a["tk"], list(tk_set)); sel = m if keep else ~m
    return {k: (v[sel] if hasattr(v, "shape") and v.shape[:1] == a["tk"].shape else v) for k, v in a.items()}


@torch.no_grad()
def _val_bin(m, a):
    m.eval(); r = _ens_recon([m], a["eeg_z"])
    s = B.mm_scores(torch.from_numpy(r), torch.from_numpy(a["cand4"])).numpy(); y = a["y4"]
    sm = s[np.arange(len(y)), y][:, None]
    return float(((sm > s).sum(1) / 3).mean())


@torch.no_grad()
def _ens_recon(models, eeg, bs=512):
    rs = []
    for m in models:
        m.eval(); out = []
        for i in range(0, len(eeg), bs):
            r = m(torch.from_numpy(eeg[i:i + bs]).to(DEV)).cpu()
            out.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
        rs.append(torch.cat(out, 0))
    return torch.stack(rs).mean(0).numpy()


def train_content(tr, va, hardneg, n_out, seed=0, epochs=60, patience=12):
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=n_out).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(tr["eeg_z"]); rng = np.random.default_rng(seed); best = -1; bstate = None; bad = 0
    ck, cy = ("cand6", "y6") if hardneg else ("cand4", "y4")
    for ep in range(epochs):
        m.train(); idx = rng.permutation(n)
        for i in range(0, n, 128):
            b = idx[i:i + 128]
            r = m(torch.from_numpy(tr["eeg_z"][b]).to(DEV))
            loss = B.neg_pearson_loss(r, torch.from_numpy(tr["tgt"][b]).to(DEV))
            if MARGIN > 0:
                sc = B.mm_scores(r, torch.from_numpy(tr[ck][b]).to(DEV))
                loss = loss + MARGIN * F.cross_entropy(12.0 * sc, torch.from_numpy(tr[cy][b]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        vb = _val_bin(m, va)
        if vb > best:
            best = vb; bstate = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(bstate); return m


def eval_cfg(tr, va, test, hardneg, n_out):
    models = [train_content(tr, va, hardneg, n_out, seed=sd) for sd in range(NSEED)]
    rbar = _ens_recon(models, test["eeg_z"]); cand = test["cand4"]; y = test["y4"]

    def acc4(rb):
        s = B.mm_scores(torch.from_numpy(rb), torch.from_numpy(cand)).numpy()
        return float((s.argmax(1) == y).mean())
    real = acc4(rbar)
    rng = np.random.default_rng(0)
    null = float(np.mean([acc4(rbar[rng.permutation(len(y))]) for _ in range(20)]))
    return real, null, len(y)


def splits(feat):
    """Yield (tr, va, test) content-disjoint dicts."""
    if PROTOCOL == "loso":
        yield feat["tr"], feat["va"], feat["test"]
    else:
        for f in feat["folds"]:
            yield f


def run():
    if PROTOCOL == "loso":
        raw = {s: prep(s) for s in ALLSUB}
        all_tk = sorted({int(k) for s in ALLSUB for k in np.unique(raw[s]["tk"])})
        rng = np.random.default_rng(42); tks = np.array(all_tk); strat = (tks - 1) % 4
        test_tk, val_tk = set(), set()
        for c in np.unique(strat):
            ks = tks[strat == c].copy(); rng.shuffle(ks)
            nt = max(1, int(round(0.30 * len(ks)))); nv = max(1, int(round(0.15 * len(ks))))
            test_tk.update(ks[:nt].tolist()); val_tk.update(ks[nt:nt + nv].tolist())
        tr_subj = [s for s in ALLSUB if s != SUBJECT]
    else:
        raw = {SUBJECT: prep(SUBJECT)}

    rows = []
    for onset_mode, hardneg in CONFIGS:
        n_out = {"env": 28, "onset": 28, "both": 56}[onset_mode]
        if PROTOCOL == "loso":
            bf = {s: build(raw[s], onset_mode) for s in ALLSUB}
            test = _slice(bf[SUBJECT], test_tk, keep=True)
            tr = {k: (np.concatenate([_slice(bf[s], test_tk | val_tk, keep=False)[k] for s in tr_subj])
                      if hasattr(bf[SUBJECT][k], "shape") and bf[SUBJECT][k].shape[:1] == bf[SUBJECT]["tk"].shape else bf[SUBJECT][k])
                  for k in bf[SUBJECT]}
            va = {k: (np.concatenate([_slice(bf[s], val_tk, keep=True)[k] for s in tr_subj])
                      if hasattr(bf[SUBJECT][k], "shape") and bf[SUBJECT][k].shape[:1] == bf[SUBJECT]["tk"].shape else bf[SUBJECT][k])
                  for k in bf[SUBJECT]}
            real, null, n = eval_cfg(tr, va, test, hardneg, n_out)
            n_tot = n
        else:
            bf = build(raw[SUBJECT], onset_mode)
            tks = np.array(sorted(set(bf["tk"].tolist()))); strat = (tks - 1) % 4
            rng = np.random.default_rng(42); order = {c: rng.permutation(tks[strat == c]) for c in np.unique(strat)}
            folds = [[] for _ in range(5)]
            for c, arr in order.items():
                for i, k in enumerate(arr):
                    folds[i % 5].append(int(k))
            reals, nulls, ns = [], [], []
            for f in range(5):
                test_tk = set(folds[f]); rest = np.array([k for k in tks.tolist() if k not in test_tk])
                np.random.default_rng(100 + f).shuffle(rest)
                val_tk = set(rest[:max(1, int(round(0.15 * len(rest))))].tolist())
                tr = _slice(bf, test_tk | val_tk, keep=False); va = _slice(bf, val_tk, keep=True)
                test = _slice(bf, test_tk, keep=True)
                r, nu, n = eval_cfg(tr, va, test, hardneg, n_out); reals.append(r); nulls.append(nu); ns.append(n)
            n_tot = sum(ns)
            real = sum(r * n for r, n in zip(reals, ns)) / n_tot; null = sum(x * n for x, n in zip(nulls, ns)) / n_tot
        tag = onset_mode + ("+hardneg" if hardneg else "")
        rows.append(dict(config=tag, onset=onset_mode, hardneg=hardneg, real=real, null=null,
                         margin=real - null, n=n_tot))
        print(f"[strict|{PROTOCOL}|s{SUBJECT}|{tag}] real={real:.3f} null={null:.3f} margin={real-null:+.3f} n={n_tot}", flush=True)
    out = f"{OUT}_{PROTOCOL}"; os.makedirs(out, exist_ok=True)
    json.dump(rows, open(f"{out}/s{SUBJECT}.json", "w"), default=float)


if __name__ == "__main__":
    run()
