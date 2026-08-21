"""K-shot LOSO calibration: how far does a few of the held-out subject's own trials push accuracy?

Train the content decoder on the other 15 subjects (content-disjoint), then FINE-TUNE on K
calibration trials of the held-out subject, and evaluate on that subject's REMAINING trials
(disjoint from the K calibration trials -> still strict, null stays ~0.25). K=0 is pure LOSO.
This is the realistic hearing-device scenario (a short per-user calibration). Reports, per K,
4-way accuracy + EEG-shuffle null. Writes results/kshot_loso/s{S}.json.

  python kshot_loso.py <held_out_subject>
"""
import glob, os, sys, json, numpy as np, torch
import torch.nn.functional as F
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
OUT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad/results/kshot_loso"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 64.0; W = 320
SUBJECT = int(sys.argv[1]); ALLSUB = list(range(1, 17))
NSEED = int(os.environ.get("NSEED", "3")); MARGIN = float(os.environ.get("MARGIN", "0.3"))
KSHOTS = [int(x) for x in os.environ.get("KSHOTS", "0,5,10,20,40").split(",")]


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def prep(s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64); env = z["env"][:, :4].astype(np.float32)
    att = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(att)
    T = eeg.shape[-1]; st = list(range(0, T - W + 1, W // 2)); tri = np.tile(np.arange(N), len(st))
    eeg_z = _zt(np.concatenate([eeg[:, :, a:a + W] for a in st], 0), 2)
    candp = _zt(np.concatenate([env[:, :, :, a:a + W] for a in st], 0).astype(np.float32), -1)
    rng = np.random.default_rng(20260619 + s); perm = np.stack([rng.permutation(4) for _ in range(N)])
    permw = np.concatenate([perm for _ in st], 0); attw = att[tri]
    yslot = np.array([np.flatnonzero(permw[i] == attw[i])[0] for i in range(len(tri))])
    cand = np.stack([candp[i][permw[i]] for i in range(len(candp))])
    return dict(eeg_z=eeg_z, tgt=candp[np.arange(len(tri)), attw], cand=cand, yslot=yslot, tk=tk[tri])


def _slice(a, tk_set, keep=True):
    m = np.isin(a["tk"], list(tk_set)); sel = m if keep else ~m
    return {k: v[sel] for k, v in a.items()}


@torch.no_grad()
def _val_bin(m, a, bs=512):
    m.eval(); rs = []
    for i in range(0, len(a["eeg_z"]), bs):
        r = m(torch.from_numpy(a["eeg_z"][i:i + bs]).to(DEV)).cpu()
        rs.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
    s = B.mm_scores(torch.cat(rs, 0), torch.from_numpy(a["cand"])).numpy(); y = a["yslot"]
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


def _epoch(m, opt, tr, bs=128, rng=None):
    n = len(tr["eeg_z"]); idx = (rng or np.random).permutation(n)
    m.train()
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        r = m(torch.from_numpy(tr["eeg_z"][b]).to(DEV))
        loss = B.neg_pearson_loss(r, torch.from_numpy(tr["tgt"][b]).to(DEV))
        if MARGIN > 0:
            sc = B.mm_scores(r, torch.from_numpy(tr["cand"][b]).to(DEV))
            loss = loss + MARGIN * F.cross_entropy(12.0 * sc, torch.from_numpy(tr["yslot"][b]).to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()


def train_base(tr, va, seed=0, epochs=60, patience=12):
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); best = -1; bstate = None; bad = 0
    for ep in range(epochs):
        _epoch(m, opt, tr, rng=rng)
        vb = _val_bin(m, va)
        if vb > best:
            best = vb; bstate = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(bstate); return m


def finetune(base_state, cal, seed=0, epochs=25, lr=3e-4):
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    m.load_state_dict(base_state)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        _epoch(m, opt, cal, bs=64, rng=rng)
    return m


def evaluate(models, test):
    rbar = _ens_recon(models, test["eeg_z"]); cand = test["cand"]; y = test["yslot"]

    def acc(rb):
        s = B.mm_scores(torch.from_numpy(rb), torch.from_numpy(cand)).numpy()
        return float((s.argmax(1) == y).mean())
    rng = np.random.default_rng(0)
    null = float(np.mean([acc(rbar[rng.permutation(len(y))]) for _ in range(20)]))
    return acc(rbar), null, len(y)


def run():
    raw = {s: prep(s) for s in ALLSUB}
    tr_subj = [s for s in ALLSUB if s != SUBJECT]
    # base training set: other 15 subjects, content-disjoint inner-val
    all_tk = sorted({int(k) for s in tr_subj for k in np.unique(raw[s]["tk"])})
    rng = np.random.default_rng(7); val_tk = set(rng.choice(all_tk, max(1, int(0.15 * len(all_tk))), replace=False))
    base_tr = {k: np.concatenate([_slice(raw[s], val_tk, keep=False)[k] for s in tr_subj]) for k in raw[SUBJECT]}
    base_va = {k: np.concatenate([_slice(raw[s], val_tk, keep=True)[k] for s in tr_subj]) for k in raw[SUBJECT]}
    bases = [train_base(base_tr, base_va, seed=sd) for sd in range(NSEED)]
    base_states = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in bases]
    print(f"  s{SUBJECT} base trained ({NSEED} seeds)", flush=True)

    # held-out subject's trials -> fixed test set (last 60 trial_k); calibration drawn from the rest
    sub_tk = sorted(set(raw[SUBJECT]["tk"].tolist()))
    rng2 = np.random.default_rng(123 + SUBJECT); rng2.shuffle(sub_tk)
    cal_pool = sub_tk[:40]; test_tk = set(sub_tk[40:])                # test = 60 held-out trials, fixed
    test = _slice(raw[SUBJECT], test_tk, keep=True)

    rows = []
    for K in KSHOTS:
        if K == 0:
            models = bases
        else:
            cal_tk = set(cal_pool[:K])
            cal = _slice(raw[SUBJECT], cal_tk, keep=True)
            models = [finetune(base_states[sd], cal, seed=sd) for sd in range(NSEED)]
        real, null, n = evaluate(models, test)
        rows.append(dict(k=K, real=real, null=null, margin=real - null, n=n))
        print(f"[kshot|s{SUBJECT}|K={K}] real={real:.3f} null={null:.3f} margin={real-null:+.3f} n={n}", flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(rows, open(f"{OUT}/s{SUBJECT}.json", "w"), default=float)


if __name__ == "__main__":
    run()
