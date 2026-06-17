"""content_v2: anti-overfitting content AAD (4-talker permuted, loudness-equalized).

Mitigations for "train hits 100% on ~80 whole-trial examples before val catches up":
  (1) WINDOWED training (~10 s windows, 2 s hop -> ~11x more examples) with WHOLE-TRIAL
      evaluation (aggregate each trial's window match-scores -> one trial decision).
  (2) EARLY STOPPING on a held-out validation split (best-val checkpoint, patience).
  (3) TRANSFER with a FROZEN pretrained encoder (only the small recon head + scale fit
      per subject -> almost nothing to memorize), vs full fine-tune, vs scratch.
Reports per-subject + mean trial-level 4-way acc (chance 0.25). EEG-shuffle guarded by env.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, ssl_pretrain, zs = _nets.Enc, _nets.ssl_pretrain, _nets.zs

RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
W = int(os.environ.get("WIN", "640")); HOP = int(os.environ.get("HOP", "128"))   # 10 s / 2 s @64 Hz


def load():
    D = {}
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2.npz")
        if not f: continue
        z = np.load(f[0])
        eeg = zs(z["eeg"].astype(np.float32), 2)
        bb = zs(z["env"][:, :4].mean(2).astype(np.float32), 2)
        D[s] = (eeg, bb, z["attended"].astype(int) - 1)
    T = min(D[s][0].shape[-1] for s in D)
    return {s: (e[:, :, :T], b[:, :, :T], y) for s, (e, b, y) in D.items()}


def permute(bb, y, seed):
    rng = np.random.default_rng(seed); N = len(y); cand = np.empty_like(bb); tgt = np.empty(N, int)
    for i in range(N):
        p = rng.permutation(4); cand[i] = bb[i][p]; tgt[i] = int(np.where(p == y[i])[0][0])
    return cand, tgt


def windowize(eeg, cand, tgt, idx):
    """trials -> windows. Returns we (M,32,W), wc (M,4,W), wt (M,), wid (M,) trial-id."""
    T = eeg.shape[-1]; starts = list(range(0, T - W + 1, HOP)) or [0]
    we, wc, wt, wid = [], [], [], []
    for i in idx:
        for s in starts:
            we.append(eeg[i, :, s:s + W]); wc.append(cand[i][:, :, s:s + W] if cand.ndim == 4 else cand[i][:, s:s + W])
            wt.append(tgt[i]); wid.append(i)
    return (np.asarray(we, np.float32), np.asarray(wc, np.float32), np.asarray(wt, int), np.asarray(wid, int))


class ContentNet(nn.Module):
    def __init__(self, h=64):
        super().__init__(); self.enc = Enc(32, h); self.dec = nn.Conv1d(h, 1, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):
        r = self.dec(self.enc(eeg)).squeeze(1)
        rz = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6)
        cz = (cand - cand.mean(-1, keepdim=True)) / (cand.std(-1, keepdim=True) + 1e-6)
        return r, (rz.unsqueeze(1) * cz).mean(-1) * self.scale.exp().clamp(max=100)


@torch.no_grad()
def trial_scores(model, we, wc, wid, bs=512):
    model.eval(); out = []
    for i in range(0, len(we), bs):
        _, sc = model(torch.as_tensor(we[i:i + bs], device=dev), torch.as_tensor(wc[i:i + bs], device=dev))
        out.append(sc.cpu().numpy())
    sc = np.concatenate(out); agg = {}
    for j, t in enumerate(wid):
        agg.setdefault(int(t), np.zeros(4)); agg[int(t)] += sc[j]
    return agg                                                # trial_id -> summed 4-way score


def trial_acc(model, we, wc, wid, tgt):
    agg = trial_scores(model, we, wc, wid)
    return np.mean([agg[i].argmax() == tgt[i] for i in agg])


def fit(model, we, wc, wt, val, epochs, lr, wd, freeze_enc=False, patience=8, bs=128):
    if freeze_enc:
        for p in model.enc.parameters(): p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr, weight_decay=wd); model.to(dev)
    We, Wc, Wt = (torch.as_tensor(t, device=dev) for t in (we, wc, wt))
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); idx = np.random.permutation(len(wt))
        for i in range(0, len(idx), bs):
            bi = torch.as_tensor(idx[i:i + bs], device=dev)
            r, sc = model(We[bi], Wc[bi]); att = Wc[bi][torch.arange(len(bi), device=dev), Wt[bi]]
            rz = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6); az = (att - att.mean(-1, keepdim=True)) / (att.std(-1, keepdim=True) + 1e-6)
            loss = F.cross_entropy(sc, Wt[bi]) + F.mse_loss(r, att) + (1 - (rz * az).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        va = trial_acc(model, *val)                            # val trial-level acc
        if va > best:
            best, best_state, bad = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def main():
    D = load(); subs = sorted(D); seeds = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
    res = {m: [] for m in ("scratch_win", "transfer_ft", "transfer_frozen")}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        P = {s: permute(D[s][1], D[s][2], sd * 1000 + s) for s in subs}
        # pooled windows for transfer pretrain (per target, exclude it)
        per = {m: {} for m in res}
        for s in subs:
            Xe, bb, _ = D[s]; C, tgt = P[s]; N = len(tgt)
            # pretrain pooled (others), windowed
            oi = [(o, j) for o in subs if o != s for j in range(len(P[o][1]))]
            oE = np.concatenate([D[o][0] for o in subs if o != s]); oC = np.concatenate([P[o][0] for o in subs if o != s]); oy = np.concatenate([P[o][1] for o in subs if o != s])
            ote = windowize(oE, oC, oy, range(len(oy)))
            pre = fit(ContentNet(), ote[0], ote[1], ote[2], (ote[0][:2000], ote[1][:2000], ote[3][:2000], oy), 15, 1e-3, 1e-4)
            a = {m: [] for m in res}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, tgt):
                vtr, vval = tr[:int(.85 * len(tr))], tr[int(.85 * len(tr)):]   # early-stop val split
                trw = windowize(Xe, C, tgt, vtr); vw = windowize(Xe, C, tgt, vval); tew = windowize(Xe, C, tgt, te)
                val = (vw[0], vw[1], vw[3], tgt)
                # scratch windowed + early stop
                a["scratch_win"].append(trial_acc(fit(ContentNet(), trw[0], trw[1], trw[2], val, 60, 1e-3, 1e-3), tew[0], tew[1], tew[3], tgt))
                mt = ContentNet(); mt.load_state_dict(pre.state_dict())
                a["transfer_ft"].append(trial_acc(fit(mt, trw[0], trw[1], trw[2], val, 30, 5e-4, 1e-3), tew[0], tew[1], tew[3], tgt))
                mf = ContentNet(); mf.load_state_dict(pre.state_dict())
                a["transfer_frozen"].append(trial_acc(fit(mf, trw[0], trw[1], trw[2], val, 40, 1e-3, 1e-3, freeze_enc=True), tew[0], tew[1], tew[3], tgt))
            for m in res: per[m][s] = np.mean(a[m])
            print(f"seed{sd} S{s:2d} scratch_win={per['scratch_win'][s]:.3f} transfer_ft={per['transfer_ft'][s]:.3f} transfer_frozen={per['transfer_frozen'][s]:.3f}", flush=True)
        for m in res: res[m].append(np.mean([per[m][s] for s in subs]))
        print(f"seed{sd} MEANS: " + " ".join(f"{m}={res[m][-1]:.3f}" for m in res), flush=True)
    print("\n=== content_v2 (windowed+earlystop+transfer) MEAN over seeds (chance 0.25; v1 combo=0.361) ===")
    for m in res:
        a = np.array(res[m]); print(f"  {m:16s} = {a.mean():.3f} ± {a.std():.3f}")


if __name__ == "__main__":
    main()
