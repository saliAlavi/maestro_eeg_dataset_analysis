"""content_2stage: two-stage frozen multi-path attended-speaker matcher, INTRA-SUBJECT.

Per the de-confounding analysis: the attended-talker audio confound is STATIC (power/SNR), and
TEMPORAL-correlation matching against a FROZEN audio embedding neutralizes it (a constant/uninformative
EEG path -> ~0 corr -> can't exploit the static prior). Design:

  STAGE 1 (frozen): per-branch EEG matchers, each trained then FROZEN ->
    content branches  : ImprovedEnc -> decode to feature dim -> TEMPORAL Pearson corr vs the FROZEN
                        candidate feature (envelope / spectral / auditory-w2v / semantic). 4 logits/speaker.
    directional branch: channel-mixing Enc (alpha-lateralization) + gaze orienting stats -> 4 logits.
  STAGE 2 (learned, low-capacity): fuse the frozen per-branch posteriors. FUSE=avg (param-free) or
    logistic (OOF-fit inner 2-fold so the content branches' train-overfit can't bias it).

Within-subject 5-fold, trial-disjoint. chance 0.25. SHUFFLE_EEG=1 = full-pipeline null (must be chance).
FEATS selects branches. Reports per-branch + fused, and (with SHUFFLE_EEG) the null.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
_c6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_v6.py")
_s6 = importlib.util.spec_from_file_location("c6", _c6); V6 = importlib.util.module_from_spec(_s6); _s6.loader.exec_module(V6)
_nets_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_np = importlib.util.spec_from_file_location("tn", _nets_p); _nets = importlib.util.module_from_spec(_np); _np.loader.exec_module(_nets)
ImprovedEnc, _corr, zs = V6.ImprovedEnc, V6._corr, V6.zs
BasicEnc = _nets.Enc
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"; H = 64
FEATS = os.environ.get("FEATS", "env,spec,w2v,sem,dir").split(",")
FUSE = os.environ.get("FUSE", "avg")
SHUF = os.environ.get("SHUFFLE_EEG", "0") == "1"          # break EEG<->label (content validation)
SHUF_IN = os.environ.get("SHUFFLE_INPUT", "0") == "1"    # break EEG AND gaze<->label (full-pipeline null)
PERMUTE = os.environ.get("PERMUTE", "0") == "1"          # per-trial candidate shuffle+relabel -> label
#                                                          = attended STREAM, decoupled from position
CONTENT = [f for f in FEATS if f != "dir"]


def _stats(x):
    return np.concatenate([x.mean(1), x.std(1), np.percentile(x, 10, 1), np.percentile(x, 90, 1)], 1)


def load():
    col = {"env": lambda z: z["env"][:, :4].mean(2, keepdims=True).astype(np.float32),
           "spec": lambda z: z["env"][:, :4].astype(np.float32),
           "w2v": lambda z: z["w2v"].astype(np.float32),
           "sem": lambda z: z["sem"].astype(np.float32)}
    E = []; CD = {k: [] for k in CONTENT}; G = []; Y = []; TK = []; SB = []; rng0 = np.random.default_rng(12345)
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        g = zs(_stats(z["gaze"].astype(np.float32)).astype(np.float32), 0)
        # PHYSICAL-speaker order (no permutation): the temporal-corr recon matcher is inherently
        # position-agnostic, so position can't leak; this keeps every branch (incl. directional) in
        # physical-speaker space so they fuse. The shuffle-null guards against any residual prior.
        cand = {k: zs(col[k](z), -1) for k in CONTENT}
        if SHUF or SHUF_IN: eeg = eeg[rng0.permutation(N)]   # EEG-shuffle (and part of input-shuffle)
        if SHUF_IN: g = g[rng0.permutation(N)]               # also break gaze<->label -> full-pipeline null
        E.append(eeg); [CD[k].append(cand[k]) for k in CONTENT]; G.append(g); Y.append(y); TK.append(tk); SB.append(np.full(N, s - 1))
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); CD = {k: np.concatenate([c[:, :, :, :T] for c in v]) for k, v in CD.items()}
    G = np.concatenate(G); Y = np.concatenate(Y); TK = np.concatenate(TK); SB = np.concatenate(SB)
    if PERMUTE:
        # Per-trial PERMUTE of the 4 candidate streams + relabel to the attended stream's new
        # slot. The label becomes "which STREAM" (pure content), DECOUPLED from physical speaker
        # position/spatial identity -> content and the spatial-label task are not entangled.
        rng = np.random.default_rng(20260619)
        for i in range(len(Y)):
            p = rng.permutation(4)
            for k in CONTENT:
                CD[k][i] = CD[k][i][p]
            Y[i] = int(np.flatnonzero(p == Y[i])[0])
    return E, CD, G, Y, TK, SB


class FeatNet(nn.Module):                                  # temporal-corr matcher vs frozen candidate
    def __init__(self, dim):
        super().__init__(); self.enc = ImprovedEnc(32, H, film=False); self.dec = nn.Conv1d(H, dim, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):
        return _corr(self.dec(self.enc(eeg)), cand) * self.scale.exp().clamp(max=100)


class DirNet(nn.Module):                                   # directional / spatial branch
    def __init__(self, gdim):
        super().__init__(); self.enc = BasicEnc(32, H); self.head = nn.Sequential(nn.Linear(H + gdim, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, 4))

    def forward(self, eeg, g):
        return self.head(torch.cat([self.enc(eeg).mean(-1), g], 1))


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def _train(model, inputs, y, ep, lr=1e-3, bs=48):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    ten = [torch.as_tensor(x, device=dev) for x in inputs]; yt = torch.as_tensor(y, device=dev)
    for _ in range(ep):
        for b in _bt(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(model(*[t[bi] for t in ten]), yt[bi], label_smoothing=0.05).backward(); opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def _post(model, inputs, bs=48):
    model.eval(); P = []
    n = len(inputs[0])
    for i in range(0, n, bs):
        P.append(F.softmax(model(*[torch.as_tensor(x[i:i + bs], device=dev) for x in inputs]), 1).cpu().numpy())
    return np.concatenate(P)


def branch_inputs(E, CD, G, k, idx):
    return [E[idx], CD[k][idx]] if k != "dir" else [E[idx], G[idx]]


def fit_branch(E, CD, G, Y, k, idx, ep, init=None, lr=1e-3):
    inp = branch_inputs(E, CD, G, k, idx)
    m = FeatNet(CD[k].shape[2]) if k != "dir" else DirNet(G.shape[1])
    if init is not None:
        m.load_state_dict(init)
    return _train(m, inp, Y[idx], ep, lr=lr)


def pretrain_content(E, CD, G, Y, SB, TK, s, ep_pre, exclude_tk=None):
    """Supervised cross-subject pretrain of CONTENT branches on OTHER subjects.
    TRIAL-DISJOINT: all subjects share the same clips + attended schedule, so the pool
    MUST exclude the held-out subject's TEST trial_k's -- otherwise a learnable audio path
    memorises the test clips (the w2v leak). dir stays per-subject. Returns init states."""
    mask = (SB != s)
    if exclude_tk is not None:
        mask &= ~np.isin(TK, np.asarray(list(exclude_tk)))
    pool = np.where(mask)[0]
    states = {}
    for k in CONTENT:
        m = fit_branch(E, CD, G, Y, k, pool, ep_pre)
        states[k] = {kk: v.detach().cpu().clone() for kk, v in m.state_dict().items()}
    return states


def post_branch(m, E, CD, G, k, idx):
    return _post(m, branch_inputs(E, CD, G, k, idx))


def _bcfg(k, pre, ep, ft_ep):
    """Per-branch (epochs, lr, init): pretrained content branches fine-tune (fewer ep,
    lower lr) from the cross-subject init; everything else trains fresh."""
    if pre and k in pre:
        return ft_ep, 5e-4, pre[k]
    return ep, 1e-3, None


def evaluate(E, CD, G, Y, tr, te, ep, pre=None, ft_ep=12):
    yte = Y[te]
    test_post = {}; out = {}
    for k in FEATS:
        e_, lr_, init_ = _bcfg(k, pre, ep, ft_ep)
        m = fit_branch(E, CD, G, Y, k, tr, e_, init=init_, lr=lr_)
        test_post[k] = post_branch(m, E, CD, G, k, te)
        out[k] = (test_post[k].argmax(1) == yte).mean()
    # stage-2 fusion of FROZEN branch posteriors
    avg = sum(test_post[k] for k in FEATS) / len(FEATS)
    out["avg"] = (avg.argmax(1) == yte).mean()
    if FUSE in ("logistic", "rel"):
        oof = {k: np.zeros((len(tr), 4), np.float32) for k in FEATS}
        for it, iv in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, Y[tr]):
            a, b = tr[it], tr[iv]
            for k in FEATS:
                e_, lr_, init_ = _bcfg(k, pre, ep, ft_ep)
                oof[k][iv] = post_branch(fit_branch(E, CD, G, Y, k, a, e_, init=init_, lr=lr_), E, CD, G, k, b)
        if FUSE == "logistic":
            Xtr = np.concatenate([np.log(oof[k] + 1e-9) for k in FEATS], 1)
            Xte = np.concatenate([np.log(test_post[k] + 1e-9) for k in FEATS], 1)
            lr = LogisticRegression(max_iter=1000).fit(Xtr, Y[tr])
            out["fused"] = (lr.predict(Xte) == yte).mean()
        else:  # rel: weight each branch by its OOF accuracy EXCESS over chance (per subject)
            w = {k: max(0.0, (oof[k].argmax(1) == Y[tr]).mean() - 0.25) for k in FEATS}
            tot = sum(w.values()) or 1.0
            fused = sum((w[k] / tot) * test_post[k] for k in FEATS)
            out["fused"] = (fused.argmax(1) == yte).mean()
            out["_relw"] = {k: w[k] / tot for k in FEATS}
    else:
        out["fused"] = out["avg"]
    return out


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, CD, G, Y, TK, SB = load(); ep = int(os.environ.get("EPOCHS", "30"))
    PRETRAIN = os.environ.get("PRETRAIN", "0") == "1"
    pre_ep = int(os.environ.get("PRE_EPOCHS", "25")); ft_ep = int(os.environ.get("FT_EPOCHS", "12"))
    print(f"FEATS={FEATS} FUSE={FUSE} SHUF={SHUF} ep={ep} PRETRAIN={PRETRAIN} "
          f"pre_ep={pre_ep} ft_ep={ft_ep} n={len(Y)}  (INTRA-SUBJECT 5-fold, chance 0.25)", flush=True)
    rows = []; relws = []
    for s in np.unique(SB):
        m_ = SB == s; idx = np.where(m_)[0]; Ys = Y[idx]
        accs = {k: [] for k in FEATS + ["avg", "fused"]}
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, Ys):
            # TRIAL-DISJOINT pretrain: exclude this fold's test trial_k's from the pool
            pre = (pretrain_content(E, CD, G, Y, SB, TK, s, pre_ep, exclude_tk=set(TK[idx[te_i]]))
                   if PRETRAIN else None)
            r = evaluate(E, CD, G, Y, idx[tr_i], idx[te_i], ep, pre=pre, ft_ep=ft_ep)
            for k in accs: accs[k].append(r[k])
            if "_relw" in r: relws.append(r["_relw"])
        row = {k: float(np.mean(v)) for k, v in accs.items()}; rows.append(row)
        print("  S%2d  " % (int(s) + 1) + " ".join(f"{k}={row[k]:.3f}" for k in FEATS) + f"  avg={row['avg']:.3f} FUSED={row['fused']:.3f}", flush=True)
    print(f"\n=== 2-STAGE intra-subject (chance 0.25) {'<-- FULL-PIPELINE EEG-SHUFFLE NULL' if SHUF else ''} ===")
    for k in FEATS + ["avg", "fused"]:
        v = np.array([r[k] for r in rows]); print(f"  {k:6s} = {v.mean():.3f} +- {v.std():.3f}")
    if relws:
        print("  mean reliability weights: " + " ".join(f"{k}={np.mean([w[k] for w in relws]):.3f}" for k in FEATS))


if __name__ == "__main__":
    main()
