"""content_fusion: LATE FUSION of the content matcher and the spatial decoder for the
4-class attended-SPEAKER task (T3, chance 0.25). The dataset's whole point is multimodal,
and content (EEG<->speech) and spatial (EEG alpha-lateralization + gaze/IMU overt orienting)
are DISSOCIATED across subjects -> fusing complementary branches should beat either alone.

  CONTENT branch : ImprovedEnc -> reconstruct 28-band spec -> Pearson corr with each of the
                   4 PHYSICAL speakers' spec (position-agnostic, leak-proof recon matcher).
  SPATIAL branch : channel-mixing Enc (alpha-lateralization) -> global pool + gaze[+IMU]
                   orienting stats -> 4-way head.  (the established ~0.47-0.55 recipe)
  FUSION         : log p_fused = a*log p_content + (1-a)*log p_spatial; a swept on TRAIN.

REGIME=trial|loso. MODS controls spatial side feats: gaze,imu. SHUFFLE_INPUT=1 breaks all
inputs<->label (null, must be chance 0.25). Reports content-only / spatial-only / fused.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
BasicEnc, zs = _nets.Enc, _nets.zs
_c6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_v6.py")
_s6 = importlib.util.spec_from_file_location("c6", _c6); V6 = importlib.util.module_from_spec(_s6); _s6.loader.exec_module(V6)
ImprovedEnc, _corr = V6.ImprovedEnc, V6._corr
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"; H = 64
REGIME = os.environ.get("REGIME", "trial")
MODS = os.environ.get("MODS", "gaze,imu").split(",")
SHUF = os.environ.get("SHUFFLE_INPUT", "0") == "1"


def _stats(x):                                   # (N,T,d) -> (N,4d) mean/std/p10/p90 over time
    return np.concatenate([x.mean(1), x.std(1), np.percentile(x, 10, 1), np.percentile(x, 90, 1)], 1)


def load():
    E, C, G, Y, TK, SB = [], [], [], [], [], []; rng0 = np.random.default_rng(12345)
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        spec = zs(z["env"][:, :4].astype(np.float32), -1)          # (N,4,28,T) PHYSICAL order
        g = []
        if "gaze" in MODS: g.append(_stats(z["gaze"].astype(np.float32)))
        if "imu" in MODS: g.append(_stats(z["imu"].astype(np.float32)))
        g = np.concatenate(g, 1) if g else np.zeros((len(eeg), 0), np.float32)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        if SHUF:                                                   # break (EEG+gaze+spec)<->label
            pe = rng0.permutation(N); eeg = eeg[pe]; spec = spec[pe]; g = g[pe]
        E.append(eeg); C.append(spec); G.append(g); Y.append(y); TK.append(tk); SB.append(np.full(N, s - 1))
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); C = np.concatenate([c[:, :, :, :T] for c in C])
    G = zs(np.concatenate(G).astype(np.float32), 0)
    return E, C, G, np.concatenate(Y), np.concatenate(TK), np.concatenate(SB)


class ContentNet(nn.Module):
    def __init__(self, nb):
        super().__init__(); self.enc = ImprovedEnc(32, H, film=False); self.dec = nn.Conv1d(H, nb, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):
        return _corr(self.dec(self.enc(eeg)), cand) * self.scale.exp().clamp(max=100)


class SpatialNet(nn.Module):
    def __init__(self, gdim):
        super().__init__(); self.enc = BasicEnc(32, H); self.head = nn.Sequential(nn.Linear(H + gdim, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, 4))

    def forward(self, eeg, g):
        return self.head(torch.cat([self.enc(eeg).mean(-1), g], 1))


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train_content(E, C, y, ep=25, lr=1e-3, bs=48):
    m = ContentNet(C.shape[2]).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    E = torch.as_tensor(E, device=dev); C = torch.as_tensor(C, device=dev); y = torch.as_tensor(y, device=dev)
    for _ in range(ep):
        for b in _bt(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(m(E[bi], C[bi]), y[bi]).backward(); opt.step(); opt.zero_grad()
    return m


def train_spatial(E, G, y, ep=60, lr=1e-3, bs=64):
    m = SpatialNet(G.shape[1]).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    E = torch.as_tensor(E, device=dev); G = torch.as_tensor(G, device=dev); y = torch.as_tensor(y, device=dev)
    for _ in range(ep):
        for b in _bt(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(m(E[bi], G[bi]), y[bi], label_smoothing=0.05).backward(); opt.step(); opt.zero_grad()
    return m


@torch.no_grad()
def _post_c(m, E, C, bs=48):
    m.eval(); P = []
    for i in range(0, len(E), bs):
        P.append(F.softmax(m(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(C[i:i + bs], device=dev)), 1).cpu().numpy())
    return np.concatenate(P)


@torch.no_grad()
def _post_s(m, E, G, bs=64):
    m.eval(); P = []
    for i in range(0, len(E), bs):
        P.append(F.softmax(m(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(G[i:i + bs], device=dev)), 1).cpu().numpy())
    return np.concatenate(P)


def _fuse_alpha(pc, ps, y):
    """pick a in [0,1] maximizing fused acc on the GIVEN (out-of-fold) posteriors."""
    best, ba = -1, 0.5
    for a in np.linspace(0, 1, 21):
        acc = (np.argmax(a * np.log(pc + 1e-9) + (1 - a) * np.log(ps + 1e-9), 1) == y).mean()
        if acc > best: best, ba = acc, a
    return ba


def _oof_posteriors(E, C, G, Y, idx_tr, k=3):
    """inner k-fold OOF content/spatial posteriors on the TRAIN set -> honest fusion fit
    (in-sample train posteriors overfit toward the content branch and break alpha)."""
    from sklearn.model_selection import StratifiedKFold
    pc = np.zeros((len(idx_tr), 4), np.float32); ps = np.zeros((len(idx_tr), 4), np.float32)
    for it, iv in StratifiedKFold(k, shuffle=True, random_state=0).split(idx_tr, Y[idx_tr]):
        a, b = idx_tr[it], idx_tr[iv]
        mc = train_content(E[a], C[a], Y[a]); ms = train_spatial(E[a], G[a], Y[a])
        pc[iv] = _post_c(mc, E[b], C[b]); ps[iv] = _post_s(ms, E[b], G[b])
    return pc, ps


def evaluate(E, C, G, Y, idx_tr, idx_te):
    yte = Y[idx_te]
    # honest fusion fit on OUT-OF-FOLD train posteriors
    pc_oof, ps_oof = _oof_posteriors(E, C, G, Y, idx_tr)
    # final branches: retrain on the FULL train fold, score test
    mc = train_content(E[idx_tr], C[idx_tr], Y[idx_tr]); ms = train_spatial(E[idx_tr], G[idx_tr], Y[idx_tr])
    pc_te, ps_te = _post_c(mc, E[idx_te], C[idx_te]), _post_s(ms, E[idx_te], G[idx_te])
    out = {"content": (pc_te.argmax(1) == yte).mean(), "spatial": (ps_te.argmax(1) == yte).mean()}
    # (a) alpha fusion, alpha tuned on OOF
    a = _fuse_alpha(pc_oof, ps_oof, Y[idx_tr]); out["alpha"] = a
    out["fused"] = (np.argmax(a * np.log(pc_te + 1e-9) + (1 - a) * np.log(ps_te + 1e-9), 1) == yte).mean()
    # (b) logistic stacker on OOF log-posteriors (8 feats -> 4 class)
    from sklearn.linear_model import LogisticRegression
    Xtr = np.concatenate([np.log(pc_oof + 1e-9), np.log(ps_oof + 1e-9)], 1)
    Xte = np.concatenate([np.log(pc_te + 1e-9), np.log(ps_te + 1e-9)], 1)
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, Y[idx_tr])
        out["stack"] = (lr.predict(Xte) == yte).mean()
    except Exception:
        out["stack"] = out["fused"]
    return out


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, G, Y, TK, SB = load()
    print(f"REGIME={REGIME} MODS={MODS} gdim={G.shape[1]} SHUFFLE_INPUT={SHUF} n={len(Y)}", flush=True)
    rows = []
    if REGIME == "loso":
        for s in np.unique(SB):
            r = evaluate(E, C, G, Y, np.where(SB != s)[0], np.where(SB == s)[0]); rows.append(r)
            print(f"  held-out S{int(s)+1:2d}  content={r['content']:.3f} spatial={r['spatial']:.3f} fused={r['fused']:.3f} stack={r['stack']:.3f} (a={r['alpha']:.2f})", flush=True)
    else:
        trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
        for fi, te_tr in enumerate(folds):
            te = np.where(np.isin(TK, te_tr))[0]; tr = np.where(~np.isin(TK, te_tr))[0]
            r = evaluate(E, C, G, Y, tr, te); rows.append(r)
            print(f"  fold{fi}  content={r['content']:.3f} spatial={r['spatial']:.3f} fused={r['fused']:.3f} stack={r['stack']:.3f} (a={r['alpha']:.2f})", flush=True)
    def m(k): return np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])
    print(f"\n=== FUSION {REGIME} (chance 0.25; OOF-tuned) {'<-- INPUT-SHUFFLE NULL' if SHUF else ''} ===")
    for k in ("content", "spatial", "fused", "stack"):
        mu, sd = m(k); print(f"  {k:8s} = {mu:.3f} +- {sd:.3f}")


if __name__ == "__main__":
    main()
