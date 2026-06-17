"""content_cfuse: a FUSED CONTENT network -- OOF-stacked ensemble of per-feature content
matchers (spectrogram / HuBERT-w2v / GPT-2 semantic), each an INDEPENDENT recon matcher.

Motivation: the old multi-target net (one encoder, learned softmax mixture over feature corrs)
did NOT survive trial-disjoint (0.355 ~ spec-only 0.356) -- an in-network mixture overfits, just
like in-sample alpha did for content+spatial fusion. The fix that worked there: train branches
INDEPENDENTLY and combine on OUT-OF-FOLD posteriors. This applies that to content's feature views.

Each branch k: ImprovedEnc -> reconstruct feature k -> Pearson corr with each of 4 physical
speakers -> softmax. Fuse: OOF logistic stacker on concatenated log-posteriors (+ OOF alpha as a
simple baseline). 4-class attended speaker (chance 0.25). REGIME=trial|loso. SHUFFLE_EEG=1 null.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
_c6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_v6.py")
_s6 = importlib.util.spec_from_file_location("c6", _c6); V6 = importlib.util.module_from_spec(_s6); _s6.loader.exec_module(V6)
ImprovedEnc, _corr, zs = V6.ImprovedEnc, V6._corr, V6.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"; H = 64
REGIME = os.environ.get("REGIME", "trial")
SPACES = os.environ.get("SPACES", "spec,w2v,sem").split(",")
SHUF = os.environ.get("SHUFFLE_EEG", "0") == "1"


def load():
    cols = {"spec": lambda z: z["env"][:, :4].astype(np.float32),
            "w2v": lambda z: z["w2v"].astype(np.float32),
            "sem": lambda z: z["sem"].astype(np.float32)}
    E = []; CD = {k: [] for k in SPACES}; Y = []; TK = []; SB = []; rng0 = np.random.default_rng(12345)
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        cand = {k: zs(cols[k](z), -1) for k in SPACES}
        perm = np.stack([np.random.default_rng(s * 1000 + int(tk[i])).permutation(4) for i in range(N)])
        tgt = np.array([int(np.where(perm[i] == y[i])[0][0]) for i in range(N)], int)
        cand = {k: np.stack([v[i][perm[i]] for i in range(N)]) for k, v in cand.items()}
        if SHUF: eeg = eeg[rng0.permutation(N)]
        E.append(eeg); [CD[k].append(cand[k]) for k in SPACES]; Y.append(tgt); TK.append(tk); SB.append(np.full(N, s - 1))
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); CD = {k: np.concatenate([c[:, :, :, :T] for c in v]) for k, v in CD.items()}
    return E, CD, np.concatenate(Y), np.concatenate(TK), np.concatenate(SB)


class FeatNet(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.enc = ImprovedEnc(32, H, film=False); self.dec = nn.Conv1d(H, dim, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):
        return _corr(self.dec(self.enc(eeg)), cand) * self.scale.exp().clamp(max=100)


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train_feat(cand, E, y, ep=25, lr=1e-3, bs=48):
    m = FeatNet(cand.shape[2]).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    Et = torch.as_tensor(E, device=dev); Ct = torch.as_tensor(cand, device=dev); yt = torch.as_tensor(y, device=dev)
    for _ in range(ep):
        for b in _bt(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(m(Et[bi], Ct[bi]), yt[bi]).backward(); opt.step(); opt.zero_grad()
    return m


@torch.no_grad()
def post(m, cand, E, bs=48):
    m.eval(); P = []
    for i in range(0, len(E), bs):
        P.append(F.softmax(m(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(cand[i:i + bs], device=dev)), 1).cpu().numpy())
    return np.concatenate(P)


def evaluate(E, CD, Y, tr, te, ep):
    yte = Y[te]; oof = {k: np.zeros((len(tr), 4), np.float32) for k in SPACES}; tep = {}
    # OOF posteriors per feature on the train set
    for it, iv in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, Y[tr]):
        a, b = tr[it], tr[iv]
        for k in SPACES:
            oof[k][iv] = post(train_feat(CD[k][a], E[a], Y[a], ep), CD[k][b], E[b])
    # final per-feature models on full train -> test posteriors
    for k in SPACES:
        tep[k] = post(train_feat(CD[k][tr], E[tr], Y[tr], ep), CD[k][te], E[te])
    out = {k: (tep[k].argmax(1) == yte).mean() for k in SPACES}
    Xtr = np.concatenate([np.log(oof[k] + 1e-9) for k in SPACES], 1)
    Xte = np.concatenate([np.log(tep[k] + 1e-9) for k in SPACES], 1)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, Y[tr])
    out["fused"] = (lr.predict(Xte) == yte).mean()
    return out


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, CD, Y, TK, SB = load()
    dims = {k: CD[k].shape[2] for k in SPACES}
    print(f"REGIME={REGIME} SPACES={SPACES} dims={dims} SHUF={SHUF} n={len(Y)}", flush=True)
    rows = []
    if REGIME == "loso":
        for s in np.unique(SB):
            r = evaluate(E, CD, Y, np.where(SB != s)[0], np.where(SB == s)[0], 25); rows.append(r)
            print("  held-out S%2d  " % (int(s) + 1) + " ".join(f"{k}={r[k]:.3f}" for k in SPACES) + f"  FUSED={r['fused']:.3f}", flush=True)
    else:
        trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
        for fi, te_tr in enumerate(folds):
            te = np.where(np.isin(TK, te_tr))[0]; tr = np.where(~np.isin(TK, te_tr))[0]
            r = evaluate(E, CD, Y, tr, te, 25); rows.append(r)
            print(f"  fold{fi}  " + " ".join(f"{k}={r[k]:.3f}" for k in SPACES) + f"  FUSED={r['fused']:.3f}", flush=True)
    print(f"\n=== CONTENT-FUSION {REGIME} (chance 0.25) {'<-- EEG-SHUFFLE NULL' if SHUF else ''} ===")
    for k in SPACES + ["fused"]:
        v = np.array([r[k] for r in rows]); print(f"  {k:6s} = {v.mean():.3f} +- {v.std():.3f}")


if __name__ == "__main__":
    main()
