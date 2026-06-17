"""content_advmm: the ADVANCED multi-feature reconstruction match-mismatch model.

Generates SEMANTIC(GPT-2) + AUDITORY(HuBERT/w2v) + SPECTRAL(28-band) + ENVELOPE features FROM
the EEG (shared ImprovedEnc -> per-feature decoder), matches each to the audio-derived feature by
Pearson corr, learned softmax mixture. KEY difference vs content_cfuse (CE-only, which leaked the
target-content confound through w2v, null 0.315): an MSE RECONSTRUCTION ANCHOR forces every decoder
to reproduce the TRUE attended feature, so it can't collapse into a pure confound-exploiting template.
Tests (a) whether the recon anchor DE-CONFOUNDS w2v (null -> chance?) and (b) the real multi-feature
content number. REGIME=trial|loso, each WITH its EEG-shuffle null. 4-class attended speaker, chance 0.25.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
_c6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_v6.py")
_s6 = importlib.util.spec_from_file_location("c6", _c6); V6 = importlib.util.module_from_spec(_s6); _s6.loader.exec_module(V6)
ImprovedEnc, _corr, zs = V6.ImprovedEnc, V6._corr, V6.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"; H = 64
REGIME = os.environ.get("REGIME", "trial")
SPACES = os.environ.get("SPACES", "spec,w2v,sem,env").split(",")
SHUF = os.environ.get("SHUFFLE_EEG", "0") == "1"


def load():
    cols = {"spec": lambda z: z["env"][:, :4].astype(np.float32),
            "w2v": lambda z: z["w2v"].astype(np.float32),
            "sem": lambda z: z["sem"].astype(np.float32),
            "env": lambda z: z["env"][:, :4].mean(2, keepdims=True).astype(np.float32)}  # broadband envelope
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


class AdvMM(nn.Module):
    def __init__(self, dims):
        super().__init__(); self.enc = ImprovedEnc(32, H, film=False); self.spaces = list(dims)
        self.dec = nn.ModuleDict({k: nn.Conv1d(H, d, 1) for k, d in dims.items()})
        self.scale = nn.ParameterDict({k: nn.Parameter(torch.tensor(2.3)) for k in dims})
        self.mix = nn.Parameter(torch.zeros(len(dims)))

    def forward(self, eeg, cand):
        z = self.enc(eeg); per = {}; recon = {}
        for k in self.spaces:
            r = self.dec[k](z); recon[k] = r
            per[k] = _corr(r, cand[k]) * self.scale[k].exp().clamp(max=100)
        mw = torch.softmax(self.mix, 0)
        return sum(mw[i] * per[k] for i, k in enumerate(self.spaces)), per, recon


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train(dims, E, cand, tg, epochs=25, lr=1e-3, bs=48):
    m = AdvMM(dims).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    Et = torch.as_tensor(E, device=dev); cb_all = {k: torch.as_tensor(v, device=dev) for k, v in cand.items()}; tgt = torch.as_tensor(tg, device=dev)
    for _ in range(epochs):
        for b in _bt(len(tg), bs):
            bi = torch.as_tensor(b, device=dev); cb = {k: v[bi] for k, v in cb_all.items()}
            logits, _, recon = m(Et[bi], cb); idx = torch.arange(len(bi), device=dev)
            rl = sum(F.mse_loss(recon[k], cb[k][idx, tgt[bi]]) for k in m.spaces) / len(m.spaces)
            (F.cross_entropy(logits, tgt[bi]) + rl).backward(); opt.step(); opt.zero_grad()
    return m


@torch.no_grad()
def evalm(m, E, cand, tg, bs=48):
    m.eval(); fz = []; pk = {k: [] for k in m.spaces}
    for i in range(0, len(tg), bs):
        cb = {k: torch.as_tensor(v[i:i + bs], device=dev) for k, v in cand.items()}
        f, per, _ = m(torch.as_tensor(E[i:i + bs], device=dev), cb)
        fz.append(f.argmax(1).cpu().numpy())
        for k in m.spaces: pk[k].append(per[k].argmax(1).cpu().numpy())
    fz = np.concatenate(fz)
    return {"fused": float((fz == tg).mean()), **{k: float((np.concatenate(pk[k]) == tg).mean()) for k in m.spaces}}


def run(E, CD, Y, tr, te, ep):
    dims = {k: CD[k].shape[2] for k in SPACES}
    m = train(dims, E[tr], {k: CD[k][tr] for k in SPACES}, Y[tr], ep)
    return evalm(m, E[te], {k: CD[k][te] for k in SPACES}, Y[te])


def main():
    sd = int(os.environ.get("SEED", "0")); torch.manual_seed(sd); np.random.seed(sd)
    E, CD, Y, TK, SB = load()
    dims = {k: CD[k].shape[2] for k in SPACES}
    print(f"REGIME={REGIME} SPACES={SPACES} dims={dims} SHUF={SHUF} n={len(Y)}", flush=True)
    rows = []
    if REGIME == "loso":
        for s in np.unique(SB):
            r = run(E, CD, Y, np.where(SB != s)[0], np.where(SB == s)[0], 25); rows.append(r)
            print("  held-out S%2d  " % (int(s) + 1) + " ".join(f"{k}={r[k]:.3f}" for k in SPACES) + f"  FUSED={r['fused']:.3f}", flush=True)
    else:
        trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
        for fi, te_tr in enumerate(folds):
            te = np.where(np.isin(TK, te_tr))[0]; tr = np.where(~np.isin(TK, te_tr))[0]
            r = run(E, CD, Y, tr, te, 25); rows.append(r)
            print(f"  fold{fi}  " + " ".join(f"{k}={r[k]:.3f}" for k in SPACES) + f"  FUSED={r['fused']:.3f}", flush=True)
    print(f"\n=== ADV multi-feature recon-MM {REGIME} (chance 0.25) {'<-- EEG-SHUFFLE NULL' if SHUF else ''} ===")
    for k in SPACES + ["fused"]:
        v = np.array([r[k] for r in rows]); print(f"  {k:6s} = {v.mean():.3f} +- {v.std():.3f}")


if __name__ == "__main__":
    main()
