"""Multi-target (spec+w2v+sem) reconstruction matcher under TRIAL-DISJOINT cross-subject
splits -- the airtight version of the 0.41 transfer number (which was NOT trial-disjoint).

5-fold over TRIAL indices, pooled across all 16 subjects: held-out trials' audio is never
in training (audio is shared across subjects, so trial-disjoint == audio-disjoint). EEG-only,
4-talker PERMUTED, loudness-equalized. Raw-correlation recon matcher (no learnable stim
encoder => immune to the target-content confound). SHUFFLE_EEG=1 gives the null.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, zs = _nets.Enc, _nets.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SPACES = os.environ.get("SPACES", "spec,w2v,sem").split(",")
SHUF = os.environ.get("SHUFFLE_EEG", "0") == "1"


def load():
    cols = {"spec": ("env", lambda z: z["env"][:, :4].astype(np.float32)),
            "w2v": ("w2v", lambda z: z["w2v"].astype(np.float32)),
            "sem": ("sem", lambda z: z["sem"].astype(np.float32))}
    E = []; CD = {k: [] for k in SPACES}; Y = []; TK = []; rng0 = np.random.default_rng(12345)
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        cand = {k: zs(cols[k][1](z), -1) for k in SPACES}                # each (N,4,D,T)
        perm = np.stack([np.random.default_rng(s * 1000 + int(tk[i])).permutation(4) for i in range(N)])
        tgt = np.array([int(np.where(perm[i] == y[i])[0][0]) for i in range(N)], int)
        cand = {k: np.stack([v[i][perm[i]] for i in range(N)]) for k, v in cand.items()}
        if SHUF: eeg = eeg[rng0.permutation(N)]
        E.append(eeg); [CD[k].append(cand[k]) for k in SPACES]; Y.append(tgt); TK.append(tk)
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); CD = {k: np.concatenate([c[:, :, :, :T] for c in v]) for k, v in CD.items()}
    return E, CD, np.concatenate(Y), np.concatenate(TK)


def _corr(r, cand):
    B, S, Dk, T = cand.shape
    rf = r.reshape(B, 1, Dk * T); cf = cand.reshape(B, S, Dk * T)
    rf = (rf - rf.mean(-1, keepdim=True)) / (rf.std(-1, keepdim=True) + 1e-6)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + 1e-6)
    return (rf * cf).mean(-1)


class MultiRecon(nn.Module):
    def __init__(self, dims, h=64):
        super().__init__(); self.enc = Enc(32, h); self.spaces = list(dims)
        self.dec = nn.ModuleDict({k: nn.Conv1d(h, d, 1) for k, d in dims.items()})
        self.scale = nn.ParameterDict({k: nn.Parameter(torch.tensor(2.3)) for k in dims})
        self.mix = nn.Parameter(torch.zeros(len(dims)))

    def forward(self, eeg, cand):
        z = self.enc(eeg); per = {}; recon = {}
        for k in self.spaces:
            r = self.dec[k](z); recon[k] = r
            per[k] = _corr(r, cand[k]) * self.scale[k].exp().clamp(max=100)
        mw = torch.softmax(self.mix, 0)
        return sum(mw[i] * per[k] for i, k in enumerate(self.spaces)), recon


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train(model, E, cand, tg, epochs=25, lr=1e-3, bs=48):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    E = torch.as_tensor(E, device=dev); cand = {k: torch.as_tensor(v, device=dev) for k, v in cand.items()}; tg = torch.as_tensor(tg, device=dev)
    for _ in range(epochs):
        for b in _bt(len(tg), bs):
            bi = torch.as_tensor(b, device=dev); cb = {k: v[bi] for k, v in cand.items()}
            logits, recon = model(E[bi], cb); idx = torch.arange(len(bi), device=dev)
            rl = sum(F.mse_loss(recon[k], cb[k][idx, tg[bi]]) for k in model.spaces) / len(model.spaces)
            (F.cross_entropy(logits, tg[bi]) + rl).backward(); opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def evalm(model, E, cand, tg, bs=48):
    model.eval(); P = []
    for i in range(0, len(tg), bs):
        cb = {k: torch.as_tensor(v[i:i + bs], device=dev) for k, v in cand.items()}
        P.append(model(torch.as_tensor(E[i:i + bs], device=dev), cb)[0].argmax(1).cpu().numpy())
    return float((np.concatenate(P) == tg).mean())


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, CD, Y, TK = load(); dims = {k: CD[k].shape[2] for k in SPACES}
    print(f"SPACES={SPACES} SHUFFLE_EEG={SHUF} dims={dims} n={len(Y)} (TRIAL-DISJOINT cross-subject)", flush=True)
    trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5); accs = []
    for fi, te_tr in enumerate(folds):
        te = np.isin(TK, te_tr); trm = ~te
        m = train(MultiRecon(dims), E[trm], {k: CD[k][trm] for k in SPACES}, Y[trm], epochs=25)
        a = evalm(m, E[te], {k: CD[k][te] for k in SPACES}, Y[te]); accs.append(a)
        print(f"  fold{fi}: n_test={te.sum()} acc={a:.3f}", flush=True)
    print(f"\n=== multi-target TRIAL-DISJOINT cross-subject (chance 0.25) ===")
    print(f"  mean acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}   {'<-- EEG-SHUFFLE NULL' if SHUF else ''}")


if __name__ == "__main__":
    main()
