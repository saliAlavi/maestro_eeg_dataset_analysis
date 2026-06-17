"""content_cca: LEAK-PROOF CCA matcher for content AAD.

The earlier CCA hit 0.97 by memorizing the SHARED audio (same FLACs across subjects +
deterministic attended schedule) under cross-subject pooling. This version blocks that:

  (1) TRIAL-DISJOINT splits: 5-fold over TRIAL indices, pooled across all 16 subjects.
      Test trials' audio is NEVER in training -> a candidate-classifier has nothing to
      memorize.  (vs the old leave-SUBJECT-out, where test audio was seen.)
  (2) EEG-SHUFFLE null built in (SHUFFLE_EEG=1): breaks EEG<->(cand,label); must be chance.
  (3) Both a learnable CCA matcher (eeg_proj + stim_enc, the real canonical-correlation
      idea) AND a reconstruction baseline run under the SAME protocol, so we see whether
      CCA adds over the backward model -- with the shuffle null proving it's EEG-driven.

4-talker PERMUTED (position removed), table-power normalized (loudness removed), EEG-only.
MODEL=cca|recon. Chance 0.25 (recon-matcher within-subject baseline 0.41).
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, zs = _nets.Enc, _nets.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "cca")
SHUF = os.environ.get("SHUFFLE_EEG", "0") == "1"
WITHIN = os.environ.get("WITHIN", "0") == "1"


def load():
    """Pool (subject,trial) samples; keep trial_k for trial-disjoint splitting."""
    E, C, Y, TK, SB = [], [], [], [], []
    rng0 = np.random.default_rng(12345)
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0])
        eeg = zs(z["eeg"].astype(np.float32), 2)
        spec = zs(z["env"][:, :4].astype(np.float32), -1)            # (N,4,28,T)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int)
        # per-trial permutation (position removed); deterministic by (subject,trial)
        N = len(y); cand = np.empty_like(spec); tgt = np.empty(N, int)
        for i in range(N):
            p = np.random.default_rng(s * 1000 + int(tk[i])).permutation(4)
            cand[i] = spec[i][p]; tgt[i] = int(np.where(p == y[i])[0][0])
        if SHUF: eeg = eeg[rng0.permutation(N)]                       # EEG-shuffle null
        E.append(eeg); C.append(cand); Y.append(tgt); TK.append(tk); SB.append(np.full(N, s))
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); C = np.concatenate([c[:, :, :, :T] for c in C])
    return E, C, np.concatenate(Y), np.concatenate(TK), np.concatenate(SB)


def _corr(r, cand):
    B, S, Dk, T = cand.shape
    rf = r.reshape(B, 1, Dk * T); cf = cand.reshape(B, S, Dk * T)
    rf = (rf - rf.mean(-1, keepdim=True)) / (rf.std(-1, keepdim=True) + 1e-6)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + 1e-6)
    return (rf * cf).mean(-1)


class CCANet(nn.Module):
    def __init__(self, nb=28, h=64, d=32):
        super().__init__(); self.enc = Enc(32, h)
        self.eeg_proj = nn.Conv1d(h, d, 1)
        self.stim_enc = nn.Sequential(nn.Conv1d(nb, d, 5, padding=2), nn.ELU(), nn.Conv1d(d, d, 1))
        self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):
        ze = self.eeg_proj(self.enc(eeg)); B, S, nb, T = cand.shape
        zc = self.stim_enc(cand.reshape(B * S, nb, T)).reshape(B, S, -1, T)
        return _corr(ze, zc) * self.scale.exp().clamp(max=100)


class ReconNet(nn.Module):
    def __init__(self, nb=28, h=64):
        super().__init__(); self.enc = Enc(32, h); self.dec = nn.Conv1d(h, nb, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):
        return _corr(self.dec(self.enc(eeg)), cand) * self.scale.exp().clamp(max=100)


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train(model, E, C, tg, epochs=25, lr=1e-3, bs=48):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    E = torch.as_tensor(E, device=dev); C = torch.as_tensor(C, device=dev); tg = torch.as_tensor(tg, device=dev)
    for _ in range(epochs):
        for b in _bt(len(tg), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(model(E[bi], C[bi]), tg[bi]).backward(); opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def evalm(model, E, C, tg, bs=48):
    model.eval(); P = []
    for i in range(0, len(tg), bs):
        P.append(model(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(C[i:i + bs], device=dev)).argmax(1).cpu().numpy())
    return float((np.concatenate(P) == tg).mean())


@torch.no_grad()
def _pred(model, E, C, bs=48):
    model.eval(); P=[]
    for i in range(0,len(E),bs): P.append(model(torch.as_tensor(E[i:i+bs],device=dev),torch.as_tensor(C[i:i+bs],device=dev)).argmax(1).cpu().numpy())
    return np.concatenate(P)


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, Y, TK, SB = load()
    nb = C.shape[2]; net = (lambda: CCANet(nb)) if MODEL == "cca" else (lambda: ReconNet(nb))
    print(f"MODEL={MODEL} SHUFFLE_EEG={SHUF}  pooled samples={len(Y)}  (trial-disjoint 5-fold)", flush=True)
    from sklearn.model_selection import StratifiedKFold
    accs = []
    if WITHIN:                       # per-subject 5-fold (pure within-subject, trial-disjoint)
        for s in np.unique(SB):
            m_ = SB == s; Es, Cs, Ys = E[m_], C[m_], Y[m_]; pr = np.zeros(len(Ys), int)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Es, Ys):
                mdl = train(net(), Es[tr], Cs[tr], Ys[tr], epochs=40)
                pr[te] = np.concatenate([mdl(torch.as_tensor(Es[te][i:i+48],device=dev),torch.as_tensor(Cs[te][i:i+48],device=dev)).argmax(1).cpu().numpy() for i in range(0,len(te),48)]) if False else _pred(mdl,Es[te],Cs[te])
            accs.append((pr == Ys).mean()); print(f"  S{int(s):2d} acc={(pr==Ys).mean():.3f}", flush=True)
        print(f"\n=== {MODEL} WITHIN-SUBJECT 5-fold (chance 0.25) ===")
        print(f"  mean acc = {np.mean(accs):.3f} +- {np.std(accs):.3f}   {'<-- EEG-SHUFFLE NULL' if SHUF else ''}"); return
    trials = np.unique(TK)
    folds = np.array_split(np.random.permutation(trials), 5)
    for fi, te_tr in enumerate(folds):
        te = np.isin(TK, te_tr); trm = ~te
        m = train(net(), E[trm], C[trm], Y[trm], epochs=25)
        a = evalm(m, E[te], C[te], Y[te]); accs.append(a)
        print(f"  fold{fi}: test_trials={len(te_tr)} n_test={te.sum()} acc={a:.3f}", flush=True)
    print(f"\n=== {MODEL} TRIAL-DISJOINT cross-subject (chance 0.25; recon within=0.41) ===")
    print(f"  mean acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}   {'<-- EEG-SHUFFLE NULL' if SHUF else ''}")


if __name__ == "__main__":
    main()
