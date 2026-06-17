"""content_audonly: the DECISIVE confound test -- predict the attended talker from the
CANDIDATE AUDIO ALONE, with NO EEG anywhere in the model.

If the EEG-shuffle null is really the target-content confound (attended talker's audio is
systematically self-identifiable, trial-invariantly), then a zero-EEG classifier should reach
the SAME accuracy as the null (~0.365 for w2v, ~0.246 for spec). If instead the null came from
"something in the EEG", this should sit at chance 0.25. Trial-disjoint inter-subject + LOSO, so
train/val share NO trials -> any above-chance here is pure audio self-identifiability, not leakage.

Permutation-equivariant scorer: a SHARED per-candidate encoder scores each of the 4 streams
independently (position can't matter), softmax over the 4. FEAT=w2v|spec|sem|env. REGIME=trial|loso.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_c6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_v6.py")
_s6 = importlib.util.spec_from_file_location("c6", _c6); V6 = importlib.util.module_from_spec(_s6); _s6.loader.exec_module(V6)
zs = V6.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
REGIME = os.environ.get("REGIME", "trial")
FEAT = os.environ.get("FEAT", "w2v")
SHUF_LBL = os.environ.get("SHUFFLE_LABEL", "0") == "1"   # break audio<->attended (leak control)
TIME_SHUF = os.environ.get("TIME_SHUFFLE", "0") == "1"   # destroy temporal dynamics (static-vs-dynamic)


def load():
    col = {"spec": lambda z: z["env"][:, :4].astype(np.float32),
           "w2v": lambda z: z["w2v"].astype(np.float32),
           "sem": lambda z: z["sem"].astype(np.float32),
           "env": lambda z: z["env"][:, :4].mean(2, keepdims=True).astype(np.float32)}[FEAT]
    C = []; Y = []; TK = []; SB = []
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        cand = zs(col(z), -1)
        if TIME_SHUF:                                          # permute time frames -> kill dynamics
            cand = cand[..., np.random.default_rng(7 * s).permutation(cand.shape[-1])]
        perm = np.stack([np.random.default_rng(s * 1000 + int(tk[i])).permutation(4) for i in range(N)])
        tgt = np.array([int(np.where(perm[i] == y[i])[0][0]) for i in range(N)], int)
        cand = np.stack([cand[i][perm[i]] for i in range(N)])
        if SHUF_LBL:                                           # random attended position (break audio<->label)
            tgt = np.random.default_rng(31 * s).integers(0, 4, N)
        C.append(cand); Y.append(tgt); TK.append(tk); SB.append(np.full(N, s - 1))
    T = min(c.shape[-1] for c in C)
    C = np.concatenate([c[:, :, :, :T] for c in C])
    return C, np.concatenate(Y), np.concatenate(TK), np.concatenate(SB)


class AudOnly(nn.Module):
    def __init__(self, dim, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(dim, h, 15, padding=7), nn.BatchNorm1d(h), nn.ELU(),
                                 nn.Conv1d(h, h, 15, padding=7), nn.BatchNorm1d(h), nn.ELU())
        self.score = nn.Linear(h, 1)

    def forward(self, cand):
        B, S, D, T = cand.shape
        z = self.enc(cand.reshape(B * S, D, T)).mean(-1)        # shared per-candidate encoder
        return self.score(z).reshape(B, S)                      # (B,4) logits, position-symmetric


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train(dim, C, y, ep=25, lr=1e-3, bs=48):
    m = AudOnly(dim).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    Ct = torch.as_tensor(C, device=dev); yt = torch.as_tensor(y, device=dev)
    for _ in range(ep):
        for b in _bt(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(m(Ct[bi]), yt[bi]).backward(); opt.step(); opt.zero_grad()
    return m


@torch.no_grad()
def acc(m, C, y, bs=48):
    m.eval(); P = []
    for i in range(0, len(y), bs):
        P.append(m(torch.as_tensor(C[i:i + bs], device=dev)).argmax(1).cpu().numpy())
    return float((np.concatenate(P) == y).mean())


def main():
    torch.manual_seed(0); np.random.seed(0)
    C, Y, TK, SB = load(); dim = C.shape[2]
    print(f"FEAT={FEAT} REGIME={REGIME} dim={dim} n={len(Y)}  (NO EEG; chance 0.25)", flush=True)
    accs = []
    if REGIME == "loso":
        for s in np.unique(SB):
            tr, te = np.where(SB != s)[0], np.where(SB == s)[0]
            a = acc(train(dim, C[tr], Y[tr]), C[te], Y[te]); accs.append(a)
            print(f"  held-out S{int(s)+1:2d} aud-only={a:.3f}", flush=True)
    else:
        trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
        for fi, te_tr in enumerate(folds):
            te = np.where(np.isin(TK, te_tr))[0]; tr = np.where(~np.isin(TK, te_tr))[0]
            a = acc(train(dim, C[tr], Y[tr]), C[te], Y[te]); accs.append(a)
            print(f"  fold{fi} aud-only={a:.3f}", flush=True)
    a = np.array(accs)
    print(f"\n=== AUDIO-ONLY attended-id (NO EEG) FEAT={FEAT} {REGIME} (chance 0.25) ===")
    print(f"  mean={a.mean():.3f} +- {a.std():.3f}")


if __name__ == "__main__":
    main()
