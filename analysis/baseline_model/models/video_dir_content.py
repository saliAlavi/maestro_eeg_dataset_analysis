"""Video (frozen V-JEPA2 scene+fovea) -> DIRECTION vs CONTENT, to ask whether vision carries
attention information beyond overt orienting.

  DIRECTION : predict the attended PHYSICAL loudspeaker location (4-way, chance .25). High => the
              egocentric scene/foveation reveals WHERE the listener attends (orienting).
  CONTENT   : predict the attended talker's permuted SLOT (4-way, chance .25). The per-trial
              candidate permutation decouples slot from physical location, so location is useless;
              above chance would mean the video carries talker CONTENT (it should not -- no faces/
              lips, static scene). Chance here => vision is orienting, not content.

Frozen 2048-d embedding -> StandardScaler -> PCA -> {shrinkage-LDA reference, small MLP network}.
Within-subject (5-fold trial-disjoint) and LOSO.  python video_dir_content.py
"""
import numpy as np, torch, torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

VJP = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa"
CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; SUBS = list(range(1, 17))
import glob


def load(s):
    zv = np.load(f"{VJP}/s{s}.npz")
    Vall = np.concatenate([zv["scene"], zv["fovea"]], 1).astype(np.float32)   # (nv,2048)
    vmap = {int(k): i for i, k in enumerate(zv["trial_k"].astype(int))}
    za = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    att_all = za["attended"].astype(int) - 1; atk = za["trial_k"].astype(int)
    keep = [(j, vmap[int(k)]) for j, k in enumerate(atk) if int(k) in vmap]   # align by trial_k
    ai = [j for j, _ in keep]; vi = [v for _, v in keep]
    V = Vall[vi]; att = att_all[ai]
    rng = np.random.default_rng(20260619 + s)
    perm = np.stack([rng.permutation(4) for _ in range(len(att))])           # slot->physical
    yslot = np.array([np.flatnonzero(perm[i] == att[i])[0] for i in range(len(att))])
    return V, att, yslot                                                     # direction=att, content=yslot


class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__(); self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.4), nn.Linear(h, 4))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, ep=150, seed=0):
    torch.manual_seed(seed)
    m = MLP(X.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=3e-3)
    Xt = torch.from_numpy(X).to(DEV); yt = torch.from_numpy(y).long().to(DEV)
    for _ in range(ep):
        m.train(); opt.zero_grad(); nn.functional.cross_entropy(m(Xt), yt).backward(); opt.step()
    return m


@torch.no_grad()
def pred(m, X):
    m.eval(); return m(torch.from_numpy(X).to(DEV)).argmax(1).cpu().numpy()


def reduce(Vtr, Vte, k=24):
    sc = StandardScaler().fit(Vtr); a, b = sc.transform(Vtr), sc.transform(Vte)
    kk = max(2, min(k, a.shape[0] - 1))
    p = PCA(kk).fit(a); return p.transform(a).astype(np.float32), p.transform(b).astype(np.float32)


def within(target):                                    # target: 'dir' or 'content'
    ml, ld = [], []
    for s in SUBS:
        V, att, yslot = load(s); y = att if target == "dir" else yslot
        skf = StratifiedKFold(5, shuffle=True, random_state=42); am, al = [], []
        for trn, tst in skf.split(V, att):             # split by trial (stratify by location)
            Xtr, Xte = reduce(V[trn], V[tst])
            am.append((pred(train_mlp(Xtr, y[trn]), Xte) == y[tst]).mean())
            al.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, y[trn]).predict(Xte) == y[tst]).mean())
        ml.append(np.mean(am)); ld.append(np.mean(al))
    return np.mean(ml), np.mean(ld)


def loso(target):
    D = {s: load(s) for s in SUBS}; ml, ld = [], []
    for ts in SUBS:
        tr = [s for s in SUBS if s != ts]
        Vtr = np.concatenate([D[s][0] for s in tr]); Vte = D[ts][0]
        ytr = np.concatenate([D[s][1 if target == "dir" else 2] for s in tr])
        yte = D[ts][1 if target == "dir" else 2]
        Xtr, Xte = reduce(Vtr, Vte)
        ml.append((pred(train_mlp(Xtr, ytr), Xte) == yte).mean())
        ld.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte) == yte).mean())
    return np.mean(ml), np.mean(ld)


for target, lab in [("dir", "DIRECTION"), ("content", "CONTENT  ")]:
    wm, wl = within(target); lm, ll = loso(target)
    print(f"VIDEO->{lab} WITHIN mlp={wm:.3f} lda={wl:.3f} | LOSO mlp={lm:.3f} lda={ll:.3f}  (chance .25)", flush=True)
