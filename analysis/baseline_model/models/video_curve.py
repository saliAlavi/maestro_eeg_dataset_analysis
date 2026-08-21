"""Video (frozen V-JEPA2 scene+fovea) -> attended DIRECTION over a decision-window curve, the
overt-orienting characterization branch (NOT part of the EEG-only fusion headline).

Builds longer-window embeddings from the 5 s windowed cache by mean-pooling the non-overlapping
5 s sub-windows (win_start in {0,5,...,25}); g = w_s/5 consecutive chunks -> one pooled embedding.
Reports DIRECTION (attended physical location, 4-way chance .25) and a CONTENT control (attended
permuted slot -> should stay ~chance: vision is orienting, not talker content), within-subject
(5-fold trial-disjoint) and LOSO, LDA + MLP. Writes results/video_curve.json.

  python video_curve.py
"""
import glob, json, os, numpy as np, torch, torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

VJP = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa"
CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
RUN_ROOT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; SUBS = list(range(1, 17))
WINS = [5, 10, 15, 20, 30]


def _labels(s):
    za = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    att = za["attended"].astype(int) - 1; atk = za["trial_k"].astype(int)
    rng = np.random.default_rng(20260619 + s); perm = np.stack([rng.permutation(4) for _ in range(len(att))])
    yslot = np.array([np.flatnonzero(perm[i] == att[i])[0] for i in range(len(att))])
    return {int(k): (int(att[i]), int(yslot[i])) for i, k in enumerate(atk)}


def load_pooled(s, w_s):
    """Pool the non-overlapping 5 s sub-windows into w_s windows -> (V, ydir, ycon, tri)."""
    lab = _labels(s)
    z = np.load(f"{VJP}/s{s}_win5.npz")
    tk = z["trial_k"].astype(int); ws = z["win_start"].astype(float)
    V = np.concatenate([z["scene"], z["fovea"]], 1).astype(np.float32)
    keep = np.array([abs(w / 5 - round(w / 5)) < 0.1 for w in ws]) & np.isin(tk, list(lab))   # win_start multiples of 5
    V, tk, ws = V[keep], tk[keep], ws[keep]
    g = max(1, int(round(w_s / 5)))
    rows_V, rows_dir, rows_con, rows_tri = [], [], [], []
    uk = {k: j for j, k in enumerate(sorted(set(int(k) for k in tk)))}
    for k in sorted(set(int(k) for k in tk)):
        m = tk == k; order = np.argsort(ws[m]); vk = V[m][order]              # this trial's chunks, time-ordered
        for j in range(0, len(vk) - g + 1, max(1, g)):                        # non-overlapping groups of g
            rows_V.append(vk[j:j + g].mean(0)); rows_tri.append(uk[k])
            rows_dir.append(lab[k][0]); rows_con.append(lab[k][1])
    return np.array(rows_V, np.float32), np.array(rows_dir), np.array(rows_con), np.array(rows_tri)


class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.4), nn.Linear(h, 4))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, ep=150, seed=0):
    torch.manual_seed(seed); m = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=3e-3)
    Xt = torch.from_numpy(X).to(DEV); yt = torch.from_numpy(y).long().to(DEV)
    for _ in range(ep):
        m.train(); opt.zero_grad(); nn.functional.cross_entropy(m(Xt), yt).backward(); opt.step()
    return m


@torch.no_grad()
def pred(m, X):
    m.eval(); return m(torch.from_numpy(X).to(DEV)).argmax(1).cpu().numpy()


def reduce(Vtr, Vte, k=24):
    sc = StandardScaler().fit(Vtr); a, b = sc.transform(Vtr), sc.transform(Vte)
    p = PCA(max(2, min(k, a.shape[0] - 1))).fit(a)
    return p.transform(a).astype(np.float32), p.transform(b).astype(np.float32)


def within(data, target):
    ld = []
    for s in SUBS:
        V, ydir, ycon, tri = data[s]; y = ydir if target == "dir" else ycon
        ntr = tri.max() + 1; strat = np.array([ydir[np.flatnonzero(tri == t)[0]] for t in range(ntr)])
        al = []
        for trn, tst in StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(ntr), strat):
            mtr, mte = np.isin(tri, trn), np.isin(tri, tst)
            Xtr, Xte = reduce(V[mtr], V[mte])
            al.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, y[mtr]).predict(Xte) == y[mte]).mean())
        ld.append(np.mean(al))
    return float(np.mean(ld))


def loso(data, target):
    ld = []
    for ts in SUBS:
        tr = [s for s in SUBS if s != ts]; idx = 1 if target == "dir" else 2
        Vtr = np.concatenate([data[s][0] for s in tr]); Vte = data[ts][0]
        ytr = np.concatenate([data[s][idx] for s in tr]); yte = data[ts][idx]
        Xtr, Xte = reduce(Vtr, Vte)
        ld.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte) == yte).mean())
    return float(np.mean(ld))


rows = []
for w in WINS:
    data = {s: load_pooled(s, w) for s in SUBS}
    npt = int(np.mean([len(data[s][0]) for s in SUBS]))
    r = dict(win_s=w, dir_within=within(data, "dir"), dir_loso=loso(data, "dir"),
             content_within=within(data, "content"), content_loso=loso(data, "content"), n_win_per_subj=npt)
    rows.append(r)
    print(f"[video|w{w}] DIR within={r['dir_within']:.3f} loso={r['dir_loso']:.3f} | "
          f"CONTENT within={r['content_within']:.3f} loso={r['content_loso']:.3f} (chance .25, n~{npt}/subj)", flush=True)

os.makedirs(f"{RUN_ROOT}/results/video_curve", exist_ok=True)
json.dump(rows, open(f"{RUN_ROOT}/results/video_curve/curve.json", "w"), indent=2, default=float)
print("wrote", f"{RUN_ROOT}/results/video_curve/curve.json", flush=True)
