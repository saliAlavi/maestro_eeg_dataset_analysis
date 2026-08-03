"""Gaze -> attended DIRECTION (4-way physical talker location, chance 0.25).

Quantifies overt orienting: how well the listener's gaze predicts which of the four fixed
loudspeaker locations they attend. Network = small MLP on rich per-window gaze statistics;
shrinkage-LDA reported as a robust reference. Within-subject (5-fold trial-disjoint) and LOSO.
Gaze is subject-relative/uncalibrated, so LOSO is expected to be much weaker than within.

  python gaze_dir.py                      # all 16 subjects, 5 s + whole-trial, within + loso
"""
import glob, numpy as np, torch, torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
SR = 64.0; DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUBS = list(range(1, 17))


def feats_window(g):                                   # g (N,W,3) -> (N,F) rich gaze stats
    d = np.diff(g, axis=1)
    parts = [g.mean(1), g.std(1), np.nanpercentile(g, 10, 1), np.nanpercentile(g, 50, 1),
             np.nanpercentile(g, 90, 1), np.abs(d).mean(1), d.std(1)]
    return np.nan_to_num(np.concatenate(parts, 1)).astype(np.float32)


def load(s, win_s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    gaze = z["gaze"].astype(np.float64); att = z["attended"].astype(int) - 1
    T = gaze.shape[1]; W = min(int(round(win_s * SR)), T); hop = max(1, W // 2)
    st = list(range(0, T - W + 1, hop)) or [0]
    G = np.concatenate([gaze[:, w:w + W, :] for w in st], 0)
    y = np.tile(att, len(st)); tri = np.tile(np.arange(len(att)), len(st))
    return feats_window(G), y, tri, att


class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h // 2, 4))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, ep=120, seed=0):
    torch.manual_seed(seed)
    m = MLP(X.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-3)
    Xt = torch.from_numpy(X).to(DEV); yt = torch.from_numpy(y).long().to(DEV)
    for _ in range(ep):
        m.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(m(Xt), yt); loss.backward(); opt.step()
    return m


@torch.no_grad()
def pred_mlp(m, X):
    m.eval(); return m(torch.from_numpy(X).to(DEV)).argmax(1).cpu().numpy()


def within(win_s):
    mlp, lda = [], []
    for s in SUBS:
        X, y, tri, att = load(s, win_s); sc = StandardScaler()
        skf = StratifiedKFold(5, shuffle=True, random_state=42); am, al = [], []
        for trn, tst in skf.split(np.arange(len(att)), att):
            mtr = np.isin(tri, trn); mte = np.isin(tri, tst)
            Xtr = sc.fit_transform(X[mtr]).astype(np.float32); Xte = sc.transform(X[mte]).astype(np.float32)
            am.append((pred_mlp(train_mlp(Xtr, y[mtr]), Xte) == y[mte]).mean())
            al.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, y[mtr]).predict(Xte) == y[mte]).mean())
        mlp.append(np.mean(am)); lda.append(np.mean(al))
    return np.mean(mlp), np.mean(lda)


def loso(win_s):
    data = {s: load(s, win_s) for s in SUBS}; mlp, lda = [], []
    for ts in SUBS:
        tr = [s for s in SUBS if s != ts]; sc = StandardScaler()
        Xtr = sc.fit_transform(np.concatenate([data[s][0] for s in tr])).astype(np.float32)
        ytr = np.concatenate([data[s][1] for s in tr])
        Xte = sc.transform(data[ts][0]).astype(np.float32); yte = data[ts][1]
        mlp.append((pred_mlp(train_mlp(Xtr, ytr), Xte) == yte).mean())
        lda.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte) == yte).mean())
    return np.mean(mlp), np.mean(lda)


for win_s, lab in [(5.0, "5s"), (29.8, "whole")]:
    wm, wl = within(win_s); lm, ll = loso(win_s)
    print(f"GAZE->direction [{lab:5s}] WITHIN mlp={wm:.3f} lda={wl:.3f} | LOSO mlp={lm:.3f} lda={ll:.3f}  (chance .25)", flush=True)
