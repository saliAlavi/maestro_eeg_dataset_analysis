"""Head IMU (accelerometer + gyroscope) -> attended DIRECTION vs CONTENT over a decision-window
curve, the head-orienting characterization branch (like the video branch; NOT in the EEG headline).

The IMU is already in the aligned pa2_af64 cache (z["imu"], 6 ch = accel xyz + gyro xyz @64 Hz):
accel senses head pitch/roll via gravity; gyro senses head turning. Per window we summarise the
6 channels (mean/std/percentiles + net gyro rotation) and decode:
  DIRECTION : attended PHYSICAL loudspeaker (4-way, chance .25) -- does head pose/motion reveal
              WHERE the listener attends (overt head orienting)?
  CONTENT   : attended permuted SLOT (4-way, chance .25) -- must stay ~chance (head motion carries
              no talker content).
Within-subject (5-fold trial-disjoint) and LOSO, LDA + MLP. Writes results/imu_curve/curve.json.

  python imu_curve.py
"""
import glob, json, os, numpy as np, torch, torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
RUN_ROOT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; SUBS = list(range(1, 17))
SR = 64.0; WINS = [5, 10, 15, 20, 30]


def _feat(win):
    """win (n, W, 6) -> (n, F) head-pose/motion summary."""
    m = win.mean(1); sd = win.std(1)
    p10 = np.percentile(win, 10, 1); p90 = np.percentile(win, 90, 1)
    gyro_rot = (np.cumsum(win[:, :, 3:], 1) / SR)                       # integrate gyro -> relative angle
    rot_range = gyro_rot.max(1) - gyro_rot.min(1)                       # net head-turn excursion (3)
    return np.concatenate([m, sd, p10, p90, rot_range], 1).astype(np.float32)


def load_windows(s, w_s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    imu = np.nan_to_num(z["imu"].astype(np.float32))                   # (100,T,6)
    att = z["attended"].astype(int) - 1
    rng = np.random.default_rng(20260619 + s); perm = np.stack([rng.permutation(4) for _ in range(len(att))])
    yslot = np.array([np.flatnonzero(perm[i] == att[i])[0] for i in range(len(att))])
    T = imu.shape[1]; W = min(int(round(w_s * SR)), T); st = list(range(0, T - W + 1, max(1, W // 2))) or [0]
    X, ydir, ycon, tri = [], [], [], []
    for j in range(len(att)):
        for a in st:
            X.append(imu[j, a:a + W]); ydir.append(att[j]); ycon.append(yslot[j]); tri.append(j)
    return _feat(np.stack(X)), np.array(ydir), np.array(ycon), np.array(tri)


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


def scale(Xtr, Xte):
    sc = StandardScaler().fit(Xtr); return sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)


def within(data, target):
    lm, ll = [], []
    for s in SUBS:
        X, ydir, ycon, tri = data[s]; y = ydir if target == "dir" else ycon
        ntr = tri.max() + 1; strat = np.array([ydir[np.flatnonzero(tri == t)[0]] for t in range(ntr)])
        am, al = [], []
        for trn, tst in StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(ntr), strat):
            mtr, mte = np.isin(tri, trn), np.isin(tri, tst)
            Xtr, Xte = scale(X[mtr], X[mte])
            am.append((pred(train_mlp(Xtr, y[mtr]), Xte) == y[mte]).mean())
            al.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, y[mtr]).predict(Xte) == y[mte]).mean())
        lm.append(np.mean(am)); ll.append(np.mean(al))
    return float(np.mean(lm)), float(np.mean(ll))


def loso(data, target):
    lm, ll = [], []
    for ts in SUBS:
        tr = [s for s in SUBS if s != ts]; idx = 1 if target == "dir" else 2
        Xtr = np.concatenate([data[s][0] for s in tr]); Xte = data[ts][0]
        ytr = np.concatenate([data[s][idx] for s in tr]); yte = data[ts][idx]
        Xtr, Xte = scale(Xtr, Xte)
        lm.append((pred(train_mlp(Xtr, ytr), Xte) == yte).mean())
        ll.append((LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte) == yte).mean())
    return float(np.mean(lm)), float(np.mean(ll))


rows = []
for w in WINS:
    data = {s: load_windows(s, w) for s in SUBS}
    npt = int(np.mean([len(data[s][0]) for s in SUBS]))
    dwm, dwl = within(data, "dir"); dlm, dll = loso(data, "dir")
    cwm, cwl = within(data, "content"); clm, cll = loso(data, "content")
    r = dict(win_s=w, dir_within_mlp=dwm, dir_within_lda=dwl, dir_loso_mlp=dlm, dir_loso_lda=dll,
             content_within_lda=cwl, content_loso_lda=cll, n_win_per_subj=npt)
    rows.append(r)
    print(f"[imu|w{w}] DIR within(mlp/lda)={dwm:.3f}/{dwl:.3f} loso={dlm:.3f}/{dll:.3f} | "
          f"CONTENT within={cwl:.3f} loso={cll:.3f} (chance .25, n~{npt}/subj)", flush=True)

os.makedirs(f"{RUN_ROOT}/results/imu_curve", exist_ok=True)
json.dump(rows, open(f"{RUN_ROOT}/results/imu_curve/curve.json", "w"), indent=2, default=float)
print("wrote", f"{RUN_ROOT}/results/imu_curve/curve.json", flush=True)
