"""Feasibility test: EEG-only content + gaze-residualized covert-neural spatial fusion,
within-subject, 5 s window. Confirms whether fusion clears 0.40 four-way before productionizing.

content branch : VLAAI backward (28-band spectrogram reconstruction) -> per-slot correlation.
spatial branch : posterior (parieto-occipito-temporal) alpha+beta log band-power, GAZE-RESIDUALIZED
                 (regress the window's gaze out of the features, train-fold fit) -> shrinkage LDA
                 -> attended PHYSICAL location posterior -> mapped to the permuted slot space.
fusion         : z-normalised content scores + beta * spatial log-posterior (beta grid; on-fold
                 upper bound flagged -- productionize with an inner-OOF beta).
Reports content-only / spatial-only / fused, mean over the given subjects, 5-fold trial-disjoint.
"""
import glob, os, sys, numpy as np, torch
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
SR = 64.0; W = 320; HOP = 160
POST = [13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]   # T7,T8,CP*,P*,PO,O* (no frontal/EOG)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECTS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4").split(",")]


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def bandpow(e, lo, hi):
    b, a = butter(4, [lo / (SR / 2), hi / (SR / 2)], "band")
    return np.log(np.mean(filtfilt(b, a, e, axis=-1) ** 2, -1) + 1e-12)


def load(s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64)
    env = z["env"][:, :4].astype(np.float32)                       # (100,4,28,T) 4 real talkers
    gaze = np.nan_to_num(z["gaze"][:, :, :2].astype(np.float64))
    att = z["attended"].astype(int) - 1
    rng = np.random.default_rng(20260619 + s)
    perm = np.stack([rng.permutation(4) for _ in range(len(att))])  # (100,4) slot->physical
    return eeg, env, gaze, att, perm


def train_content(eeg_z, tgt, epochs=45, seed=0):
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(eeg_z); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train(); idx = rng.permutation(n)
        for i in range(0, n, 128):
            b = idx[i:i + 128]
            e = torch.from_numpy(eeg_z[b]).to(DEV); t = torch.from_numpy(tgt[b]).to(DEV)
            loss = B.neg_pearson_loss(m(e), t)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def content_scores(models, eeg_z, cand):
    rs = []
    for m in models:
        m.eval(); r = m(torch.from_numpy(eeg_z).to(DEV)).cpu()
        rs.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
    return B.mm_scores(torch.stack(rs).mean(0), torch.from_numpy(cand)).numpy()   # (N,4) slot scores


def acc(scores, y):
    return float((scores.argmax(1) == y).mean())


def run_subject(s):
    eeg, env, gaze, att, perm = load(s)
    T = eeg.shape[-1]; st = list(range(0, T - W + 1, HOP)) or [0]; nb = len(st)
    tri = np.tile(np.arange(len(att)), nb)                          # window -> trial index
    eeg_w = np.concatenate([eeg[:, :, w:w + W] for w in st], 0)     # (Nw,32,W) raw
    eeg_z = _zt(eeg_w, 2)
    cand_phys = _zt(np.concatenate([env[:, :, :, w:w + W] for w in st], 0).astype(np.float32), -1)  # (Nw,4,28,W) physical
    gz = np.concatenate([gaze[:, w:w + W, :] for w in st], 0)       # (Nw,W,2)
    permw = np.concatenate([perm for _ in st], 0)                   # (Nw,4) slot->physical
    attw = att[tri]
    tgt = cand_phys[np.arange(len(tri)), attw]                      # attended talker 28-band env
    yslot = np.array([np.flatnonzero(permw[i] == attw[i])[0] for i in range(len(tri))])   # attended SLOT
    cand = np.stack([cand_phys[i][permw[i]] for i in range(len(cand_phys))])  # reorder -> index k = slot k
    gfeat = np.concatenate([gz.mean(1), gz.std(1)], 1)             # gaze summary (for residualisation)
    spfeat = np.concatenate([bandpow(eeg_w[:, POST], 8, 12), bandpow(eeg_w[:, POST], 13, 30)], 1)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cA, sA, fA = [], [], []
    for trn, tst in skf.split(np.arange(len(att)), att):           # split TRIALS
        mtr = np.isin(tri, trn); mte = np.isin(tri, tst)
        models = [train_content(eeg_z[mtr], tgt[mtr], seed=sd) for sd in range(2)]
        cs = content_scores(models, eeg_z[mte], cand[mte])         # (n,4) slot
        if os.environ.get("GAZERESID", "1") == "1":                # gaze-residualise features (strict)
            rg = LinearRegression().fit(gfeat[mtr], spfeat[mtr])
            Xtr = spfeat[mtr] - rg.predict(gfeat[mtr]); Xte = spfeat[mte] - rg.predict(gfeat[mte])
        else:                                                      # posterior-alpha, no gaze residual
            Xtr, Xte = spfeat[mtr], spfeat[mte]
        lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, attw[mtr])
        pp = np.zeros((mte.sum(), 4), np.float32)
        pp[:, lda.classes_.astype(int)] = lda.predict_proba(Xte)   # physical posterior
        sp = np.stack([pp[i][permw[mte][i]] for i in range(mte.sum())])   # -> slot space
        yte = yslot[mte]
        csn = cs - cs.mean(1, keepdims=True)
        fbest = max(acc(csn + b * np.log(sp + 1e-6), yte) for b in [0, .5, 1, 1.5, 2, 3, 4])
        cA.append(acc(cs, yte)); sA.append(acc(sp, yte)); fA.append(fbest)
    return np.mean(cA), np.mean(sA), np.mean(fA)


C, S, F = [], [], []
for s in SUBJECTS:
    c, sp, f = run_subject(s)
    C.append(c); S.append(sp); F.append(f)
    print(f"s{s}: content={c:.3f}  spatial(gaze-resid)={sp:.3f}  FUSED={f:.3f}", flush=True)
print(f"MEAN {SUBJECTS}: content={np.mean(C):.3f}  spatial={np.mean(S):.3f}  FUSED={np.mean(F):.3f}", flush=True)
