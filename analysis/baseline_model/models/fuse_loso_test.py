"""LOSO fusion: EEG-only content (VLAAI 28-band reconstruction) + posterior-alpha spatial
(shrinkage LDA, optionally gaze-residualized) -> late fusion. One held-out TEST subject per call;
train on the other 15. Reports content-only / spatial-only / fused (fixed-beta + best-beta), 5 s.

  python fuse_loso_test.py <test_subject> [ALL|POST]      GAZERESID env: 1 (strict) | 0
"""
import glob, os, sys, numpy as np, torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
SR = 64.0; W = 320; HOP = 160
POST = [13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAZERESID = os.environ.get("GAZERESID", "1") == "1"
TEST = int(sys.argv[1]); ALLSUB = list(range(1, 17))


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def bandpow(e, lo, hi):
    b, a = butter(4, [lo / (SR / 2), hi / (SR / 2)], "band")
    return np.log(np.mean(filtfilt(b, a, e, axis=-1) ** 2, -1) + 1e-12)


def prep(s):
    """Per-subject windowed arrays for content + spatial (5 s, 0.5 overlap)."""
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64); env = z["env"][:, :4].astype(np.float32)
    gaze = np.nan_to_num(z["gaze"][:, :, :2].astype(np.float64)); att = z["attended"].astype(int) - 1
    rng = np.random.default_rng(20260619 + s); perm = np.stack([rng.permutation(4) for _ in range(len(att))])
    T = eeg.shape[-1]; st = list(range(0, T - W + 1, HOP)) or [0]; tri = np.tile(np.arange(len(att)), len(st))
    eeg_w = np.concatenate([eeg[:, :, w:w + W] for w in st], 0)
    cand_phys = _zt(np.concatenate([env[:, :, :, w:w + W] for w in st], 0).astype(np.float32), -1)
    gz = np.concatenate([gaze[:, w:w + W, :] for w in st], 0)
    permw = np.concatenate([perm for _ in st], 0); attw = att[tri]
    tgt = cand_phys[np.arange(len(tri)), attw]
    yslot = np.array([np.flatnonzero(permw[i] == attw[i])[0] for i in range(len(tri))])
    cand = np.stack([cand_phys[i][permw[i]] for i in range(len(cand_phys))])
    return dict(eeg_z=_zt(eeg_w, 2), tgt=tgt, cand=cand, yslot=yslot, attphys=attw, permw=permw,
                gfeat=np.concatenate([gz.mean(1), gz.std(1)], 1),
                spfeat=np.concatenate([bandpow(eeg_w[:, POST], 8, 12), bandpow(eeg_w[:, POST], 13, 30)], 1))


def train_content(eeg_z, tgt, cand, yslot, epochs=60, seed=0, margin=0.3, mscale=12.0):
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(eeg_z); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train(); idx = rng.permutation(n)
        for i in range(0, n, 256):
            b = idx[i:i + 256]
            r = m(torch.from_numpy(eeg_z[b]).to(DEV))
            loss = B.neg_pearson_loss(r, torch.from_numpy(tgt[b]).to(DEV))
            if margin > 0:                                        # match-mismatch margin (helps LOSO)
                sc = B.mm_scores(r, torch.from_numpy(cand[b]).to(DEV))
                loss = loss + margin * F.cross_entropy(mscale * sc, torch.from_numpy(yslot[b]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def content_scores(models, eeg_z, cand, bs=512):
    rs = []
    for m in models:
        m.eval(); out = []
        for i in range(0, len(eeg_z), bs):
            out.append(m(torch.from_numpy(eeg_z[i:i + bs]).to(DEV)).cpu())
        r = torch.cat(out, 0); rs.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
    return B.mm_scores(torch.stack(rs).mean(0), torch.from_numpy(cand)).numpy()


def acc(sc, y):
    return float((sc.argmax(1) == y).mean())


D = {s: prep(s) for s in ALLSUB}
tr = [s for s in ALLSUB if s != TEST]
Xz = np.concatenate([D[s]["eeg_z"] for s in tr]); Tg = np.concatenate([D[s]["tgt"] for s in tr])
Cd = np.concatenate([D[s]["cand"] for s in tr]); Ys = np.concatenate([D[s]["yslot"] for s in tr])
NSEED = int(os.environ.get("NSEED", "5"))
models = [train_content(Xz, Tg, Cd, Ys, seed=sd) for sd in range(NSEED)]
te = D[TEST]
cs = content_scores(models, te["eeg_z"], te["cand"])
sp_tr = np.concatenate([D[s]["spfeat"] for s in tr]); g_tr = np.concatenate([D[s]["gfeat"] for s in tr])
a_tr = np.concatenate([D[s]["attphys"] for s in tr])
y = te["yslot"]; csn = cs - cs.mean(1, keepdims=True)
for resid in (True, False):                                       # strict (gaze-resid) and standard
    Xtr, Xte = sp_tr, te["spfeat"]
    if resid:
        rg = LinearRegression().fit(g_tr, sp_tr)
        Xtr = sp_tr - rg.predict(g_tr); Xte = te["spfeat"] - rg.predict(te["gfeat"])
    lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, a_tr)
    pp = np.zeros((len(Xte), 4), np.float32); pp[:, lda.classes_.astype(int)] = lda.predict_proba(Xte)
    sp = np.stack([pp[i][te["permw"][i]] for i in range(len(pp))])
    f_fixed = acc(csn + 1.5 * np.log(sp + 1e-6), y)
    f_best = max(acc(csn + b * np.log(sp + 1e-6), y) for b in [0, .5, 1, 1.5, 2, 3, 4])
    print(f"s{TEST}: gazeresid={int(resid)} content={acc(cs, y):.3f} spatial={acc(sp, y):.3f} "
          f"FUSED_b1.5={f_fixed:.3f} FUSED_best={f_best:.3f} n={len(y)}", flush=True)
