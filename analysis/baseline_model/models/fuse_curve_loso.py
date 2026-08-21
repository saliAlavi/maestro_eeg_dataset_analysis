"""LOSO fusion x decision-window curve (companion to fuse_curve.py's within path).

One held-out TEST subject per call; train on the other 15.
  content branch : VLAAI 28-band backward decoder, trained @5s on 15 subjects (5-seed recon
                   ensemble, match-mismatch margin=0.3 -- helps LOSO), early-stopped on a
                   CONTENT-DISJOINT inner-val (a held-out set of trial_k, shared across the
                   training subjects). Fully-convolutional -> scores at any window.
  spatial branch : posterior alpha+beta log band-power, GAZE-RESIDUALISED covert-direction LDA
                   (trained on the 15 subjects @ the eval window), -> permuted slot space.
  fusion         : content + b * spatial log-posterior. Per-window we save the (mean-centred
                   content, spatial log-posterior, label) arrays so the aggregator can tune b
                   ADMISSIBLY across subjects (b for subject S chosen on the OTHER 15). A fixed
                   b=1.5 fused accuracy is also written.

Candidates are the four real talkers -> chance 0.25 at every window. Writes one JSON per subject
to results/curve_fusewin_loso/.
"""
import glob, os, sys, json, numpy as np, torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
RUN_ROOT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad"
SR = 64.0; TRAIN_W = 320
POST = [13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = int(sys.argv[1]); ALLSUB = list(range(1, 17))
WINS = [float(x) for x in os.environ.get("WINS", "5,10,15,20,30").split(",")]
NSEED = int(os.environ.get("NSEED", "5"))
MARGIN = float(os.environ.get("MARGIN", "0.3"))
GAZERESID = os.environ.get("GAZERESID", "1") == "1"
TAG = os.environ.get("TAG", "curve_fusewin_loso")


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def bandpow(e, lo, hi):
    b, a = butter(4, [lo / (SR / 2), hi / (SR / 2)], "band")
    return np.log(np.mean(filtfilt(b, a, e, axis=-1) ** 2, -1) + 1e-12)


def acc(sc, y):
    return float((sc.argmax(1) == y).mean())


def _wins(T, W):
    return list(range(0, T - W + 1, max(1, W // 2))) or [0]


def prep(s, W):
    """Windowed arrays for subject s at window W samples."""
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64); env = z["env"][:, :4].astype(np.float32)
    gaze = np.nan_to_num(z["gaze"][:, :, :2].astype(np.float64)); att = z["attended"].astype(int) - 1
    tk = z["trial_k"].astype(int)
    rng = np.random.default_rng(20260619 + s); perm = np.stack([rng.permutation(4) for _ in range(len(att))])
    T = eeg.shape[-1]; W = min(W, T); st = _wins(T, W); tri = np.tile(np.arange(len(att)), len(st))
    eeg_w = np.concatenate([eeg[:, :, w:w + W] for w in st], 0)
    candp = _zt(np.concatenate([env[:, :, :, w:w + W] for w in st], 0).astype(np.float32), -1)
    gz = np.concatenate([gaze[:, w:w + W, :] for w in st], 0)
    permw = np.concatenate([perm for _ in st], 0); attw = att[tri]
    yslot = np.array([np.flatnonzero(permw[i] == attw[i])[0] for i in range(len(tri))])
    cand = np.stack([candp[i][permw[i]] for i in range(len(candp))])
    return dict(eeg_z=_zt(eeg_w, 2), tgt=candp[np.arange(len(tri)), attw], cand=cand, yslot=yslot,
                attphys=attw, permw=permw, tk=tk[tri],
                gfeat=np.concatenate([gz.mean(1), gz.std(1)], 1),
                spfeat=np.concatenate([bandpow(eeg_w[:, POST], 8, 12), bandpow(eeg_w[:, POST], 13, 30)], 1))


def _val_bin(m, eeg_z, cand, y, bs=512):
    m.eval(); rs = []
    with torch.no_grad():
        for i in range(0, len(eeg_z), bs):
            r = m(torch.from_numpy(eeg_z[i:i + bs]).to(DEV)).cpu()
            rs.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
    s = B.mm_scores(torch.cat(rs, 0), torch.from_numpy(cand)).numpy()
    sm = s[np.arange(len(y)), y][:, None]
    return float(((sm > s).sum(1) / 3).mean())


def train_content(tr, va, seed=0, epochs=60, patience=12):
    """tr/va = dict(eeg_z,tgt,cand,yslot). Early-stop on val binary MM; margin term (LOSO)."""
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(tr["eeg_z"]); rng = np.random.default_rng(seed); best = -1.0; bstate = None; bad = 0
    for ep in range(epochs):
        m.train(); idx = rng.permutation(n)
        for i in range(0, n, 256):
            b = idx[i:i + 256]
            r = m(torch.from_numpy(tr["eeg_z"][b]).to(DEV))
            loss = B.neg_pearson_loss(r, torch.from_numpy(tr["tgt"][b]).to(DEV))
            if MARGIN > 0:
                sc = B.mm_scores(r, torch.from_numpy(tr["cand"][b]).to(DEV))
                loss = loss + MARGIN * F.cross_entropy(12.0 * sc, torch.from_numpy(tr["yslot"][b]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        v = _val_bin(m, va["eeg_z"], va["cand"], va["yslot"])
        if v > best:
            best = v; bstate = {k: vv.detach().cpu().clone() for k, vv in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if bstate is not None:
        m.load_state_dict(bstate)
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


def run():
    tr_subj = [s for s in ALLSUB if s != SUBJECT]
    # --- content: train @5s on the 15 training subjects, content-disjoint inner-val ---
    D5 = {s: prep(s, TRAIN_W) for s in ALLSUB}
    all_tk = sorted({int(k) for s in tr_subj for k in np.unique(D5[s]["tk"])})
    rng = np.random.default_rng(7); val_tk = set(rng.choice(all_tk, max(1, int(0.15 * len(all_tk))), replace=False))
    def cat(subs, key, mask_val=None):
        out = []
        for s in subs:
            d = D5[s]; m = np.isin(d["tk"], list(val_tk))
            sel = m if mask_val else ~m
            out.append(d[key][sel])
        return np.concatenate(out)
    tr_d = {k: cat(tr_subj, k, False) for k in ("eeg_z", "tgt", "cand", "yslot")}
    va_d = {k: cat(tr_subj, k, True) for k in ("eeg_z", "tgt", "cand", "yslot")}
    models = [train_content(tr_d, va_d, seed=sd) for sd in range(NSEED)]
    print(f"  s{SUBJECT} content trained ({NSEED} seeds, margin={MARGIN})", flush=True)

    rows = []
    for w_s in WINS:
        W = int(round(w_s * SR))
        Dw = {s: prep(s, W) for s in ALLSUB}
        te = Dw[SUBJECT]
        cs = content_scores(models, te["eeg_z"], te["cand"])                # (n,4) slot
        # spatial LDA on the 15 training subjects @W, gaze-residualised
        Xtr = np.concatenate([Dw[s]["spfeat"] for s in tr_subj]); gtr = np.concatenate([Dw[s]["gfeat"] for s in tr_subj])
        atr = np.concatenate([Dw[s]["attphys"] for s in tr_subj])
        Xte = te["spfeat"]
        if GAZERESID:
            rg = LinearRegression().fit(gtr, Xtr); Xtr = Xtr - rg.predict(gtr); Xte = Xte - rg.predict(te["gfeat"])
        lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, atr)
        pp = np.zeros((len(Xte), 4), np.float32); pp[:, lda.classes_.astype(int)] = lda.predict_proba(Xte)
        sp = np.stack([pp[i][te["permw"][i]] for i in range(len(pp))])      # -> slot space
        y = te["yslot"]; csn = cs - cs.mean(1, keepdims=True); lsp = np.log(sp + 1e-6)
        fs = csn + 1.5 * lsp; rng2 = np.random.default_rng(0)
        null = float(np.mean([acc(fs[rng2.permutation(len(y))], y) for _ in range(50)]))
        row = dict(win_s=w_s, content=acc(cs, y), spatial=acc(sp, y), fused_b15=acc(fs, y),
                   null_four=null, n=len(y), test_subject=SUBJECT, protocol="loso",
                   csn=csn.tolist(), lsp=lsp.tolist(), y=y.tolist())          # for cross-subject OOF-b
        rows.append(row)
        print(f"[fusewin-loso|s{SUBJECT}|w{w_s:g}] content={row['content']:.3f} spatial={row['spatial']:.3f} "
              f"FUSED_b1.5={row['fused_b15']:.3f} null={null:.3f} n={row['n']}", flush=True)

    out = f"{RUN_ROOT}/results/{TAG}/s{SUBJECT}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), default=float)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    run()
