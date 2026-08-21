"""Fusion x decision-window curve, within-subject (task #20, user steer = "Fusion x window").

EEG-only two-branch fuse, evaluated over a curve of decision windows so the four-way pushes
past 0.40 honestly:
  content branch : VLAAI backward (28-band spectrogram reconstruction), trained @5s (data-rich,
                   fully-convolutional -> scores at any window), 3-seed reconstruction ensemble.
  spatial branch : posterior alpha+beta log band-power, GAZE-RESIDUALISED (regress the window's
                   gaze out, train-fold fit) -> shrinkage LDA -> attended location posterior ->
                   permuted slot space. Retrained per window (band-power matches the eval window).
  fusion         : z-normalised content scores + b * spatial log-posterior. b is chosen
                   ADMISSIBLY (per outer fold, on the OTHER folds' out-of-fold predictions -- never
                   the fold being scored). A fixed-b=1.5 variant is also reported.

Per window we report content-only / spatial-only / fused(OOF-b) / fused(b1.5) + the EEG-shuffle
null (permute the fused scores against the labels) so the window gain stays confound-free.
Candidates are the four real talkers -> chance 0.25 at every window. Writes one JSON per subject.
"""
import glob, os, sys, json, numpy as np, torch
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import backward as B

CACHE = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
RUN_ROOT = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad__neuroclip_aad"
SR = 64.0; TRAIN_W = 320
POST = [13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]   # T7,T8,CP*,P*,PO,O*
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = int(sys.argv[1])
WINS = [float(x) for x in os.environ.get("WINS", "5,10,15,20,30").split(",")]
NSEED = int(os.environ.get("NSEED", "5"))
GAZERESID = os.environ.get("GAZERESID", "1") == "1"
BGRID = [0, .5, 1, 1.5, 2, 3, 4]
TAG = os.environ.get("TAG", "curve_fusewin_within")


def _zt(x, ax):
    return ((x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)).astype(np.float32)


def bandpow(e, lo, hi):
    b, a = butter(4, [lo / (SR / 2), hi / (SR / 2)], "band")
    return np.log(np.mean(filtfilt(b, a, e, axis=-1) ** 2, -1) + 1e-12)


def acc(scores, y):
    return float((scores.argmax(1) == y).mean())


def load(s):
    z = np.load(sorted(glob.glob(f"{CACHE}/s{s}_main_*_pa2_af64.npz"))[0])
    eeg = z["eeg"][:, :32].astype(np.float64)
    env = z["env"][:, :4].astype(np.float32)                        # (100,4,28,T) 4 real talkers
    gaze = np.nan_to_num(z["gaze"][:, :, :2].astype(np.float64))
    att = z["attended"].astype(int) - 1
    rng = np.random.default_rng(20260619 + s)
    perm = np.stack([rng.permutation(4) for _ in range(len(att))])  # (100,4) slot k -> physical
    return eeg, env, gaze, att, perm


def _bin_acc(scores, y):
    sm = scores[np.arange(len(y)), y][:, None]
    return float(((sm > scores).sum(1) / (scores.shape[1] - 1)).mean())


@torch.no_grad()
def _val_bin(m, eeg_z, cand, y, bs=512):
    m.eval(); rs = []
    for i in range(0, len(eeg_z), bs):
        r = m(torch.from_numpy(eeg_z[i:i + bs]).to(DEV)).cpu()
        rs.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
    return _bin_acc(B.mm_scores(torch.cat(rs, 0), torch.from_numpy(cand)).numpy(), y)


def train_content(tr_eeg, tr_tgt, va_eeg, va_cand, va_y, seed=0, epochs=60, patience=12):
    """Production-matched: early-stop on inner-val binary MM accuracy (like train_bwd)."""
    torch.manual_seed(seed)
    m = B.build_backward("vlaai", hidden=128, n_blocks=4, n_out=28).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(tr_eeg); rng = np.random.default_rng(seed); best = -1.0; best_state = None; bad = 0
    for ep in range(epochs):
        m.train(); idx = rng.permutation(n)
        for i in range(0, n, 128):
            b = idx[i:i + 128]
            loss = B.neg_pearson_loss(m(torch.from_numpy(tr_eeg[b]).to(DEV)),
                                      torch.from_numpy(tr_tgt[b]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        v = _val_bin(m, va_eeg, va_cand, va_y)
        if v > best:
            best = v; best_state = {k: vv.detach().cpu().clone() for k, vv in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


@torch.no_grad()
def content_scores(models, eeg_z, cand, bs=512):
    rs = []
    for m in models:
        m.eval(); out = []
        for i in range(0, len(eeg_z), bs):
            r = m(torch.from_numpy(eeg_z[i:i + bs]).to(DEV)).cpu()
            out.append((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6))
        rs.append(torch.cat(out, 0))
    return B.mm_scores(torch.stack(rs).mean(0), torch.from_numpy(cand)).numpy()   # (N,4) slot scores


def _wins(T, W):
    return list(range(0, T - W + 1, max(1, W // 2))) or [0]


def run():
    eeg, env, gaze, att, perm = load(SUBJECT)
    T = eeg.shape[-1]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(len(att)), att))

    # --- content models: train NSEED per fold @5s (once), early-stopped on a trial-disjoint
    #     inner-val (matches the production train_bwd content quality) ---
    st5 = _wins(T, TRAIN_W); tri5 = np.tile(np.arange(len(att)), len(st5))
    eeg5 = _zt(np.concatenate([eeg[:, :, w:w + TRAIN_W] for w in st5], 0), 2)
    cand5p = _zt(np.concatenate([env[:, :, :, w:w + TRAIN_W] for w in st5], 0).astype(np.float32), -1)
    perm5 = np.concatenate([perm for _ in st5], 0); att5 = att[tri5]
    tgt5 = cand5p[np.arange(len(tri5)), att5]                       # attended physical talker 28-band
    cand5 = np.stack([cand5p[i][perm5[i]] for i in range(len(cand5p))])            # slot-indexed
    yslot5 = np.array([np.flatnonzero(perm5[i] == att5[i])[0] for i in range(len(tri5))])
    fold_models = []
    for fi, (trn, _) in enumerate(folds):
        rng = np.random.default_rng(100 + fi)                      # trial-disjoint inner-val (~15%)
        va_tr = np.concatenate([np.random.default_rng(100 + fi + c).permutation(trn[att[trn] == c])
                                [:max(1, int(round(0.15 * (att[trn] == c).sum())))]
                                for c in np.unique(att[trn])])
        itr = np.setdiff1d(trn, va_tr)
        mtr = np.isin(tri5, itr); mva = np.isin(tri5, va_tr)
        fold_models.append([train_content(eeg5[mtr], tgt5[mtr], eeg5[mva], cand5[mva], yslot5[mva], seed=sd)
                            for sd in range(NSEED)])
        print(f"  s{SUBJECT} fold{fi} content trained ({NSEED} seeds, early-stop)", flush=True)

    rows = []
    for w_s in WINS:
        W = min(int(round(w_s * SR)), T)
        st = _wins(T, W); triw = np.tile(np.arange(len(att)), len(st))
        eegw = np.concatenate([eeg[:, :, x:x + W] for x in st], 0)          # (Nw,32,W) raw
        eegz = _zt(eegw, 2)
        candp = _zt(np.concatenate([env[:, :, :, x:x + W] for x in st], 0).astype(np.float32), -1)
        permw = np.concatenate([perm for _ in st], 0); attw = att[triw]
        cand = np.stack([candp[i][permw[i]] for i in range(len(candp))])    # slot-indexed
        yslot = np.array([np.flatnonzero(permw[i] == attw[i])[0] for i in range(len(triw))])
        gz = np.concatenate([gaze[:, x:x + W, :] for x in st], 0)
        gfeat = np.concatenate([gz.mean(1), gz.std(1)], 1)
        spfeat = np.concatenate([bandpow(eegw[:, POST], 8, 12), bandpow(eegw[:, POST], 13, 30)], 1)

        CS = np.zeros((len(triw), 4), np.float32); SP = np.zeros((len(triw), 4), np.float32)
        FO = np.full(len(triw), -1)
        for fi, (trn, tst) in enumerate(folds):
            mtr = np.isin(triw, trn); mte = np.isin(triw, tst); FO[mte] = fi
            CS[mte] = content_scores(fold_models[fi], eegz[mte], cand[mte])
            if GAZERESID:
                rg = LinearRegression().fit(gfeat[mtr], spfeat[mtr])
                Xtr = spfeat[mtr] - rg.predict(gfeat[mtr]); Xte = spfeat[mte] - rg.predict(gfeat[mte])
            else:
                Xtr, Xte = spfeat[mtr], spfeat[mte]
            lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, attw[mtr])
            pp = np.zeros((mte.sum(), 4), np.float32)
            pp[:, lda.classes_.astype(int)] = lda.predict_proba(Xte)       # physical posterior
            SP[mte] = np.stack([pp[i][permw[mte][i]] for i in range(mte.sum())])   # -> slot space

        csn = CS - CS.mean(1, keepdims=True); lsp = np.log(SP + 1e-6)
        # admissible OOF-tuned b: pick b on OTHER folds, apply to this fold
        fused = np.zeros(len(yslot), int)
        for fi in range(5):
            other, this = FO != fi, FO == fi
            bstar = max(BGRID, key=lambda b: acc(csn[other] + b * lsp[other], yslot[other]))
            fused[this] = (csn[this] + bstar * lsp[this]).argmax(1)
        f_oof = float((fused == yslot).mean())
        f_15 = acc(csn + 1.5 * lsp, yslot)
        fs = csn + 1.5 * lsp; rng = np.random.default_rng(0)
        null = float(np.mean([acc(fs[rng.permutation(len(yslot))], yslot) for _ in range(50)]))
        row = dict(win_s=w_s, content=acc(CS, yslot), spatial=acc(SP, yslot),
                   fused_oof=f_oof, fused_b15=f_15, null_four=null, n=len(yslot),
                   test_subject=SUBJECT, protocol="within")
        rows.append(row)
        print(f"[fusewin|s{SUBJECT}|w{w_s:g}] content={row['content']:.3f} spatial={row['spatial']:.3f} "
              f"FUSED_oof={f_oof:.3f} FUSED_b1.5={f_15:.3f} null={null:.3f} n={row['n']}", flush=True)

    out = f"{RUN_ROOT}/results/{TAG}/s{SUBJECT}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2, default=float)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    run()
