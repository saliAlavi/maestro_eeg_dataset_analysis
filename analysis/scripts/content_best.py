"""content_best: strongest LEAK-PROOF AAD model, by the three required properties.

  * TRIAL-DISENTANGLED: within-subject 5-fold trial-disjoint CV; NO cross-subject pretrain
    (proven to add zero genuine content -- only inflated the leaky branch).
  * CONTENT PERMUTED: content candidates are shuffled per trial; the content branch predicts
    the attended STREAM (slot), and its posteriors are REMAPPED to physical-speaker order so
    the content label is decoupled from spatial position yet can still fuse with the spatial
    branch. (Content matcher is position-agnostic, so this is leak-proof by construction.)
  * LEAK-PROOF: w2v/HuBERT is DROPPED (audio-side foreground-identifiability leak, null ~0.32
    even permuted/trial-disjoint). Clean content = env/spec/sem (nulls ~chance). Validated by
    the full INPUT-SHUFFLE null (break EEG AND gaze <-> label) which must collapse to chance.

Branches: content env/spec/sem (temporal-corr vs frozen audio) + directional (EEG alpha + RICH
23-dim gaze, the 0.547-grade features). Fusion: avg (param-free) or rel (OOF reliability weights).
chance 0.25. Reuses nets from content_2stage.

IMPROVEMENTS (2026-06-20), all opt-in via env so the validated baseline is the default:
  * MSREC=1   -> content branches use a MULTI-SCALE DILATED encoder (1/2/4/8, the content_trf
                 lever, +0.056 there) with a RECONSTRUCTION auxiliary (MSE + 1-corr to the
                 attended candidate) on top of CE. WRECON scales the aux (default 1.0).
  * FUSE=stack-> regularized OOF logistic stacker on per-branch log-posteriors (learns per
                 branch/class weights instead of a scalar; honest, fit on inner-OOF only).
  * FUSE=relv -> reliability weights, but the directional branch is gated PER-TRIAL by gaze
                 validity (gx/L/R_valid): low-validity trials lean on content, not overt gaze.
  * SUBJECTS=1,2 limits subjects (smoke); EPOCHS overrides training length.

VIDEO (2026-06-21), the "what does video add" analysis (needs scripts/extract_vjepa.py cache):
  * VIS=1    -> add a "vis" BRANCH: frozen V-JEPA2 scene+gaze-fovea embeddings -> PCA+shrinkage-
                LDA -> 4-class PHYSICAL speaker (which loudspeaker is foveated, room-frame). This
                is SPATIAL/overt-orienting (report under S3, not as content); fused via the same
                leak-free OOF reliability machinery as the dir branch.
  * VALIGN=1 -> Tier 2, the honest CONTENT test: at TRAIN time only, regularize the MS+recon
                content encoder so its pooled EEG reconstruction aligns (InfoNCE) with the trial's
                frozen V-JEPA2 SCENE embedding; video is DROPPED at test (scoring is EEG<->audio
                corr). Tests whether video improves the EEG-only content representation. Forces
                MSREC on. WALIGN scales the aux (default 0.3). Both default OFF -> baseline intact.
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as _sps
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_base = os.path.dirname(os.path.abspath(__file__))


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


C2 = _imp("c2", os.path.join(_base, "content_2stage.py"))
FeatNet, DirNet, _train, _post, zs = C2.FeatNet, C2.DirNet, C2._train, C2._post, C2.zs
_corr, _bt, dev = C2._corr, C2._bt, C2.dev

RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
VJP = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa"   # V-JEPA2 scene+fovea cache
GZP = os.path.join(_base, "..", "results", "fusion_gaze_features.parquet")
CONTENT = ["env", "spec", "sem"]
VCOLS = ["gx_valid", "L_valid", "R_valid"]
SHUF_IN = os.environ.get("SHUFFLE_INPUT", "0") == "1"
FUSE = os.environ.get("FUSE", "avg")                       # avg | rel | relv | stack
PROTOCOL = os.environ.get("PROTOCOL", "intra")             # intra (within-subj 5-fold) | loso
DIRMODE = os.environ.get("DIRMODE", "mlp")                 # mlp (DirNet EEG+gaze) | lda (shrinkage-LDA gaze)
VIS = os.environ.get("VIS", "0") == "1"                    # Tier 1: add V-JEPA2 scene+fovea vis branch
VALIGN = os.environ.get("VALIGN", "0") == "1"             # Tier 2: train-time EEG<->video align aux
W_ALIGN = float(os.environ.get("WALIGN", "0.1"))         # lowered: the aux must not dominate CE+recon
VALIGN_TGT = os.environ.get("VALIGN_TGT", "scene")        # scene (motion/context) | fovea (orienting)
ALIGN_WARM = int(os.environ.get("ALIGN_WARM", "5"))      # epochs to ramp the align weight from 0
FOVEA_ONLY = os.environ.get("FOVEA_ONLY", "0") == "1"    # vis BRANCH uses only the fovea half (scene
#                                  embedding is near-constant -> mostly noise); align target = VALIGN_TGT
SCENE_DIM = 1024                                           # V-JEPA2 ViT-L hidden (one half of VS)
MSREC = (os.environ.get("MSREC", "0") == "1") or VALIGN    # valign requires the MS+recon encoder
W_RECON = float(os.environ.get("WRECON", "1.0"))
EP = int(os.environ.get("EPOCHS", "30"))
SUBJECTS = [int(x) for x in os.environ.get("SUBJECTS", "").split(",") if x.strip()]
# --- audit-driven additions (2026-06-24) -----------------------------------------------------
TASK = os.environ.get("TASK", "spk")                       # spk(T3,4-class,0.25) | hemi(T1) | inout(T2)
NSEED = int(os.environ.get("NSEED", "1"))                  # # outer-CV seeds to average (stability)
ORDER_RESID = os.environ.get("ORDER_RESID", "0") == "1"   # S3 control: residualize trial-ORDER drift
ORDER_DEG = int(os.environ.get("ORDER_DEG", "6"))         # polynomial degree of the per-subject drift
NPERM = int(os.environ.get("NPERM", "0"))                  # >0 with PERM_NULL=1 -> permutation null distn
PERM_NULL = os.environ.get("PERM_NULL", "0") == "1"
# Task = which grouping the 4-class posteriors/labels collapse to before scoring (decode 4-way, score
# coarser). T1 hemisphere {1,2}|{3,4}; T2 inner{2,3}|outer{1,4}; physical idx are 0-based (spk-1).
_TASK_GROUPS = {"spk": None, "hemi": [[0, 1], [2, 3]], "inout": [[1, 2], [0, 3]]}
CHANCE = 0.25 if TASK == "spk" else 0.5
BRANCHES = CONTENT + ["dir"] + (["vis"] if VIS else [])   # vis = spatial/foveation branch (S3)
CONTENT_B = CONTENT                                       # env/spec/sem = the genuine CONTENT test
SPATIAL_B = ["dir"] + (["vis"] if VIS else [])           # dir/vis = overt ORIENTING (report under S3)
GROUPED = ["cavg", "cfused", "savg", "sfused"]           # content-only / spatial-only headline keys


# ----------------------------------------------------------------------------- nets
class _MSEnc(nn.Module):
    """Spatial 1x1 -> parallel dilated temporal convs (1,2,4,8) -> project to feature dim."""

    def __init__(self, n_chans, dim, hidden=64, dropout=0.2):
        super().__init__()
        self.spatial = nn.Conv1d(n_chans, hidden, 1)
        self.bn = nn.BatchNorm1d(hidden)
        self.dils = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8)])
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden * 4, dim, 1)

    def forward(self, eeg):                                # (B,C,T) -> (B,dim,T)
        h = F.elu(self.bn(self.spatial(eeg)))
        h = torch.cat([F.elu(c(h)) for c in self.dils], dim=1)
        return self.proj(self.drop(h))


class MSFeatNet(nn.Module):
    """Multi-scale dilated content matcher; returns (4 corr-logits, reconstruction)."""

    def __init__(self, dim):
        super().__init__()
        self.enc = _MSEnc(32, dim)
        self.scale = nn.Parameter(torch.tensor(2.3))
        self.vproj = nn.Linear(SCENE_DIM, dim) if VALIGN else None   # EEG<->scene align head

    def forward(self, eeg, cand):                          # cand (B,4,dim,T)
        r = self.enc(eeg)                                  # (B,dim,T)
        return _corr(r, cand) * self.scale.exp().clamp(max=100), r


def _train_msrec(model, eeg, cand, y, ep, lr=1e-3, bs=48, w_recon=1.0, vtgt=None):
    """Train MSFeatNet with CE on corr-logits + reconstruction aux to the attended candidate.
    If vtgt (per-trial V-JEPA2 embedding) is given, also add a within-batch InfoNCE aux that pulls
    the pooled EEG reconstruction toward the trial's video embedding (Tier 2: video shapes the EEG
    encoder at TRAIN time; it is never used at test).

    FIX (2026-06-24): the raw V-JEPA2 embeddings are near-identical across trials (pairwise cosine
    ~0.9 -- same room, static loudspeakers), so an in-batch InfoNCE over them is degenerate: the
    positives are indistinguishable from the negatives, the loss can't fall, and its gradient just
    perturbs the content encoder (this is why VALIGN HURT content rather than testing it). We
    MEAN-CENTER the target across the train fold (cosine spread ~0.9 -> ~0, fold-internal so no
    leak), giving a well-posed contrastive task; we also WARM UP the weight (ALIGN_WARM epochs)
    so CE+recon establish a content representation before the aux engages."""
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    et = torch.as_tensor(eeg, device=dev); ct = torch.as_tensor(cand, device=dev)
    yt = torch.as_tensor(y, device=dev)
    vt = None
    if vtgt is not None:
        vt = torch.as_tensor(vtgt, device=dev)
        vt = vt - vt.mean(0, keepdim=True)                # fold-internal mean-center -> spread targets
    for e in range(ep):
        wal = W_ALIGN * min(1.0, (e + 1) / max(1, ALIGN_WARM))            # ramp the align weight
        for b in _bt(len(y), bs):
            bi = torch.as_tensor(b, device=dev)
            eb, cb, yb = et[bi], ct[bi], yt[bi]
            scores, r = model(eb, cb)
            ce = F.cross_entropy(scores, yb, label_smoothing=0.05)
            att = cb[torch.arange(len(yb), device=dev), yb]               # (B,dim,T) attended cand
            rl = F.mse_loss(r, att) + (1.0 - _corr(r, att.unsqueeze(1)).squeeze(1).mean())
            loss = ce + w_recon * rl
            if vt is not None and model.vproj is not None and len(bi) > 2:
                pooled = F.normalize(r.mean(-1), dim=1)                   # (B,dim) EEG content embed
                proj = F.normalize(model.vproj(vt[bi]), dim=1)           # (B,dim) projected video
                logits = pooled @ proj.t() / 0.1                          # InfoNCE over the batch
                lab = torch.arange(len(bi), device=dev)
                loss = loss + wal * 0.5 * (F.cross_entropy(logits, lab)
                                           + F.cross_entropy(logits.t(), lab))
            loss.backward(); opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def _post_msrec(model, eeg, cand, bs=48):
    model.eval(); out = []
    for i in range(0, len(eeg), bs):
        s, _ = model(torch.as_tensor(eeg[i:i + bs], device=dev),
                     torch.as_tensor(cand[i:i + bs], device=dev))
        out.append(F.softmax(s, 1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- data
def _load_vis(s, tk, N):
    """Per-subject V-JEPA2 [scene|fovea] matrix aligned to this subject's trials (by trial_k).
    Returns (N, 2*SCENE_DIM) float32; zeros where video/cache is absent (degrades to chance)."""
    vs = np.zeros((N, 2 * SCENE_DIM), np.float32)
    if not (VIS or VALIGN):
        return vs
    vf = os.path.join(VJP, f"s{s}.npz")
    if not os.path.exists(vf):
        print(f"  [vis] WARNING: no V-JEPA2 cache for S{s} ({vf}); vis branch -> zeros", flush=True)
        return vs
    zv = np.load(vf); sc = zv["scene"].astype(np.float32); fo = zv["fovea"].astype(np.float32)
    pres = zv["present"]
    for i in range(N):
        ki = int(tk[i]) - 1
        if 0 <= ki < len(sc) and bool(pres[ki]):
            vs[i] = np.concatenate([sc[ki], fo[ki]])
    return vs


def _vis_slice(VS):                                   # features the vis branch sees
    return VS[:, SCENE_DIM:] if FOVEA_ONLY else VS    # fovea-only drops the near-constant scene half


def _align_tgt(VS):                                   # half feeding the EEG<->video InfoNCE (Tier 2)
    return VS[:, SCENE_DIM:] if VALIGN_TGT == "fovea" else VS[:, :SCENE_DIM]   # VALIGN_TGT only


def _vis_post(Vtr, ytr, Vte, n_comp=24):
    """Frozen V-JEPA2 scene+fovea -> StandardScaler -> PCA -> shrinkage-LDA -> 4-class posteriors.
    Spatial/foveation branch (which loudspeaker is looked at, room-frame). PCA is mandatory: the
    raw 2048-d embedding vastly over-parameterizes ~80 train trials/subject. Train-fold fit only."""
    sc = StandardScaler().fit(Vtr)
    Vtr2, Vte2 = sc.transform(Vtr), sc.transform(Vte)
    k = max(2, min(n_comp, Vtr2.shape[0] - 1, Vtr2.shape[1]))
    pca = PCA(n_components=k).fit(Vtr2)
    Vtr2, Vte2 = pca.transform(Vtr2), pca.transform(Vte2)
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Vtr2, ytr)
    p = m.predict_proba(Vte2).astype(np.float32)
    out = np.zeros((len(Vte2), 4), np.float32)         # map to fixed 4-class layout (folds may
    out[:, m.classes_.astype(int)] = p                 # miss a class -> fewer predict_proba cols)
    return out


def load():
    col = {"env": lambda z: z["env"][:, :4].mean(2, keepdims=True).astype(np.float32),
           "spec": lambda z: z["env"][:, :4].astype(np.float32),
           "sem": lambda z: z["sem"].astype(np.float32)}
    GZ = pd.read_parquet(GZP)
    gcols = [c for c in GZ.columns if c not in ("subject", "trial", "attended", "group", "snr")]
    vcols = [c for c in VCOLS if c in GZ.columns]
    E = []; CD = {k: [] for k in CONTENT}; G = []; V = []; Y = []; TK = []; SB = []; VS = []
    subs = SUBJECTS or list(range(1, 17))
    for s in subs:
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f:
            continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        cand = {k: zs(col[k](z), -1) for k in CONTENT}
        vs = _load_vis(s, tk, N)                          # (N, 2*SCENE_DIM) scene|fovea, 0 if absent
        gs = GZ[GZ.subject == s].set_index("trial")
        gmat = np.zeros((N, len(gcols)), np.float32); vvec = np.full(N, np.nan, np.float32)
        for i in range(N):
            if tk[i] in gs.index:
                gmat[i] = np.nan_to_num(gs.loc[tk[i], gcols].to_numpy(np.float32))
                if vcols:
                    vvec[i] = float(np.nanmean(gs.loc[tk[i], vcols].to_numpy(np.float32)))
        med = np.nanmedian(vvec) if np.isfinite(vvec).any() else 1.0
        vvec = np.where(np.isfinite(vvec), vvec, med)
        g = gmat                                          # RAW gaze; z-score is fit on TRAIN fold only
        E.append(eeg); [CD[k].append(cand[k]) for k in CONTENT]
        G.append(g); V.append(vvec); Y.append(y); TK.append(tk); SB.append(np.full(N, s - 1))
        VS.append(vs)
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E])
    CD = {k: np.concatenate([c[:, :, :, :T] for c in v]) for k, v in CD.items()}
    G = np.concatenate(G); V = np.concatenate(V); Y = np.concatenate(Y)
    TK = np.concatenate(TK); SB = np.concatenate(SB); VS = np.concatenate(VS)
    # CONTENT PERMUTE: shuffle the 4 candidate streams per trial; keep Y PHYSICAL; track P.
    rng = np.random.default_rng(20260619); P = np.zeros((len(Y), 4), int)
    for i in range(len(Y)):
        p = rng.permutation(4); P[i] = p
        for k in CONTENT:
            CD[k][i] = CD[k][i][p]
    if SHUF_IN:                                       # full-pipeline null: break EEG/gaze/video <-> label
        r = np.random.default_rng(99)
        E = E[r.permutation(len(Y))]; G = G[r.permutation(len(Y))]; V = V[r.permutation(len(Y))]
        VS = VS[r.permutation(len(Y))]
    return E, CD, G, V, Y, TK, SB, P, VS


def yperm(Y, P):                                      # slot of the physical-attended in permuted cands
    return np.array([int(np.flatnonzero(P[i] == Y[i])[0]) for i in range(len(Y))])


def remap(slot_post, Pidx):                           # content slot posteriors -> physical-speaker order
    phys = np.zeros_like(slot_post)
    for n in range(len(Pidx)):
        phys[n, Pidx[n]] = slot_post[n]
    return phys


# ----------------------------------------------------------------------------- branches
def fit_branch(E, CD, G, VS, ylab, k, idx, ep):
    if k == "dir":
        return _train(DirNet(G.shape[1]), [E[idx], G[idx]], ylab[idx], ep)
    if MSREC:
        vt = _align_tgt(VS)[idx] if VALIGN else None      # scene|fovea align target (train only)
        return _train_msrec(MSFeatNet(CD[k].shape[2]), E[idx], CD[k][idx], ylab[idx], ep,
                            w_recon=W_RECON, vtgt=vt)
    return _train(FeatNet(CD[k].shape[2]), [E[idx], CD[k][idx]], ylab[idx], ep)


def post_branch(m, E, CD, G, k, idx):
    if k == "dir":
        return _post(m, [E[idx], G[idx]])
    if MSREC:
        return _post_msrec(m, E[idx], CD[k][idx])
    return _post(m, [E[idx], CD[k][idx]])


def _lda_post(Gtr, ytr, Gte):
    """Regularized linear gaze decoder (shrinkage LDA) -> 4-class posteriors. Beats the DirNet
    MLP by ~+0.03 (diagnostic): the MLP overfits gaze on ~80 trials/subject; LDA's Ledoit-Wolf
    shrinkage generalizes better, esp. for subjects who DO overtly orient. Gaze-only, leak-free."""
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Gtr, ytr)
    p = m.predict_proba(Gte).astype(np.float32)
    out = np.zeros((len(Gte), 4), np.float32)          # pad to fixed 4-class layout (a fold may miss a
    out[:, m.classes_.astype(int)] = p                 # class -> fewer predict_proba cols; match _vis_post)
    return out


def _branch_post(E, CD, G, VS, Y, YP, P, k, tr, te, ep):
    """Train branch k on tr, return its TEST posteriors in PHYSICAL-speaker order."""
    if k == "vis":                                    # frozen V-JEPA2 scene+fovea -> physical speaker
        return _vis_post(_vis_slice(VS)[tr], Y[tr], _vis_slice(VS)[te])
    if k == "dir" and DIRMODE == "lda":
        return _lda_post(G[tr], Y[tr], G[te])         # gaze already z-scored on train fold by caller
    ylab = YP if k != "dir" else Y                    # content: permuted-slot label; dir: physical
    post = post_branch(fit_branch(E, CD, G, VS, ylab, k, tr, ep), E, CD, G, k, te)
    return remap(post, P[te]) if k != "dir" else post


def _acc(post, y):                                    # accuracy under the configured TASK (T1/T2/T3)
    g = _TASK_GROUPS[TASK]
    if g is None:
        return (post.argmax(1) == y).mean()
    pc = np.stack([post[:, grp].sum(1) for grp in g], 1)          # collapse 4-class posteriors
    yc = np.empty(len(y), int)
    for gi, grp in enumerate(g):
        for c in grp:
            yc[y == c] = gi
    return (pc.argmax(1) == yc).mean()


def _acc_clf(clf, X, y):                              # stacker: predict_proba -> 4-col -> TASK score
    p = clf.predict_proba(X)
    post = np.zeros((len(X), 4), np.float32); post[:, clf.classes_.astype(int)] = p
    return _acc(post, y)


def _order_resid_subj(X, TK, SB, deg=ORDER_DEG):
    """Per-subject removal of the smooth trial-ORDER drift from each feature dim. LABEL-FREE (uses
    only trial_k), so leak-free like the per-subject z-score. The attended schedule is a DETERMINISTIC
    period-4 function of trial_k (attended==((tk-1)%4)+1, all subjects), which is ORTHOGONAL to any
    smooth/monotonic session drift -> this kills the order nuisance (verified: V-JEPA2 fovea early/late
    decoding 0.74->~0.28) while the period-4 orienting signal survives (0.66->0.69). The honest S3
    motion/order-residualised control for the overt-orienting (vis/dir) branches."""
    Xr = X.astype(np.float64).copy()
    for s in np.unique(SB):
        m = np.where(SB == s)[0]
        tk = TK[m].astype(np.float64); rng = (tk.max() - tk.min()) or 1.0
        t = (tk - tk.min()) / rng
        B = np.vstack([t ** d for d in range(deg + 1)]).T
        coef, *_ = np.linalg.lstsq(B, Xr[m], rcond=None)
        Xr[m] = Xr[m] - B @ coef
    return Xr.astype(np.float32)


def _print_stats(rows, keys):
    """Per-subject inferential stats across the n subjects (no more bare mean+-std): Wilcoxon vs
    chance + bootstrap 95% CI for each key, and paired Wilcoxon for the comparisons that carry the
    paper's claims. Operates on the per-subject scalar accuracies already collected in rows."""
    def boot_ci(v, B=5000):
        r = np.random.default_rng(0)
        bs = np.array([r.choice(v, len(v), replace=True).mean() for _ in range(B)])
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"  -- inferential stats (n={len(rows)} subjects, chance={CHANCE}) --")
    for k in keys:
        v = np.array([r[k] for r in rows if k in r])
        if v.size < 3:
            continue
        lo, hi = boot_ci(v)
        try:
            p = _sps.wilcoxon(v - CHANCE, alternative="greater").pvalue
        except ValueError:
            p = float("nan")
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        print(f"  {k:7s} {v.mean():.3f}  95%CI[{lo:.3f},{hi:.3f}]  vs chance p={p:.1e} {star}")
    comps = [("fused", "dir"), ("fused", "cfused"), ("vis", "dir"), ("sfused", "cfused")]
    out = []
    for a, b in comps:
        if all(a in r and b in r for r in rows):
            va = np.array([r[a] for r in rows]); vb = np.array([r[b] for r in rows])
            if np.allclose(va, vb):
                continue
            try:
                p = _sps.wilcoxon(va, vb).pvalue
            except ValueError:
                p = float("nan")
            out.append(f"{a}>{b}: d={va.mean() - vb.mean():+.3f} p={p:.1e}")
    if out:
        print("  paired (Wilcoxon): " + " | ".join(out))


def _avggrp(tp, subset):                              # param-free mean fusion over a branch subset
    return sum(tp[k] for k in subset) / len(subset) if subset else None


def _relgrp(tp, oof, ytr, subset):
    """rel-weighted fused posteriors over a branch SUBSET (weights = OOF excess-over-chance). Used
    to report CONTENT-only and SPATIAL-only fused numbers separately, so the vis/dir orienting
    branches never inflate a headline labeled 'content'."""
    if not subset:
        return None
    w = {k: max(0.0, (oof[k].argmax(1) == ytr).mean() - 0.25) for k in subset}
    tot = sum(w.values()) or 1.0
    return sum((w[k] / tot) * tp[k] for k in subset)


def _grouped(out, tp, oof, ytr, yte):
    """Add content-only / spatial-only fused accuracies (cfused/sfused) to out, given branch tp and
    (optionally) OOF posteriors. With no oof (FUSE=avg) the groups fall back to mean fusion."""
    out["cavg"] = _acc(_avggrp(tp, CONTENT_B), yte)
    if SPATIAL_B:
        out["savg"] = _acc(_avggrp(tp, SPATIAL_B), yte)
    if oof is None:
        out["cfused"] = out["cavg"]
        if SPATIAL_B:
            out["sfused"] = out["savg"]
        return
    out["cfused"] = _acc(_relgrp(tp, oof, ytr, CONTENT_B), yte)
    if SPATIAL_B:
        out["sfused"] = _acc(_relgrp(tp, oof, ytr, SPATIAL_B), yte)


# ----------------------------------------------------------------------------- evaluate
def evaluate(E, CD, G, V, VS, Y, YP, P, tr, te):
    # gaze z-score fit on TRAIN fold only (no transductive leak); EEG/content already per-trial
    mu = G[tr].mean(0, keepdims=True); sd = G[tr].std(0, keepdims=True) + 1e-6
    G = (G - mu) / sd
    yte = Y[te]; out = {}; tp = {}
    for k in BRANCHES:
        tp[k] = _branch_post(E, CD, G, VS, Y, YP, P, k, tr, te, EP)
        out[k] = _acc(tp[k], yte)
    out["avg"] = _acc(sum(tp[k] for k in BRANCHES) / len(BRANCHES), yte)

    if FUSE == "avg":
        out["fused"] = out["avg"]; _grouped(out, tp, None, None, yte); return out

    # --- modes needing OOF branch posteriors on the TRAIN portion (leak-free) ---
    oof = {k: np.zeros((len(tr), 4), np.float32) for k in BRANCHES}
    for it, iv in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, Y[tr]):
        a, b = tr[it], tr[iv]
        for k in BRANCHES:
            oof[k][iv] = _branch_post(E, CD, G, VS, Y, YP, P, k, a, b, EP)
    _grouped(out, tp, oof, Y[tr], yte)               # content-only / spatial-only headline numbers

    if FUSE in ("rel", "relv"):
        w = {k: max(0.0, (oof[k].argmax(1) == Y[tr]).mean() - 0.25) for k in BRANCHES}
        if FUSE == "rel":
            tot = sum(w.values()) or 1.0
            fused = sum((w[k] / tot) * tp[k] for k in BRANCHES)
        else:                                          # relv: gate dir per-trial by gaze validity
            vte = V[te][:, None]                       # (n,1) validity in [0,1]
            wk = {k: np.full((len(te), 1), w[k], np.float32) for k in BRANCHES}
            wk["dir"] = wk["dir"] * vte                # down-weight overt gaze when validity low
            tot = sum(wk.values()); tot[tot == 0] = 1.0
            fused = sum((wk[k] / tot) * tp[k] for k in BRANCHES)
        out["fused"] = _acc(fused, yte)
        out["_w"] = {k: float(np.mean(w[k] if np.isscalar(w[k]) else w[k])) for k in BRANCHES}
        return out

    if FUSE == "stack":                                # regularized OOF logistic on log-posteriors
        Xtr = np.concatenate([np.log(oof[k] + 1e-9) for k in BRANCHES] + [V[tr][:, None]], 1)
        Xte = np.concatenate([np.log(tp[k] + 1e-9) for k in BRANCHES] + [V[te][:, None]], 1)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), Y[tr])
        out["fused"] = _acc_clf(clf, sc.transform(Xte), yte)
        return out

    out["fused"] = out["avg"]; return out


# ----------------------------------------------------------------------------- LOSO
def _subj_zscore(G, SB):
    """Per-subject unsupervised gaze standardization (each subject by its OWN mean/std over its
    trials; NO labels). The leak-free cross-subject calibration LOSO needs because raw gaze is
    subject-relative/uncalibrated -- without it a pooled gaze decoder sees mismatched scales."""
    Gn = G.copy()
    for s in np.unique(SB):
        m = SB == s
        mu = Gn[m].mean(0, keepdims=True); sd = Gn[m].std(0, keepdims=True) + 1e-6
        Gn[m] = (Gn[m] - mu) / sd
    return Gn


def evaluate_loso(E, CD, G, V, VS, Y, YP, P, tr, te):
    """Leave-one-subject-out: train every branch on the 15 train subjects, score the held-out one.
    Gaze G is already per-subject z-scored (calibration), so no train-fold rescale here."""
    yte = Y[te]; out = {}; tp = {}
    for k in BRANCHES:
        tp[k] = _branch_post(E, CD, G, VS, Y, YP, P, k, tr, te, EP)
        out[k] = _acc(tp[k], yte)
    out["avg"] = _acc(sum(tp[k] for k in BRANCHES) / len(BRANCHES), yte)
    if FUSE == "avg":
        out["fused"] = out["avg"]; _grouped(out, tp, None, None, yte); return out
    oof = {k: np.zeros((len(tr), 4), np.float32) for k in BRANCHES}
    for it, iv in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, Y[tr]):
        a, b = tr[it], tr[iv]
        for k in BRANCHES:
            oof[k][iv] = _branch_post(E, CD, G, VS, Y, YP, P, k, a, b, EP)
    _grouped(out, tp, oof, Y[tr], yte)               # content-only / spatial-only headline numbers
    if FUSE in ("rel", "relv"):
        w = {k: max(0.0, (oof[k].argmax(1) == Y[tr]).mean() - 0.25) for k in BRANCHES}
        if FUSE == "rel":
            tot = sum(w.values()) or 1.0
            fused = sum((w[k] / tot) * tp[k] for k in BRANCHES)
        else:
            vte = V[te][:, None]; wk = {k: np.full((len(te), 1), w[k], np.float32) for k in BRANCHES}
            wk["dir"] = wk["dir"] * vte; tot = sum(wk.values()); tot[tot == 0] = 1.0
            fused = sum((wk[k] / tot) * tp[k] for k in BRANCHES)
        out["fused"] = _acc(fused, yte); out["_w"] = {k: float(w[k]) for k in BRANCHES}
    elif FUSE == "stack":
        Xtr = np.concatenate([np.log(oof[k] + 1e-9) for k in BRANCHES] + [V[tr][:, None]], 1)
        Xte = np.concatenate([np.log(tp[k] + 1e-9) for k in BRANCHES] + [V[te][:, None]], 1)
        scl = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(scl.transform(Xtr), Y[tr])
        out["fused"] = _acc_clf(clf, scl.transform(Xte), yte)
    return out


def _print_grouped(rows):
    """Report the CONTENT and SPATIAL groups separately so an orienting branch (dir/vis) never sits
    inside a number labeled 'content'. cfused = env/spec/sem only (the honest content headline, S1);
    sfused = dir(/vis) only (overt-orienting baseline, S3)."""
    g = {k: np.array([r[k] for r in rows if k in r]) for k in GROUPED}
    print(f"  -- grouped (chance {CHANCE}) --")
    print(f"  CONTENT  cavg={g['cavg'].mean():.3f}  cfused={g['cfused'].mean():.3f} "
          f"+- {g['cfused'].std():.3f}   <- env/spec/sem only (the honest content test)")
    if g["savg"].size:
        print(f"  SPATIAL  savg={g['savg'].mean():.3f}  sfused={g['sfused'].mean():.3f} "
              f"+- {g['sfused'].std():.3f}   <- dir/vis orienting (report under S3, NOT content)")


def _calibrate(G, VS, SB, loso):
    """Per-subject leak-free preprocessing. LOSO needs the cross-subject subject-z-score calibration
    (intra does NOT -- evaluate() z-scores gaze per fold and _vis_post StandardScales per fold, so
    adding it here would perturb the validated intra baseline). ORDER_RESID (residualize the smooth
    trial-order drift, label-free) applies in BOTH; with ORDER_RESID=0 this is a no-op so the
    validated baseline is reproduced exactly."""
    if loso:
        G = _subj_zscore(G, SB)
        if VIS or VALIGN:
            VS = _subj_zscore(VS, SB)
    if ORDER_RESID:                                    # S3: kill the session-order nuisance (label-free)
        G = _order_resid_subj(G, _TK, SB)
        if VIS or VALIGN:
            VS = _order_resid_subj(VS, _TK, SB)
    return G, VS


def _intra_rows(E, CD, G, V, VS, Y, YP, P, SB, seeds):
    """Per-subject within-subject 5-fold accuracies, averaged over `seeds` outer-CV shuffles."""
    rows = []; ws = []
    for s in np.unique(SB):
        idx = np.where(SB == s)[0]
        acc = {k: [] for k in BRANCHES + ["avg", "fused"] + GROUPED}
        for seed in seeds:
            for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, Y[idx]):
                r = evaluate(E, CD, G, V, VS, Y, YP, P, idx[tr_i], idx[te_i])
                for k in acc:
                    acc[k].append(r[k])
                if "_w" in r:
                    ws.append(r["_w"])
        rows.append({k: float(np.mean(v)) for k, v in acc.items()})
    return rows, ws


def _shuffle_inputs(E, G, V, VS, seed):                # break EEG/gaze/video <-> label (one null draw)
    r = np.random.default_rng(seed)
    return (E[r.permutation(len(E))], G[r.permutation(len(G))],
            V[r.permutation(len(V))], VS[r.permutation(len(VS))])


def _hdr(extra):
    enc = "MS+recon" if MSREC else "FeatNet"
    tag = f"TASK={TASK}(ch={CHANCE}) ORDER_RESID={ORDER_RESID} NSEED={NSEED}"
    print(f"content_best  {extra} | ENC={enc} DIR={DIRMODE} | VIS={VIS} VALIGN={VALIGN} | "
          f"FUSE={FUSE} {tag} SHUF_IN={SHUF_IN}", flush=True)


def loso_main():
    torch.manual_seed(0); np.random.seed(0)
    E, CD, G, V, Y, TK, SB, P, VS = load(); YP = yperm(Y, P)
    globals()["_TK"] = TK
    G, VS = _calibrate(G, VS, SB, loso=True)
    _hdr(f"LOSO PERMUTE+remap RICHGAZE={G.shape[1]}d(subj-z) n={len(Y)}")
    rows = []; ws = []
    for s in np.unique(SB):
        te = np.where(SB == s)[0]; tr = np.where(SB != s)[0]
        r = evaluate_loso(E, CD, G, V, VS, Y, YP, P, tr, te)
        rows.append(r)
        if "_w" in r:
            ws.append(r["_w"])
        print("  S%2d " % (s + 1) + " ".join(f"{k}={r[k]:.3f}" for k in BRANCHES)
              + f"  avg={r['avg']:.3f} FUSED={r['fused']:.3f}", flush=True)
    print(f"\n=== content_best LOSO (chance {CHANCE}){'  <-- INPUT-SHUFFLE NULL' if SHUF_IN else ''} ===")
    for k in BRANCHES + ["avg", "fused"]:
        v = np.array([r[k] for r in rows]); print(f"  {k:5s} = {v.mean():.3f} +- {v.std():.3f}")
    _print_grouped(rows)
    _print_stats(rows, BRANCHES + ["avg", "fused", "cfused", "sfused"])
    if ws:
        print("  rel weights: " + " ".join(f"{k}={np.mean([w[k] for w in ws]):.3f}" for k in BRANCHES))


def perm_null_main(E, CD, G, V, VS, Y, YP, P, SB):
    """Permutation NULL DISTRIBUTION (not a single draw): re-run the within-subject pipeline NPERM
    times with EEG/gaze/video independently shuffled vs label, build the null for the headline
    metrics, and report empirical one-sided p = (#null >= obs + 1)/(NPERM + 1)."""
    obs_rows, _ = _intra_rows(E, CD, G, V, VS, Y, YP, P, SB, [0])
    obs = {k: np.mean([r[k] for r in obs_rows]) for k in ("fused", "cfused", "sfused")}
    print(f"OBSERVED: " + " ".join(f"{k}={obs[k]:.3f}" for k in obs), flush=True)
    null = {k: [] for k in obs}
    for i in range(NPERM):
        Es, Gs, Vs, VSs = _shuffle_inputs(E, G, V, VS, 1000 + i)
        rws, _ = _intra_rows(Es, CD, Gs, Vs, VSs, Y, YP, P, SB, [0])
        for k in obs:
            null[k].append(np.mean([r[k] for r in rws]))
        print(f"  perm {i + 1}/{NPERM}: " + " ".join(f"{k}={null[k][-1]:.3f}" for k in obs), flush=True)
    print(f"\n=== PERMUTATION NULL (NPERM={NPERM}, chance {CHANCE}) ===")
    for k in obs:
        nd = np.array(null[k]); p = (int((nd >= obs[k]).sum()) + 1) / (NPERM + 1)
        print(f"  {k:7s} obs={obs[k]:.3f}  null={nd.mean():.3f}+-{nd.std():.3f} "
              f"(95pct={np.percentile(nd, 95):.3f})  p={p:.3f}")


def main():
    torch.manual_seed(0); np.random.seed(0)
    if PROTOCOL == "loso":
        return loso_main()
    E, CD, G, V, Y, TK, SB, P, VS = load(); YP = yperm(Y, P)
    globals()["_TK"] = TK
    G, VS = _calibrate(G, VS, SB, loso=False)
    if PERM_NULL and NPERM > 0:
        _hdr(f"PERMUTATION-NULL n={len(Y)}")
        return perm_null_main(E, CD, G, V, VS, Y, YP, P, SB)
    _hdr(f"PERMUTE+remap RICHGAZE={G.shape[1]}d NO-w2v NO-pretrain wrec={W_RECON} wal={W_ALIGN} n={len(Y)}")
    seeds = list(range(NSEED))
    rows, ws = _intra_rows(E, CD, G, V, VS, Y, YP, P, SB, seeds)
    for s, r in zip(np.unique(SB), rows):
        print("  S%2d " % (int(s) + 1) + " ".join(f"{k}={r[k]:.3f}" for k in BRANCHES)
              + f"  avg={r['avg']:.3f} FUSED={r['fused']:.3f}", flush=True)
    print(f"\n=== content_best (chance {CHANCE}){'  <-- INPUT-SHUFFLE NULL' if SHUF_IN else ''} ===")
    for k in BRANCHES + ["avg", "fused"]:
        v = np.array([r[k] for r in rows]); print(f"  {k:5s} = {v.mean():.3f} +- {v.std():.3f}")
    _print_grouped(rows)
    _print_stats(rows, BRANCHES + ["avg", "fused", "cfused", "sfused"])
    if ws:
        print("  rel weights: " + " ".join(f"{k}={np.mean([w[k] for w in ws]):.3f}" for k in BRANCHES))


_TK = None

if __name__ == "__main__":
    main()
