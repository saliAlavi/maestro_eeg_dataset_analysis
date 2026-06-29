"""content_loc: one end-to-end network that predicts the ATTENDED STREAM (content-task,
permuted-slot label -> reported metric is CONTENT) but is ALSO given LOCATION information.

Why two paths: the content task uses permuted candidates (position hidden), so a spatial
"where" cue can only help if the net knows which slot sits at which location. We therefore
feed each candidate slot its SPATIAL CODE (hemisphere, inner/outer of the physical speaker
behind it). The net then has:
  * CONTENT path: EEG -> reconstruct -> temporal-corr vs each permuted candidate (env/spec,
    NO w2v leak) -> per-slot content logits.
  * LOCATION path: EEG-alpha + gaze -> predicted attention direction, matched against each
    slot's spatial code -> per-slot location prior.
  * learned GATE combines them (gate init 0 => pure content; learns how much location to use).

Reported accuracy is on the permuted-slot (content) label. use_loc=False => pure content (~0.42).
Leak check: the spatial code is a candidate property, so under input-shuffle (EEG AND gaze
broken vs label) you still can't tell WHICH location is attended -> null must be chance.
Within-subject 5-fold trial-disjoint, no pretrain. chance 0.25.
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import StratifiedKFold

_base = os.path.dirname(os.path.abspath(__file__))


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


C2 = _imp("c2", os.path.join(_base, "content_2stage.py"))
ImprovedEnc, _corr, zs, BasicEnc = C2.ImprovedEnc, C2._corr, C2.zs, C2.BasicEnc

RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
GZP = os.path.join(_base, "..", "results", "fusion_gaze_features.parquet")
dev = "cuda" if torch.cuda.is_available() else "cpu"
H = 64
CFEAT = os.environ.get("CFEAT", "spec")              # content feature: env | spec
USE_LOC = os.environ.get("USE_LOC", "1") == "1"
SHUF_IN = os.environ.get("SHUFFLE_INPUT", "0") == "1"
EP = int(os.environ.get("EPOCHS", "30"))
HEMI = np.array([-1., -1., 1., 1.], np.float32)       # physical 0..3: {1,2}=L, {3,4}=R
INOUT = np.array([-1., 1., 1., -1.], np.float32)      # {2,3}=inner(+1), {1,4}=outer(-1)


def load():
    col = {"env": lambda z: z["env"][:, :4].mean(2, keepdims=True).astype(np.float32),
           "spec": lambda z: z["env"][:, :4].astype(np.float32)}[CFEAT]
    GZ = pd.read_parquet(GZP)
    gcols = [c for c in GZ.columns if c not in ("subject", "trial", "attended", "group", "snr")]
    E = []; CD = []; G = []; Y = []; SB = []
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f:
            continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(y)
        cd = zs(col(z), -1)
        gs = GZ[GZ.subject == s].set_index("trial")
        gmat = np.zeros((N, len(gcols)), np.float32)
        for i in range(N):
            if tk[i] in gs.index:
                gmat[i] = np.nan_to_num(gs.loc[tk[i], gcols].to_numpy(np.float32))
        E.append(eeg); CD.append(cd); G.append(gmat); Y.append(y); SB.append(np.full(N, s - 1))  # RAW gaze
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); CD = np.concatenate([c[:, :, :, :T] for c in CD])
    G = np.concatenate(G); Y = np.concatenate(Y); SB = np.concatenate(SB)
    # CONTENT PERMUTE: shuffle candidates per trial; label = attended SLOT; SLOC = each slot's
    # spatial code (hemi, inner/outer) of the physical speaker behind it.
    rng = np.random.default_rng(20260619)
    Yp = np.zeros(len(Y), int); SLOC = np.zeros((len(Y), 4, 2), np.float32)
    for i in range(len(Y)):
        p = rng.permutation(4)
        CD[i] = CD[i][p]
        Yp[i] = int(np.flatnonzero(p == Y[i])[0])
        SLOC[i] = np.stack([HEMI[p], INOUT[p]], 1)       # slot j -> location of physical p[j]
    if SHUF_IN:                                          # break EEG AND gaze <-> label
        r = np.random.default_rng(99)
        E = E[r.permutation(len(Y))]; G = G[r.permutation(len(Y))]
    return E, CD, G, SLOC, Yp, SB


class ContentLoc(nn.Module):
    def __init__(self, dim, gdim, use_loc=True):
        super().__init__()
        self.enc = ImprovedEnc(32, H, film=False); self.dec = nn.Conv1d(H, dim, 1)
        self.scale = nn.Parameter(torch.tensor(2.3)); self.use_loc = use_loc
        if use_loc:
            self.denc = BasicEnc(32, H)
            # explicitly DECODE the attended location code (hemi, inner/outer) from EEG-alpha+gaze
            self.lochead = nn.Sequential(nn.Linear(H + gdim, 64), nn.ELU(), nn.Dropout(0.3),
                                         nn.Linear(64, 2), nn.Tanh())
            self.gate = nn.Parameter(torch.tensor(1.0))   # init ON; location can contribute
            self.ltemp = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg, cand, gaze, sloc):              # cand (B,4,dim,T), sloc (B,4,2)
        cl = _corr(self.dec(self.enc(eeg)), cand) * self.scale.exp().clamp(max=100)   # (B,4)
        if not self.use_loc:
            return cl, None
        loc_hat = self.lochead(torch.cat([self.denc(eeg).mean(-1), gaze], 1))         # (B,2) decoded loc
        d2 = ((loc_hat.unsqueeze(1) - sloc) ** 2).sum(-1)                             # (B,4) dist to slot codes
        ll = -d2 * self.ltemp.exp().clamp(max=50)                                     # closest slot wins
        return cl + self.gate * ll, loc_hat


def _train(model, E, CD, G, SL, Y, idx, ep, bs=48):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    te = lambda a: torch.as_tensor(a, device=dev)
    for _ in range(ep):
        perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            yb = te(Y[b]); sl = te(SL[b])
            out, loc_hat = model(te(E[b]), te(CD[b]), te(G[b]), sl)
            loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            if loc_hat is not None:                          # aux: decode the attended slot's loc code
                true_loc = sl[torch.arange(len(b), device=dev), yb]
                loss = loss + 0.3 * F.mse_loss(loc_hat, true_loc)
            loss.backward(); opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def _acc(model, E, CD, G, SL, Y, idx, bs=64):
    model.eval(); te = lambda a: torch.as_tensor(a, device=dev); pr = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        pr.append(model(te(E[b]), te(CD[b]), te(G[b]), te(SL[b]))[0].argmax(1).cpu().numpy())
    pr = np.concatenate(pr)
    return (pr == Y[idx]).mean()


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, CD, G, SL, Y, SB = load()
    print(f"content_loc CFEAT={CFEAT} USE_LOC={USE_LOC} SHUF_IN={SHUF_IN} n={len(Y)} gaze={G.shape[1]} "
          f"(content-task label, within-subject 5-fold, chance 0.25)", flush=True)
    accs = []
    for s in np.unique(SB):
        idx = np.where(SB == s)[0]
        fa = []
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, Y[idx]):
            tr, te = idx[tr_i], idx[te_i]
            mu = G[tr].mean(0, keepdims=True); sd = G[tr].std(0, keepdims=True) + 1e-6
            Gn = (G - mu) / sd                            # gaze z-score fit on TRAIN fold only
            m = ContentLoc(CD.shape[2], G.shape[1], use_loc=USE_LOC)
            _train(m, E, CD, Gn, SL, Y, tr, EP)
            fa.append(_acc(m, E, CD, Gn, SL, Y, te))
        accs.append(float(np.mean(fa)))
        print(f"  S{int(s)+1:2d} content-task acc={accs[-1]:.3f}", flush=True)
    a = np.array(accs)
    tag = "content+LOCATION" if USE_LOC else "content-ONLY"
    print(f"\n=== content_loc [{tag}] content-task acc = {a.mean():.3f} +- {a.std():.3f}"
          f"{'  <-- INPUT-SHUFFLE NULL' if SHUF_IN else ''} ===")
    print("  REF: pure content (env/spec/sem fused) ~0.42 | content+spatial late-fusion 0.569 | chance 0.25")


if __name__ == "__main__":
    main()
