"""Transfer (#1) + masked-SSL (#2) for 4-way attended-source decoding.

Compact EEG encoder (channel-mixing temporal conv) + gaze, whole-trial. Three modes
share the SAME data/encoder/eval so gains are directly comparable:
  scratch  : per-subject 5-fold from random init.
  transfer : pretrain classifier on the OTHER 15 subjects (pooled, supervised),
             fine-tune on the target's train fold.
  ssl      : pretrain the encoder by masked-CHANNEL reconstruction on all pooled
             unlabeled trials (learns spatial covariance structure WITHOUT noise/
             dropout, which hurt source_net), then fine-tune per subject 5-fold.
Reports per-subject + mean for each mode vs source_net (within 0.471).
"""
import glob, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold

SC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_spec/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def zs(x, ax=-1): return (x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-6)
def gfeat(gz):
    g = gz[:, :, :2]
    return np.concatenate([g.mean(1), g.std(1), np.percentile(g, 10, 1), np.percentile(g, 90, 1)], 1)


def load():
    shuffle = os.environ.get("SHUFFLE_INPUT", "0") == "1"; rng0 = np.random.default_rng(12345)
    D = {}
    for s in range(1, 17):
        f = glob.glob(f"{SC}/s{s}_main_*_pa2*.npz")
        if not f: continue
        z = np.load(f[0])
        eeg = zs(z["eeg"].astype(np.float32), 2)           # (N,32,T) per-ch z-score
        gz = zs(gfeat(z["gaze"]).astype(np.float32), 0); y = z["attended"].astype(int) - 1
        if shuffle:                                        # break (EEG+gaze)<->label pairing
            p = rng0.permutation(len(y)); eeg = eeg[p]; gz = gz[p]
        D[s] = (eeg, gz, y)
    T = min(D[s][0].shape[-1] for s in D)                  # common length across subjects
    return {s: (e[:, :, :T], g, y) for s, (e, g, y) in D.items()}


class Enc(nn.Module):
    def __init__(self, C=32, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, h, 25, padding=12), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(0.3),
            nn.Conv1d(h, h, 15, padding=7), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(0.3),
            nn.Conv1d(h, h, 15, padding=7), nn.BatchNorm1d(h), nn.ELU())
    def forward(self, x): return self.net(x)               # (B,h,T)


class Clf(nn.Module):
    def __init__(self, gdim, h=64, enc=None):
        super().__init__()
        self.enc = enc or Enc(h=h)
        self.head = nn.Sequential(nn.Linear(h + gdim, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, 4))
    def forward(self, eeg, g):
        z = self.enc(eeg).mean(-1)                          # global temporal pool
        return self.head(torch.cat([z, g], 1))


class Recon(nn.Module):                                     # masked-channel SSL
    def __init__(self, C=32, h=64):
        super().__init__(); self.enc = Enc(C, h); self.dec = nn.Conv1d(h, C, 1)
    def forward(self, x): return self.dec(self.enc(x))


def batches(n, bs, shuffle=True):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train_clf(model, Xe, Xg, y, epochs, lr=1e-3, wd=1e-3):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xe, Xg, y = torch.tensor(Xe, device=dev), torch.tensor(Xg, device=dev), torch.tensor(y, device=dev)
    for ep in range(epochs):
        for b in batches(len(y), 64):
            bi = torch.tensor(b, device=dev)
            opt.zero_grad(); loss = F.cross_entropy(model(Xe[bi], Xg[bi]), y[bi], label_smoothing=0.05)
            loss.backward(); opt.step()
    return model


def acc(model, Xe, Xg, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(Xe, device=dev), torch.tensor(Xg, device=dev)).argmax(1).cpu().numpy()
    return (p == y).mean()


def ssl_pretrain(allE, epochs=60):
    m = Recon().to(dev).train(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    E = torch.tensor(allE, device=dev)
    for ep in range(epochs):
        for b in batches(len(E), 128):
            bi = torch.tensor(b, device=dev); x = E[bi]
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=dev) > 0.3).float()  # zero 30% channels
            opt.zero_grad(); rec = m(x * mask)
            loss = (((rec - x) ** 2) * (1 - mask)).sum() / ((1 - mask).sum() * x.shape[-1] + 1e-6)
            loss.backward(); opt.step()
    return m.enc


def run_once(D, subs, gdim, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    res = {m: {} for m in ("scratch", "transfer", "ssl", "combo")}
    allE = np.concatenate([D[s][0] for s in subs]); ssl_enc_state = ssl_pretrain(allE).state_dict()
    for s in subs:
        Xe, Xg, y = D[s]; skf = StratifiedKFold(5, shuffle=True, random_state=0)
        oE = np.concatenate([D[o][0] for o in subs if o != s]); oG = np.concatenate([D[o][1] for o in subs if o != s]); oy = np.concatenate([D[o][2] for o in subs if o != s])
        pre = train_clf(Clf(gdim), oE, oG, oy, epochs=25)
        cpre = Clf(gdim); cpre.enc.load_state_dict(ssl_enc_state); cpre = train_clf(cpre, oE, oG, oy, epochs=25)
        a_sc, a_tr, a_ss, a_co = [], [], [], []
        for tr, te in skf.split(Xe, y):
            a_sc.append(acc(train_clf(Clf(gdim), Xe[tr], Xg[tr], y[tr], 60), Xe[te], Xg[te], y[te]))
            mt = Clf(gdim); mt.load_state_dict(pre.state_dict())
            a_tr.append(acc(train_clf(mt, Xe[tr], Xg[tr], y[tr], 25, lr=5e-4), Xe[te], Xg[te], y[te]))
            ms = Clf(gdim); ms.enc.load_state_dict(ssl_enc_state)
            a_ss.append(acc(train_clf(ms, Xe[tr], Xg[tr], y[tr], 60), Xe[te], Xg[te], y[te]))
            mc = Clf(gdim); mc.load_state_dict(cpre.state_dict())
            a_co.append(acc(train_clf(mc, Xe[tr], Xg[tr], y[tr], 25, lr=5e-4), Xe[te], Xg[te], y[te]))
        for k, v in (("scratch", a_sc), ("transfer", a_tr), ("ssl", a_ss), ("combo", a_co)): res[k][s] = np.mean(v)
        print(f"  seed{seed} S{s:2d} scratch={res['scratch'][s]:.3f} transfer={res['transfer'][s]:.3f} ssl={res['ssl'][s]:.3f} combo={res['combo'][s]:.3f}", flush=True)
    return {m: np.mean([res[m][s] for s in subs]) for m in res}


def main():
    import os
    D = load(); subs = sorted(D); gdim = D[subs[0]][1].shape[1]
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    base = {1:.261,2:.652,3:.618,4:.823,5:.501,6:.363,7:.582,8:.293,9:.271,10:.485,11:.274,12:.345,13:.491,14:.347,15:.853,16:.374}
    perseed = {m: [] for m in ("scratch", "transfer", "ssl", "combo")}
    for sd in seeds:
        r = run_once(D, subs, gdim, sd)
        for m in perseed: perseed[m].append(r[m])
        print(f"seed{sd} MEANS: " + " ".join(f"{m}={r[m]:.3f}" for m in perseed), flush=True)
    print(f"\n=== {len(seeds)}-SEED MEAN±STD (source_net base={np.mean(list(base.values())):.3f}) ===")
    for m in ("scratch", "transfer", "ssl", "combo"):
        a = np.array(perseed[m]); print(f"  {m:9s} = {a.mean():.3f} ± {a.std():.3f}")


if __name__ == "__main__":
    main()
