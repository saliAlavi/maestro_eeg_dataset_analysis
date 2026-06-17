"""content_v5: all content levers combined.

EEG side (richer features):  broadband(1-10) + delta(1-4) + theta(4-8) band signals +
                             their Hilbert amplitudes -> explicit entrainment/phase.
Stimulus side (multi-target): spectrogram(28) + HuBERT(64) + semantic(3) reconstruction.
Matching: backward-reconstruction correlation per space  +  a CCA/contrastive branch
          (EEG and spectrogram projected into a shared space, correlated).
All branch scores combined by learned mixture weights. 4-talker PERMUTED, loudness-
equalized, whole-trial, EEG-only. transfer/combo. Chance 0.25 (multi-target=0.41).
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import StratifiedKFold
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, ssl_pretrain, zs = _nets.Enc, _nets.ssl_pretrain, _nets.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SPACES = os.environ.get("SPACES", "spec,w2v,sem").split(",")
USE_CCA = os.environ.get("CCA", "1") == "1"
USE_PHASE = os.environ.get("PHASE", "1") == "1"


def _band(x, lo, hi, fs=64):
    b, a = butter(4, [lo / (fs / 2), min(hi, fs / 2 - 1) / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def enrich(eeg):                                  # (N,32,T) -> (N, 32 or 160, T) z-scored
    if not USE_PHASE:
        return zs(eeg, -1)
    d = _band(eeg, 1, 4); t = _band(eeg, 4, 8)
    da = np.abs(hilbert(d, axis=-1)); ta = np.abs(hilbert(t, axis=-1))
    return zs(np.concatenate([eeg, d, t, da, ta], axis=1).astype(np.float32), -1)


def load():
    shuffle = os.environ.get("SHUFFLE_EEG","0")=="1"; rng0=np.random.default_rng(12345)
    D = {}
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = enrich(z["eeg"].astype(np.float32))
        cand = {}
        if "spec" in SPACES: cand["spec"] = zs(z["env"][:, :4].astype(np.float32), -1)
        if "w2v" in SPACES:  cand["w2v"] = zs(z["w2v"].astype(np.float32), -1)
        if "sem" in SPACES:  cand["sem"] = zs(z["sem"].astype(np.float32), -1)
        y = z["attended"].astype(int) - 1
        if shuffle: eeg = eeg[rng0.permutation(len(y))]
        D[s] = (eeg, cand, y)
    T = min(D[s][0].shape[-1] for s in D)
    return {s: (e[:, :, :T], {k: v[:, :, :, :T] for k, v in c.items()}, y) for s, (e, c, y) in D.items()}


def permute(cand, y, seed):
    rng = np.random.default_rng(seed); N = len(y); perms = np.stack([rng.permutation(4) for _ in range(N)])
    tgt = np.array([int(np.where(perms[i] == y[i])[0][0]) for i in range(N)], int)
    return {k: np.stack([v[i][perms[i]] for i in range(N)]) for k, v in cand.items()}, tgt


def _corr(r, cand):                               # r (B,Dk,T) ; cand (B,S,Dk,T) -> (B,S)
    B, S, Dk, T = cand.shape
    rf = r.reshape(B, 1, Dk * T); cf = cand.reshape(B, S, Dk * T)
    rf = (rf - rf.mean(-1, keepdim=True)) / (rf.std(-1, keepdim=True) + 1e-6)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + 1e-6)
    return (rf * cf).mean(-1)


class CombinedNet(nn.Module):
    def __init__(self, Cin, dims, h=64, d=32):
        super().__init__(); self.enc = Enc(Cin, h); self.spaces = list(dims)
        self.dec = nn.ModuleDict({k: nn.Conv1d(h, dd, 1) for k, dd in dims.items()})
        self.branches = list(dims) + (["cca"] if USE_CCA else [])
        self.scale = nn.ParameterDict({b: nn.Parameter(torch.tensor(2.3)) for b in self.branches})
        self.mix = nn.Parameter(torch.zeros(len(self.branches)))
        if USE_CCA:                               # shared-embedding matcher on spectrogram
            self.eeg_proj = nn.Conv1d(h, d, 1)
            self.stim_enc = nn.Sequential(nn.Conv1d(dims["spec"], d, 5, padding=2), nn.ELU(), nn.Conv1d(d, d, 1))

    def forward(self, eeg, cand):
        z = self.enc(eeg); per = {}; recon = {}
        for k in self.spaces:
            r = self.dec[k](z); recon[k] = r
            per[k] = _corr(r, cand[k]) * self.scale[k].exp().clamp(max=100)
        if USE_CCA:
            ze = self.eeg_proj(z)                 # (B,d,T)
            B, S, Dk, T = cand["spec"].shape
            zc = self.stim_enc(cand["spec"].reshape(B * S, Dk, T)).reshape(B, S, -1, T)   # (B,S,d,T)
            per["cca"] = _corr(ze, zc) * self.scale["cca"].exp().clamp(max=100)
        mw = torch.softmax(self.mix, 0)
        logits = sum(mw[i] * per[b] for i, b in enumerate(self.branches))
        return logits, per, recon, mw


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def to_dev(c): return {k: torch.as_tensor(v, device=dev) for k, v in c.items()}


def train(model, Xe, cand, tgt, epochs, lr=1e-3, wd=1e-3, bs=40):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    Xe = torch.as_tensor(Xe, device=dev); cand = to_dev(cand); tgt = torch.as_tensor(tgt, device=dev)
    for _ in range(epochs):
        for b in _bt(len(tgt), bs):
            bi = torch.as_tensor(b, device=dev); cb = {k: v[bi] for k, v in cand.items()}
            logits, per, recon, mw = model(Xe[bi], cb)
            idx = torch.arange(len(bi), device=dev)
            rl = sum(F.mse_loss(recon[k], cb[k][idx, tgt[bi]]) for k in model.spaces) / len(model.spaces)
            loss = F.cross_entropy(logits, tgt[bi]) + rl
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evalm(model, Xe, cand, tgt, bs=40):
    model.eval(); Xe = torch.as_tensor(Xe, device=dev); cand = to_dev(cand); P = []; mw = None
    for i in range(0, len(tgt), bs):
        cb = {k: v[i:i + bs] for k, v in cand.items()}
        logits, _, _, mw = model(Xe[i:i + bs], cb); P.append(logits.argmax(1).cpu().numpy())
    return float((np.concatenate(P) == tgt).mean()), mw.detach().cpu().numpy()


def main():
    D = load(); subs = sorted(D); seeds = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
    Cin = D[subs[0]][0].shape[1]; dims = {k: D[subs[0]][1][k].shape[2] for k in SPACES}
    print(f"PHASE={USE_PHASE} CCA={USE_CCA} Cin={Cin} dims={dims}", flush=True)
    res = {m: [] for m in ("scratch", "transfer", "combo")}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        P = {s: permute(D[s][1], D[s][2], sd * 1000 + s) for s in subs}
        allE = np.concatenate([D[s][0] for s in subs]); ssl_state = ssl_pretrain(allE, dev, epochs=50, h=64).state_dict()
        per = {m: {} for m in res}; mws = []
        for s in subs:
            Xe = D[s][0]; C, tgt = P[s]
            oE = np.concatenate([D[o][0] for o in subs if o != s])
            oC = {k: np.concatenate([P[o][0][k] for o in subs if o != s]) for k in SPACES}; ot = np.concatenate([P[o][1] for o in subs if o != s])
            pre = train(CombinedNet(Cin, dims), oE, oC, ot, 20)
            cpre = CombinedNet(Cin, dims); cpre.enc.load_state_dict(ssl_state); cpre = train(cpre, oE, oC, ot, 20)
            a = {m: [] for m in res}; lastmw = None
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, tgt):
                trC = {k: C[k][tr] for k in SPACES}; teC = {k: C[k][te] for k in SPACES}
                a["scratch"].append(evalm(train(CombinedNet(Cin, dims), Xe[tr], trC, tgt[tr], 50), Xe[te], teC, tgt[te])[0])
                mt = CombinedNet(Cin, dims); mt.load_state_dict(pre.state_dict())
                ac, mw = evalm(train(mt, Xe[tr], trC, tgt[tr], 20, lr=5e-4), Xe[te], teC, tgt[te]); a["transfer"].append(ac); lastmw = mw
                mc = CombinedNet(Cin, dims); mc.load_state_dict(cpre.state_dict())
                a["combo"].append(evalm(train(mc, Xe[tr], trC, tgt[tr], 20, lr=5e-4), Xe[te], teC, tgt[te])[0])
            for m in res: per[m][s] = np.mean(a[m])
            mws.append(lastmw)
            print(f"seed{sd} S{s:2d} scratch={per['scratch'][s]:.3f} transfer={per['transfer'][s]:.3f} combo={per['combo'][s]:.3f} mix={np.round(lastmw,2)}", flush=True)
        for m in res: res[m].append(np.mean([per[m][s] for s in subs]))
        print(f"seed{sd} MEANS: " + " ".join(f"{m}={res[m][-1]:.3f}" for m in res) + f"  branches={['spec','w2v','sem']+(['cca'] if USE_CCA else [])} mean_mix={np.round(np.mean(mws,0),3)}", flush=True)
    print(f"\n=== content_v5 (ALL levers: phase={USE_PHASE} cca={USE_CCA}) MEAN (chance .25; multi-target=0.41) ===")
    for m in res:
        a = np.array(res[m]); print(f"  {m:9s} = {a.mean():.3f} ± {a.std():.3f}")


if __name__ == "__main__":
    main()
