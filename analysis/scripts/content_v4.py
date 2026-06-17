"""content_v4: multi-target content matcher (spectrogram + HuBERT + semantic).

Tests your semantic + deep-audio ideas in the regime where content actually works
(whole-trial, 4-talker PERMUTED, loudness-equalized) -- NOT the per-window regime where
recon_mix saw chance. The EEG is reconstructed into THREE stimulus spaces and each scores
the permuted candidates; LEARNED MIXTURE WEIGHTS combine them and are logged, so the run
itself reports whether HuBERT/semantic add anything over the spectrogram.

  spec : 28-band mel-spectrogram     (frequency-specific envelope; v3 winner)
  w2v  : HuBERT layer-9 PCA-64       (self-supervised deep audio / phonetic)
  sem  : GPT-2 surprisal/entropy/onset (linguistic content)

EEG-only, transfer + combo. Chance 0.25 (broadband 0.36, spectrogram 0.39).
SPACES env var picks subset (default spec,w2v,sem).
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, ssl_pretrain, zs = _nets.Enc, _nets.ssl_pretrain, _nets.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SPACES = os.environ.get("SPACES", "spec,w2v,sem").split(",")


def load():
    shuffle = os.environ.get("SHUFFLE_EEG", "0") == "1"; rng0 = np.random.default_rng(12345)
    D = {}
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        cand = {}
        if "spec" in SPACES: cand["spec"] = zs(z["env"][:, :4].astype(np.float32), -1)        # (N,4,28,T)
        if "w2v" in SPACES:  cand["w2v"] = zs(z["w2v"].astype(np.float32), -1)                 # (N,4,64,T)
        if "sem" in SPACES:  cand["sem"] = zs(z["sem"].astype(np.float32), -1)                 # (N,4,3,T)
        y = z["attended"].astype(int) - 1
        if shuffle: eeg = eeg[rng0.permutation(len(y))]   # EEG-shuffle control
        D[s] = (eeg, cand, y)
    T = min(D[s][0].shape[-1] for s in D)
    return {s: (e[:, :, :T], {k: v[:, :, :, :T] for k, v in c.items()}, y) for s, (e, c, y) in D.items()}


def permute(cand, y, seed):                       # same per-trial perm applied to all spaces
    rng = np.random.default_rng(seed); N = len(y); perms = np.stack([rng.permutation(4) for _ in range(N)])
    tgt = np.array([int(np.where(perms[i] == y[i])[0][0]) for i in range(N)], int)
    out = {k: np.stack([v[i][perms[i]] for i in range(N)]) for k, v in cand.items()}
    return out, tgt


class MultiNet(nn.Module):
    def __init__(self, dims, h=64):
        super().__init__(); self.enc = Enc(32, h); self.spaces = list(dims)
        self.dec = nn.ModuleDict({k: nn.Conv1d(h, d, 1) for k, d in dims.items()})
        self.scale = nn.ParameterDict({k: nn.Parameter(torch.tensor(2.3)) for k in dims})
        self.mix = nn.Parameter(torch.zeros(len(dims)))

    def forward(self, eeg, cand):                 # cand[k] (B,4,Dk,T)
        z = self.enc(eeg); recon = {}; per = {}
        for k in self.spaces:
            r = self.dec[k](z); recon[k] = r        # (B,Dk,T)
            B, S, Dk, T = cand[k].shape
            rf = r.reshape(B, 1, Dk * T); cf = cand[k].reshape(B, S, Dk * T)
            rf = (rf - rf.mean(-1, keepdim=True)) / (rf.std(-1, keepdim=True) + 1e-6)
            cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + 1e-6)
            per[k] = (rf * cf).mean(-1) * self.scale[k].exp().clamp(max=100)
        mw = torch.softmax(self.mix, 0)
        logits = sum(mw[i] * per[k] for i, k in enumerate(self.spaces))
        return logits, per, recon, mw


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def to_dev(cand): return {k: torch.as_tensor(v, device=dev) for k, v in cand.items()}


def train(model, Xe, cand, tgt, epochs, lr=1e-3, wd=1e-3, bs=48):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    Xe = torch.as_tensor(Xe, device=dev); cand = to_dev(cand); tgt = torch.as_tensor(tgt, device=dev)
    for _ in range(epochs):
        for b in _bt(len(tgt), bs):
            bi = torch.as_tensor(b, device=dev)
            cb = {k: v[bi] for k, v in cand.items()}
            logits, per, recon, mw = model(Xe[bi], cb)
            idx = torch.arange(len(bi), device=dev)
            rl = sum(F.mse_loss(recon[k], cb[k][idx, tgt[bi]]) for k in model.spaces) / len(model.spaces)
            loss = F.cross_entropy(logits, tgt[bi]) + rl
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evalm(model, Xe, cand, tgt, bs=48):
    model.eval(); Xe = torch.as_tensor(Xe, device=dev); cand = to_dev(cand); P = []
    for i in range(0, len(tgt), bs):
        cb = {k: v[i:i + bs] for k, v in cand.items()}
        logits, _, _, mw = model(Xe[i:i + bs], cb); P.append(logits.argmax(1).cpu().numpy())
    return float((np.concatenate(P) == tgt).mean()), mw.detach().cpu().numpy()


def main():
    D = load(); subs = sorted(D); seeds = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
    dims = {k: D[subs[0]][1][k].shape[2] for k in SPACES}; print(f"SPACES={SPACES} dims={dims}", flush=True)
    res = {m: [] for m in ("scratch", "transfer", "combo")}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        P = {s: permute(D[s][1], D[s][2], sd * 1000 + s) for s in subs}
        allE = np.concatenate([D[s][0] for s in subs]); ssl_state = ssl_pretrain(allE, dev, epochs=60).state_dict()
        per = {m: {} for m in res}; mws = []
        for s in subs:
            Xe = D[s][0]; C, tgt = P[s]
            oE = np.concatenate([D[o][0] for o in subs if o != s])
            oC = {k: np.concatenate([P[o][0][k] for o in subs if o != s]) for k in SPACES}; ot = np.concatenate([P[o][1] for o in subs if o != s])
            pre = train(MultiNet(dims), oE, oC, ot, 20)
            cpre = MultiNet(dims); cpre.enc.load_state_dict(ssl_state); cpre = train(cpre, oE, oC, ot, 20)
            a = {m: [] for m in res}; lastmw = None
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, tgt):
                trC = {k: C[k][tr] for k in SPACES}; teC = {k: C[k][te] for k in SPACES}
                a["scratch"].append(evalm(train(MultiNet(dims), Xe[tr], trC, tgt[tr], 50), Xe[te], teC, tgt[te])[0])
                mt = MultiNet(dims); mt.load_state_dict(pre.state_dict())
                ac, mw = evalm(train(mt, Xe[tr], trC, tgt[tr], 20, lr=5e-4), Xe[te], teC, tgt[te]); a["transfer"].append(ac); lastmw = mw
                mc = MultiNet(dims); mc.load_state_dict(cpre.state_dict())
                a["combo"].append(evalm(train(mc, Xe[tr], trC, tgt[tr], 20, lr=5e-4), Xe[te], teC, tgt[te])[0])
            for m in res: per[m][s] = np.mean(a[m])
            mws.append(lastmw)
            print(f"seed{sd} S{s:2d} scratch={per['scratch'][s]:.3f} transfer={per['transfer'][s]:.3f} combo={per['combo'][s]:.3f} mix={np.round(lastmw,2)}", flush=True)
        for m in res: res[m].append(np.mean([per[m][s] for s in subs]))
        print(f"seed{sd} MEANS: " + " ".join(f"{m}={res[m][-1]:.3f}" for m in res) + f"  mean_mix({SPACES})={np.round(np.mean(mws,0),3)}", flush=True)
    print(f"\n=== content_v4 multi-target ({SPACES}) MEAN (chance .25; spectrogram-only=0.39) ===")
    for m in res:
        a = np.array(res[m]); print(f"  {m:9s} = {a.mean():.3f} ± {a.std():.3f}")


if __name__ == "__main__":
    main()
