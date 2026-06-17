"""content_v3: richer stimulus representation for content AAD.

Instead of matching the EEG reconstruction to a single BROADBAND envelope (content_v2/v1),
match against the full SPECTRO-TEMPORAL representation per talker:
  - 28-band gammatone mel-spectrogram (frequency-specific envelopes), and
  - 28-band ACOUSTIC ONSETS (half-wave-rectified temporal derivative — the edges the
    cortex tracks most strongly).
-> 56 feature-bands per talker. The EEG conv reconstructs the (56,T) representation and
candidates are scored by correlation over the full spectro-temporal pattern. 4-talker
PERMUTED (position removed), table-power normalized (loudness removed), whole-trial,
EEG-only. Modes: scratch / transfer / ssl / combo. Chance 0.25 (broadband baseline 0.36).
FEATS env|onset|both via env var (default both).
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, ssl_pretrain, zs = _nets.Enc, _nets.ssl_pretrain, _nets.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
FEATS = os.environ.get("FEATS", "both")          # env | onset | both


def specfeat(env4):                               # env4 (N,4,28,T) -> (N,4,F,T) z-scored per band
    parts = []
    if FEATS in ("env", "both"):
        parts.append(env4)
    if FEATS in ("onset", "both"):
        on = np.diff(env4, axis=-1, prepend=env4[..., :1]); on = np.maximum(on, 0)   # acoustic onsets
        parts.append(on)
    x = np.concatenate(parts, axis=2).astype(np.float32)                              # (N,4,F,T)
    return zs(x, ax=-1)


def load():
    D = {}
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2.npz")
        if not f: continue
        z = np.load(f[0])
        eeg = zs(z["eeg"].astype(np.float32), 2)
        env4 = z["env"][:, :4].astype(np.float32)          # (N,4,28,T) table-power normalized
        D[s] = (eeg, specfeat(env4), z["attended"].astype(int) - 1)
    T = min(D[s][0].shape[-1] for s in D)
    return {s: (e[:, :, :T], c[:, :, :, :T], y) for s, (e, c, y) in D.items()}


def permute(cand, y, seed):                        # cand (N,4,F,T)
    rng = np.random.default_rng(seed); N = len(y); out = np.empty_like(cand); tgt = np.empty(N, int)
    for i in range(N):
        p = rng.permutation(4); out[i] = cand[i][p]; tgt[i] = int(np.where(p == y[i])[0][0])
    return out, tgt


class SpecNet(nn.Module):
    def __init__(self, Fdim, h=64):
        super().__init__(); self.enc = Enc(32, h); self.dec = nn.Conv1d(h, Fdim, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):                  # cand (B,4,F,T)
        r = self.dec(self.enc(eeg))                # (B,F,T)
        B, S, Fd, T = cand.shape
        rf = r.reshape(B, 1, Fd * T); cf = cand.reshape(B, S, Fd * T)
        rf = (rf - rf.mean(-1, keepdim=True)) / (rf.std(-1, keepdim=True) + 1e-6)
        cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + 1e-6)
        return r, (rf * cf).mean(-1) * self.scale.exp().clamp(max=100)


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train(model, Xe, C, tgt, epochs, lr=1e-3, wd=1e-3, bs=64):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    Xe, C, tgt = (torch.as_tensor(t, device=dev) for t in (Xe, C, tgt))
    for _ in range(epochs):
        for b in _bt(len(tgt), bs):
            bi = torch.as_tensor(b, device=dev); r, sc = model(Xe[bi], C[bi])
            att = C[bi][torch.arange(len(bi), device=dev), tgt[bi]]                   # (B,F,T)
            recon = F.mse_loss(r, att)
            loss = F.cross_entropy(sc, tgt[bi]) + recon
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def acc(model, Xe, C, tgt):
    model.eval(); _, sc = model(torch.as_tensor(Xe, device=dev), torch.as_tensor(C, device=dev))
    return float((sc.argmax(1).cpu().numpy() == tgt).mean())


def main():
    D = load(); subs = sorted(D); seeds = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
    Fdim = D[subs[0]][1].shape[2]; print(f"FEATS={FEATS} Fdim={Fdim}", flush=True)
    res = {m: {} for m in ("scratch", "transfer", "ssl", "combo")}
    persd = {m: [] for m in res}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        P = {s: permute(D[s][1], D[s][2], sd * 1000 + s) for s in subs}
        allE = np.concatenate([D[s][0] for s in subs]); ssl_state = ssl_pretrain(allE, dev, epochs=60).state_dict()
        for s in subs:
            Xe = D[s][0]; C, tgt = P[s]
            oE = np.concatenate([D[o][0] for o in subs if o != s]); oC = np.concatenate([P[o][0] for o in subs if o != s]); ot = np.concatenate([P[o][1] for o in subs if o != s])
            pre = train(SpecNet(Fdim), oE, oC, ot, 25)
            cpre = SpecNet(Fdim); cpre.enc.load_state_dict(ssl_state); cpre = train(cpre, oE, oC, ot, 25)
            a = {m: [] for m in res}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, tgt):
                a["scratch"].append(acc(train(SpecNet(Fdim), Xe[tr], C[tr], tgt[tr], 60), Xe[te], C[te], tgt[te]))
                mt = SpecNet(Fdim); mt.load_state_dict(pre.state_dict())
                a["transfer"].append(acc(train(mt, Xe[tr], C[tr], tgt[tr], 25, lr=5e-4), Xe[te], C[te], tgt[te]))
                ms = SpecNet(Fdim); ms.enc.load_state_dict(ssl_state)
                a["ssl"].append(acc(train(ms, Xe[tr], C[tr], tgt[tr], 60), Xe[te], C[te], tgt[te]))
                mc = SpecNet(Fdim); mc.load_state_dict(cpre.state_dict())
                a["combo"].append(acc(train(mc, Xe[tr], C[tr], tgt[tr], 25, lr=5e-4), Xe[te], C[te], tgt[te]))
            for m in res: res[m][s] = np.mean(a[m])
            print(f"seed{sd} S{s:2d} scratch={res['scratch'][s]:.3f} transfer={res['transfer'][s]:.3f} ssl={res['ssl'][s]:.3f} combo={res['combo'][s]:.3f}", flush=True)
        for m in res: persd[m].append(np.mean([res[m][s] for s in subs]))
        print(f"seed{sd} MEANS: " + " ".join(f"{m}={persd[m][-1]:.3f}" for m in res), flush=True)
    print(f"\n=== content_v3 ({FEATS}, spectro-temporal) MEAN over seeds (chance 0.25; broadband=0.36) ===")
    for m in ("scratch", "transfer", "ssl", "combo"):
        a = np.array(persd[m]); print(f"  {m:9s} = {a.mean():.3f} ± {a.std():.3f}")


if __name__ == "__main__":
    main()
