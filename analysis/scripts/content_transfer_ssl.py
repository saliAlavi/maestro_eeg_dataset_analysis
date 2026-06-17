"""Content-based AAD with transfer + masked-SSL (re-attack the chance-level content task).

PURE content match-mismatch (NO gaze, NO fixed position): EEG -> reconstruct the
attended broadband envelope -> correlate with the 4 attendable talkers' envelopes in a
per-trial RANDOM order. Label = permuted index of the attended talker, so direction/
location cannot be a shortcut. Whole-trial (content needs long integration).

Tests whether cross-subject pretraining (subject-independent backward model) + masked-
channel SSL rescue content signal that single-subject training leaves at chance (~0.29).
Modes: scratch / transfer / ssl / combo. 4-way, chance 0.25.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
# load the self-contained nets.py by path (avoids triggering src/models/__init__ chain)
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p)
_nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
Enc, ssl_pretrain, zs = _nets.Enc, _nets.ssl_pretrain, _nets.zs

RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    import os
    shuffle = os.environ.get("SHUFFLE_EEG", "0") == "1"
    rng0 = np.random.default_rng(12345)
    D = {}
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2.npz")
        if not f: continue
        z = np.load(f[0])
        eeg = zs(z["eeg"].astype(np.float32), 2)               # (N,32,T)
        bb = zs(z["env"][:, :4].mean(2).astype(np.float32), 2)  # (N,4,T) broadband talkers
        y = z["attended"].astype(int) - 1
        if shuffle:                                            # EEG-shuffle: break EEG<->talker pairing
            eeg = eeg[rng0.permutation(len(y))]
        D[s] = (eeg, bb, y)
    T = min(D[s][0].shape[-1] for s in D)
    return {s: (e[:, :, :T], b[:, :, :T], y) for s, (e, b, y) in D.items()}


def permute(bb, y, seed):
    """Return cand (N,4,T) in per-trial random order + target index (permuted attended)."""
    rng = np.random.default_rng(seed)
    N = len(y); cand = np.empty_like(bb); tgt = np.empty(N, int)
    for i in range(N):
        p = rng.permutation(4); cand[i] = bb[i][p]; tgt[i] = int(np.where(p == y[i])[0][0])
    return cand, tgt


class ContentNet(nn.Module):
    def __init__(self, h=64):
        super().__init__(); self.enc = Enc(32, h); self.dec = nn.Conv1d(h, 1, 1)
        self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand):                              # cand (B,4,T)
        r = self.dec(self.enc(eeg)).squeeze(1)                 # (B,T) reconstructed envelope
        rz = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6)
        cz = (cand - cand.mean(-1, keepdim=True)) / (cand.std(-1, keepdim=True) + 1e-6)
        scores = (rz.unsqueeze(1) * cz).mean(-1) * self.scale.exp().clamp(max=100)
        return r, scores


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train(model, Xe, C, tgt, epochs, lr=1e-3, wd=1e-3, bs=64):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
    Xe, C, tgt = (torch.as_tensor(t, device=dev) for t in (Xe, C, tgt))
    idxb = torch.arange(len(tgt), device=dev)
    for _ in range(epochs):
        for b in _bt(len(tgt), bs):
            bi = torch.as_tensor(b, device=dev)
            r, sc = model(Xe[bi], C[bi]); att = C[bi][torch.arange(len(bi), device=dev), tgt[bi]]
            rz = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6)
            az = (att - att.mean(-1, keepdim=True)) / (att.std(-1, keepdim=True) + 1e-6)
            recon = F.mse_loss(r, att) + (1 - (rz * az).mean())
            loss = F.cross_entropy(sc, tgt[bi]) + recon
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def acc(model, Xe, C, tgt):
    model.eval()
    _, sc = model(torch.as_tensor(Xe, device=dev), torch.as_tensor(C, device=dev))
    return float((sc.argmax(1).cpu().numpy() == tgt).mean())


def main():
    import os
    D = load(); subs = sorted(D); seeds = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
    per = {m: [] for m in ("scratch", "transfer", "ssl", "combo")}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        # precompute permuted candidates per subject (fixed for this seed)
        P = {s: permute(D[s][1], D[s][2], sd * 1000 + s) for s in subs}
        allE = np.concatenate([D[s][0] for s in subs]); ssl_state = ssl_pretrain(allE, dev, epochs=60).state_dict()
        res = {m: {} for m in per}
        for s in subs:
            Xe = D[s][0]; C, tgt = P[s]
            oE = np.concatenate([D[o][0] for o in subs if o != s])
            oC = np.concatenate([P[o][0] for o in subs if o != s]); ot = np.concatenate([P[o][1] for o in subs if o != s])
            pre = train(ContentNet(), oE, oC, ot, 25)
            cpre = ContentNet(); cpre.enc.load_state_dict(ssl_state); cpre = train(cpre, oE, oC, ot, 25)
            a = {m: [] for m in per}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, tgt):
                a["scratch"].append(acc(train(ContentNet(), Xe[tr], C[tr], tgt[tr], 60), Xe[te], C[te], tgt[te]))
                mt = ContentNet(); mt.load_state_dict(pre.state_dict())
                a["transfer"].append(acc(train(mt, Xe[tr], C[tr], tgt[tr], 25, lr=5e-4), Xe[te], C[te], tgt[te]))
                ms = ContentNet(); ms.enc.load_state_dict(ssl_state)
                a["ssl"].append(acc(train(ms, Xe[tr], C[tr], tgt[tr], 60), Xe[te], C[te], tgt[te]))
                mc = ContentNet(); mc.load_state_dict(cpre.state_dict())
                a["combo"].append(acc(train(mc, Xe[tr], C[tr], tgt[tr], 25, lr=5e-4), Xe[te], C[te], tgt[te]))
            for m in per: res[m][s] = np.mean(a[m])
            print(f"seed{sd} S{s:2d} scratch={res['scratch'][s]:.3f} transfer={res['transfer'][s]:.3f} ssl={res['ssl'][s]:.3f} combo={res['combo'][s]:.3f}", flush=True)
        for m in per: per[m].append(np.mean([res[m][s] for s in subs]))
        print(f"seed{sd} MEANS: " + " ".join(f"{m}={per[m][-1]:.3f}" for m in per), flush=True)
    print("\n=== CONTENT (permuted, no gaze/direction) MEAN over seeds (chance 0.25) ===")
    for m in ("scratch", "transfer", "ssl", "combo"):
        a = np.array(per[m]); print(f"  {m:9s} = {a.mean():.3f} ± {a.std():.3f}")


if __name__ == "__main__":
    main()
