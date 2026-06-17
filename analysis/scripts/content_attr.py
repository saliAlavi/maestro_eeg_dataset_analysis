"""Clean attribution: basic (pooled) vs improved (multi-scale, full-res) EEG encoder,
SAME harness, within-subject 5-fold, multi-seed. No SSL, no FiLM (both shown inert).
Tests whether the within-subject content gain (~0.40 vs prior 0.325) is the ARCHITECTURE
-- specifically full temporal resolution for envelope reconstruction.
"""
import os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
os.environ.setdefault("SSL", "0"); os.environ.setdefault("FILM", "0")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_v6 as V6
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
BasicEnc = _nets.Enc           # 3-layer conv, pools time by 8 -> coarse
dev = "cuda" if torch.cuda.is_available() else "cpu"; V6.dev = dev; H = 64
ENC = os.environ.get("ENC", "improved")


class Recon(nn.Module):
    def __init__(self, nb, n_subj):
        super().__init__()
        if ENC == "basic":
            self.enc = BasicEnc(32, H); self.basic = True
        else:
            self.enc = V6.ImprovedEnc(32, H, n_subj, film=False); self.basic = False
        self.dec = nn.Conv1d(H, nb, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand, subj=None):
        # both encoders return (B,h,T) full-resolution; only the conv stack differs
        # (basic = single-scale sequential, improved = multi-scale dilated)
        r = self.dec(self.enc(eeg))
        return V6._corr(r, cand) * self.scale.exp().clamp(max=100)


def train(m, E, C, tg, ep=40, lr=1e-3, bs=48):
    m.to(dev).train(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3)
    E = torch.as_tensor(E, device=dev); C = torch.as_tensor(C, device=dev); tg = torch.as_tensor(tg, device=dev)
    for _ in range(ep):
        for b in V6._bt(len(tg), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(m(E[bi], C[bi]), tg[bi]).backward(); opt.step(); opt.zero_grad()
    return m


@torch.no_grad()
def pred(m, E, C, bs=48):
    m.eval(); P = []
    for i in range(0, len(E), bs):
        P.append(m(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(C[i:i + bs], device=dev)).argmax(1).cpu().numpy())
    return np.concatenate(P)


def main():
    E, C, Y, TK, SB = V6.load(); nb = C.shape[2]; n_subj = int(SB.max()) + 1
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    print(f"ENC={ENC} n={len(Y)} seeds={seeds}", flush=True)
    seedmeans = []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd); subj_acc = []
        for s in np.unique(SB):
            m_ = SB == s; Es, Cs, Ys = E[m_], C[m_], Y[m_]; pr = np.zeros(len(Ys), int)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Es, Ys):
                mdl = train(Recon(nb, n_subj), Es[tr], Cs[tr], Ys[tr]); pr[te] = pred(mdl, Es[te], Cs[te])
            subj_acc.append((pr == Ys).mean())
        seedmeans.append(np.mean(subj_acc)); print(f"  seed{sd} mean={np.mean(subj_acc):.3f}", flush=True)
    a = np.array(seedmeans)
    print(f"\n=== {ENC} encoder, within-subject, {len(seeds)}-seed (chance 0.25) ===\n  mean={a.mean():.3f} +- {a.std():.3f}")


if __name__ == "__main__":
    main()
