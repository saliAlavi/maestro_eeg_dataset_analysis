"""content_v6: improved EEG REPRESENTATION, all levers stacked, leak-proof + airtight.

EEG encoder upgrades (vs the 3-conv baseline):
  (#3 architecture) spatial 1x1 mix -> MULTI-SCALE DILATED temporal convs (dilations 1/2/4/8,
                    captures multi-timescale / TRF-lag structure) -> proj.
  (#4 subject-adapt) per-subject FiLM (learned scale/shift) on the latent.
  (#2 SSL) masked-TIME self-supervised pretrain (mask time spans, reconstruct EEG) on pooled
           data, then fine-tune.
Matching unchanged & leak-proof: reconstruct the spectrogram, RAW-candidate Pearson corr.
4-talker PERMUTED, table-power normalized, EEG-only. Trial-disjoint cross-subject (or WITHIN).
SHUFFLE_EEG=1 = null. Baseline to beat: recon spectrogram trial-disjoint 0.356.
"""
import glob, os, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "models", "transfer_ssl", "nets.py")
_sp = importlib.util.spec_from_file_location("tssl_nets", _p); _nets = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_nets)
zs = _nets.zs
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SHUF = os.environ.get("SHUFFLE_EEG", "0") == "1"
WITHIN = os.environ.get("WITHIN", "0") == "1"
SSL = os.environ.get("SSL", "1") == "1"
FILM = os.environ.get("FILM", "1") == "1"
H = 64


def load():
    E, C, Y, TK, SB = [], [], [], [], []; rng0 = np.random.default_rng(12345)
    for s in range(1, 17):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); eeg = zs(z["eeg"].astype(np.float32), 2)
        spec = zs(z["env"][:, :4].astype(np.float32), -1); y = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int)
        N = len(y); cand = np.empty_like(spec); tgt = np.empty(N, int)
        for i in range(N):
            p = np.random.default_rng(s * 1000 + int(tk[i])).permutation(4); cand[i] = spec[i][p]; tgt[i] = int(np.where(p == y[i])[0][0])
        if SHUF: eeg = eeg[rng0.permutation(N)]
        E.append(eeg); C.append(cand); Y.append(tgt); TK.append(tk); SB.append(np.full(N, s - 1))
    T = min(e.shape[-1] for e in E)
    E = np.concatenate([e[:, :, :T] for e in E]); C = np.concatenate([c[:, :, :, :T] for c in C])
    return E, C, np.concatenate(Y), np.concatenate(TK), np.concatenate(SB)


class ImprovedEnc(nn.Module):
    def __init__(self, C=32, h=H, n_subj=16, film=True):
        super().__init__()
        self.spatial = nn.Conv1d(C, h, 1)
        self.temporal = nn.ModuleList([nn.Conv1d(h, h, 15, dilation=d, padding=7 * d) for d in (1, 2, 4, 8)])
        self.bn = nn.BatchNorm1d(h * 4); self.proj = nn.Conv1d(h * 4, h, 1); self.drop = nn.Dropout(0.3)
        self.film = film
        if film:
            self.fg = nn.Embedding(n_subj, h); self.fb = nn.Embedding(n_subj, h)
            nn.init.ones_(self.fg.weight); nn.init.zeros_(self.fb.weight)

    def forward(self, eeg, subj=None):
        x = self.spatial(eeg)
        x = torch.cat([F.elu(t(x)) for t in self.temporal], 1)
        x = self.drop(self.proj(F.elu(self.bn(x))))
        if self.film and subj is not None:
            x = x * self.fg(subj)[:, :, None] + self.fb(subj)[:, :, None]
        return x


def _corr(r, cand):
    B, S, Dk, T = cand.shape
    rf = r.reshape(B, 1, Dk * T); cf = cand.reshape(B, S, Dk * T)
    rf = (rf - rf.mean(-1, keepdim=True)) / (rf.std(-1, keepdim=True) + 1e-6)
    cf = (cf - cf.mean(-1, keepdim=True)) / (cf.std(-1, keepdim=True) + 1e-6)
    return (rf * cf).mean(-1)


class ReconV6(nn.Module):
    def __init__(self, nb, n_subj=16):
        super().__init__(); self.enc = ImprovedEnc(32, H, n_subj, film=FILM); self.dec = nn.Conv1d(H, nb, 1); self.scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, eeg, cand, subj):
        return _corr(self.dec(self.enc(eeg, subj)), cand) * self.scale.exp().clamp(max=100)


def _bt(n, bs):
    idx = np.random.permutation(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def ssl_pretrain(E, SB, n_subj, epochs=40, bs=64, mask=0.4):
    enc = ImprovedEnc(32, H, n_subj, film=FILM).to(dev); dec = nn.Conv1d(H, 32, 1).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), 1e-3, weight_decay=1e-4)
    Et = torch.as_tensor(E, device=dev); St = torch.as_tensor(SB, device=dev)
    enc.train()
    for _ in range(epochs):
        for b in _bt(len(E), bs):
            bi = torch.as_tensor(b, device=dev); x = Et[bi]; T = x.shape[-1]
            m = (torch.rand(x.shape[0], 1, T, device=dev) > mask).float()      # mask TIME spans
            rec = dec(enc(x * m, St[bi]))
            loss = (((rec - x) ** 2) * (1 - m)).sum() / ((1 - m).sum() * 32 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
    return enc.state_dict()


def train(model, E, C, tg, sb, epochs=25, lr=1e-3, bs=48):
    model.to(dev).train(); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    E = torch.as_tensor(E, device=dev); C = torch.as_tensor(C, device=dev); tg = torch.as_tensor(tg, device=dev); sb = torch.as_tensor(sb, device=dev)
    for _ in range(epochs):
        for b in _bt(len(tg), bs):
            bi = torch.as_tensor(b, device=dev)
            F.cross_entropy(model(E[bi], C[bi], sb[bi]), tg[bi]).backward(); opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def _pred(model, E, C, sb, bs=48):
    model.eval(); P = []
    for i in range(0, len(E), bs):
        P.append(model(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(C[i:i + bs], device=dev), torch.as_tensor(sb[i:i + bs], device=dev)).argmax(1).cpu().numpy())
    return np.concatenate(P)


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, Y, TK, SB = load(); nb = C.shape[2]; n_subj = int(SB.max()) + 1
    ssl_state = ssl_pretrain(E, SB, n_subj) if SSL else None
    print(f"SSL={SSL} SHUFFLE_EEG={SHUF} WITHIN={WITHIN} n={len(Y)} n_subj={n_subj}", flush=True)

    def mk():
        m = ReconV6(nb, n_subj)
        if ssl_state is not None: m.enc.load_state_dict(ssl_state)
        return m
    from sklearn.model_selection import StratifiedKFold
    accs = []
    if WITHIN:
        for s in np.unique(SB):
            mk_ = SB == s; Es, Cs, Ys, Ss = E[mk_], C[mk_], Y[mk_], SB[mk_]; pr = np.zeros(len(Ys), int)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Es, Ys):
                m = train(mk(), Es[tr], Cs[tr], Ys[tr], Ss[tr], epochs=40); pr[te] = _pred(m, Es[te], Cs[te], Ss[te])
            accs.append((pr == Ys).mean()); print(f"  S{int(s)+1:2d} acc={(pr==Ys).mean():.3f}", flush=True)
        print(f"\n=== content_v6 WITHIN-SUBJECT (chance 0.25) ===\n  mean={np.mean(accs):.3f} +- {np.std(accs):.3f}  {'<-NULL' if SHUF else ''}"); return
    trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
    for fi, te_tr in enumerate(folds):
        te = np.isin(TK, te_tr); trm = ~te
        m = train(mk(), E[trm], C[trm], Y[trm], SB[trm], epochs=25)
        pr = _pred(m, E[te], C[te], SB[te]); a = (pr == Y[te]).mean(); accs.append(a)
        print(f"  fold{fi}: n_test={te.sum()} acc={a:.3f}", flush=True)
    print(f"\n=== content_v6 TRIAL-DISJOINT cross-subj (chance 0.25; baseline recon 0.356) ===\n  mean={np.mean(accs):.3f} +- {np.std(accs):.3f}  {'<-EEG-SHUFFLE NULL' if SHUF else ''}")


if __name__ == "__main__":
    main()
