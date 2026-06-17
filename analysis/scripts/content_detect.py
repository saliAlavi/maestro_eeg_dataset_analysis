"""content_detect: BINARY attended-audio DETECTION (match-mismatch) with REAL competing
talkers. Chance 0.5 (the field-standard AAD detection metric; cf. our 4-class ID ~0.40).

The ImprovedEnc recon matcher reconstructs the 28-band spec from EEG and scores each of the 4
physical speakers by Pearson corr. For each test trial the binary decision is
    score(attended) > score(competitor)
for every OTHER talker in {1,2,3,4} (real, simultaneously-present unattended streams; no
time-shift imposter -> no roll artifact). Detection acc = mean of those indicators.
REGIME=within|trial|loso. ENC=improved|basic. SHUFFLE_EEG=1 -> null (must be ~0.5).
"""
import os, numpy as np, torch
os.environ.setdefault("ENC", "improved"); os.environ["SSL"] = "0"; os.environ["FILM"] = "0"
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_v6 as V6
from content_attr import Recon, train
from sklearn.model_selection import StratifiedKFold
dev = V6.dev
REGIME = os.environ.get("REGIME", "within")


@torch.no_grad()
def scores(m, E, C, bs=48):
    m.eval(); S = []
    for i in range(0, len(E), bs):
        S.append(m(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(C[i:i + bs], device=dev)).cpu().numpy())
    return np.concatenate(S)                                   # (N,4) corr*scale per speaker


def binary_acc(sc, y):
    """mean over the 3 competitors of [score(attended) > score(competitor)] -> chance 0.5."""
    N = len(y); wins = tot = 0
    for i in range(N):
        a = sc[i, y[i]]
        for k in range(4):
            if k == y[i]: continue
            wins += int(a > sc[i, k]); tot += 1
    return wins / tot


def fit_eval(E, C, Y, tr, te, ep):
    m = train(Recon(C.shape[2], 16), E[tr], C[tr], Y[tr], ep=ep)
    return binary_acc(scores(m, E[te], C[te]), Y[te])


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, Y, TK, SB = V6.load(); nb = C.shape[2]
    print(f"REGIME={REGIME} ENC={os.environ['ENC']} SHUFFLE_EEG={V6.SHUF} n={len(Y)}  (binary detection, chance 0.5)", flush=True)
    accs = []
    if REGIME == "within":
        for s in np.unique(SB):
            m_ = SB == s; Es, Cs, Ys = E[m_], C[m_], Y[m_]; pr = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Es, Ys):
                pr.append((fit_eval(Es, Cs, Ys, tr, te, 40), len(te)))
            a = sum(p * n for p, n in pr) / sum(n for _, n in pr); accs.append(a)
            print(f"  S{int(s)+1:2d} det={a:.3f}", flush=True)
    elif REGIME == "loso":
        for s in np.unique(SB):
            a = fit_eval(E, C, Y, np.where(SB != s)[0], np.where(SB == s)[0], 25); accs.append(a)
            print(f"  held-out S{int(s)+1:2d} det={a:.3f}", flush=True)
    else:
        trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
        for fi, te_tr in enumerate(folds):
            te = np.where(np.isin(TK, te_tr))[0]; tr = np.where(~np.isin(TK, te_tr))[0]
            a = fit_eval(E, C, Y, tr, te, 25); accs.append(a)
            print(f"  fold{fi} det={a:.3f}", flush=True)
    a = np.array(accs)
    print(f"\n=== BINARY attended-audio DETECTION  {REGIME}  (chance 0.5) {'<-- EEG-SHUFFLE NULL' if V6.SHUF else ''} ===")
    print(f"  mean={a.mean():.3f} +- {a.std():.3f}")


if __name__ == "__main__":
    main()
