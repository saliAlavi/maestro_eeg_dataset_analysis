"""content_snr: is the recon matcher's content signal BOTTOM-UP (loudness/salience) or TOP-DOWN?

The attended stream was +3..18 dB louder during presentation (trials.csv SNR), so the EEG entrained
to a louder stream even though the candidate features are power-equalized. Test: stratify the recon
matcher's per-trial content accuracy by the trial's stimulus SNR. If accuracy SCALES with SNR ->
the signal is largely bottom-up salience; if FLAT -> a genuine top-down/attentional component.
Recon matcher (ImprovedEnc spec, raw-corr -> confound-immune), inter-subject trial-disjoint.
"""
import os, importlib.util, numpy as np, torch, pandas as pd
from scipy.stats import pearsonr
os.environ.setdefault("ENC", "improved"); os.environ["SSL"] = "0"; os.environ["FILM"] = "0"
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_v6 as V6
from content_attr import Recon, train
dev = V6.dev
CSV = "/fs/ess/PAS2301/Data/AAD Data Collection/experiment_data/trials.csv"


def snr_by_trial():
    df = pd.read_csv(CSV)
    out = {}
    for _, r in df.iterrows():
        t = str(r["Trial No."]).strip()
        if t.isdigit():
            out[int(t)] = float(r["SNR"])
    return out


@torch.no_grad()
def pred(m, E, C, bs=48):
    m.eval(); P = []
    for i in range(0, len(E), bs):
        P.append(m(torch.as_tensor(E[i:i + bs], device=dev), torch.as_tensor(C[i:i + bs], device=dev)).argmax(1).cpu().numpy())
    return np.concatenate(P)


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, Y, TK, SB = V6.load(); nb = C.shape[2]
    snr = snr_by_trial()
    # 5-fold trial-disjoint; each trial predicted once (pooled across subjects)
    correct = np.zeros(len(Y), bool)
    trials = np.unique(TK); folds = np.array_split(np.random.permutation(trials), 5)
    for fi, te_tr in enumerate(folds):
        te = np.isin(TK, te_tr); tr = ~te
        m = train(Recon(nb, 16), E[tr], C[tr], Y[tr], ep=25)
        correct[te] = pred(m, E[te], C[te]) == Y[te]
        print(f"  fold{fi} done acc={correct[te].mean():.3f}", flush=True)
    # per-trial accuracy (avg over subjects) + SNR
    tk_u = np.unique(TK); pacc = np.array([correct[TK == t].mean() for t in tk_u])
    psnr = np.array([snr.get(int(t), np.nan) for t in tk_u])
    ok = ~np.isnan(psnr); pacc, psnr, tk_u = pacc[ok], psnr[ok], tk_u[ok]
    r, p = pearsonr(psnr, pacc)
    print(f"\n=== CONTENT vs SNR (recon matcher, inter-subject trial-disjoint; chance 0.25) ===")
    print(f"  overall acc = {correct.mean():.3f}   n_trials={len(tk_u)}")
    print(f"  Pearson r(SNR, per-trial acc) = {r:+.3f}  (p={p:.3f})")
    # tertile split
    q1, q2 = np.percentile(psnr, [33, 67])
    for name, mask in [("LOW  SNR", psnr <= q1), ("MID  SNR", (psnr > q1) & (psnr <= q2)), ("HIGH SNR", psnr > q2)]:
        sm = correct[np.isin(TK, tk_u[mask])]
        print(f"  {name} (SNR {psnr[mask].min():.0f}-{psnr[mask].max():.0f}): acc={sm.mean():.3f}  n={len(sm)}")


if __name__ == "__main__":
    main()
