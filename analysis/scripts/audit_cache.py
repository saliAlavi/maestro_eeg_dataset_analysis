"""audit_cache: is the audio-only 'attended is identifiable' result a REAL confound or a
cache/code LEAK? Direct data audit (no EEG, no training tricks).

Checks:
 (1) Is env[trial_k] IDENTICAL across subjects (shared audio -> LOSO memorization expected)?
 (2) Does the attended speaker have a SIMPLE static tell? Linear/logistic on per-candidate
     TIME-MEAN spectrum (28-d), trial-disjoint, predict attended position. If ~0.76 -> the tell
     is a static spectral signature (suspicious). Compare to chance 0.25.
 (3) Raw attended-vs-unattended summary stats (energy, time-mean spectrum L2) for an obvious tell.
 (4) Sanity: are the 4 candidate envelopes within a trial distinct (not duplicated/misaligned)?
"""
import glob, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
RC = "/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials"
NS = 16


def zt(x):  # z-score over time (axis -1), per (trial,speaker,band)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9)


def main():
    SP = []; TK = []; SB = []; ATT = []; ENERGY = []; specfp = {}
    for s in range(1, NS + 1):
        f = glob.glob(f"{RC}/s{s}_main_*_pa2_af64.npz")
        if not f: continue
        z = np.load(f[0]); env = z["env"][:, :4].astype(np.float64)        # (N,4,28,T)
        att = z["attended"].astype(int) - 1; tk = z["trial_k"].astype(int); N = len(att)
        tmean = env.mean(-1)                                                # (N,4,28) time-mean spectrum
        energy = np.log(np.maximum(env.mean((-1, -2)), 1e-12))             # (N,4) log energy
        SP.append(zt(env).mean(-1)); ENERGY.append(energy)                 # z-scored-in-time then time-mean
        TK.append(tk); SB.append(np.full(N, s)); ATT.append(att)
        for i in range(N):                                                 # fingerprint for shared-audio check
            specfp.setdefault(int(tk[i]), []).append((s, np.round(tmean[i].ravel(), 4)))
    SP = np.concatenate(SP); ENERGY = np.concatenate(ENERGY); TK = np.concatenate(TK)
    SB = np.concatenate(SB); ATT = np.concatenate(ATT)
    print(f"loaded {len(ATT)} trials from {len(np.unique(SB))} subjects", flush=True)

    # (1) shared audio across subjects?
    ident = 0; tot = 0
    for t, lst in specfp.items():
        if len(lst) < 2: continue
        tot += 1
        base = lst[0][1]; ident += int(all(np.array_equal(base, o[1]) for _, o in lst))
    print(f"\n(1) env time-mean spectrum IDENTICAL across subjects: {ident}/{tot} trials "
          f"-> {'SHARED audio (LOSO memorizable)' if ident > 0.8*tot else 'NOT identical (audio differs per subject!)'}")

    # (3) attended vs unattended static tell
    att_e = ENERGY[np.arange(len(ATT)), ATT]
    un_e = np.array([ENERGY[i, [k for k in range(4) if k != ATT[i]]].mean() for i in range(len(ATT))])
    print(f"\n(3) log-energy attended-minus-unattended: mean={np.mean(att_e-un_e):+.3f} "
          f"(if ~0, loudness already equalized)")

    # (2) static-spectrum linear leakage test (predict attended POSITION, trial-disjoint by trial group)
    rng = np.random.default_rng(0)
    X = []; Yp = []
    for i in range(len(ATT)):
        perm = np.random.default_rng(1000 + int(SB[i]) * 97 + int(TK[i])).permutation(4)
        X.append(SP[i][perm].reshape(-1))                                  # 4*28 features, permuted
        Yp.append(int(np.where(perm == ATT[i])[0][0]))
    X = np.array(X); Yp = np.array(Yp)
    # GroupKFold by trial_k => trial-disjoint (test trials' audio never in train)
    accs = []
    for tr, te in GroupKFold(5).split(X, Yp, groups=TK):
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], Yp[tr])
        accs.append((lr.predict(X[te]) == Yp[te]).mean())
    print(f"\n(2) LINEAR audio-only on TIME-MEAN spectrum, trial-disjoint (chance 0.25): "
          f"{np.mean(accs):.3f} +- {np.std(accs):.3f}")
    print("    -> if ~0.25: the tell is NOT static-spectral (needs temporal conv); "
          "if high: a static per-speaker spectral signature leaks attended-ness.")

    # (4) candidate distinctness within a trial (mean pairwise corr of the 4 time-mean spectra)
    c = []
    for i in range(0, len(ATT), 50):
        v = SP[i]; cc = np.corrcoef(v); c.append(cc[np.triu_indices(4, 1)].mean())
    print(f"\n(4) mean pairwise corr of 4 candidate spectra within a trial: {np.mean(c):.3f} "
          f"(near 1.0 would mean candidates are near-duplicates => bug)")


if __name__ == "__main__":
    main()
