"""Candidate construction that removes the acoustic shortcut, plus the
audio-only acceptance probe that verifies it was removed.

The upstream pipeline z-scores each candidate envelope (`dataloader.py:252`),
which removes scalar loudness but NOT the shape signature loudness leaves
behind: the attended talker is the loudest of the four in 100/100 trials
(median +15.1 dB), and louder speech has lower kurtosis/skew/Gini-sparsity and
a wider dynamic range.  Those are level-free and transfer across content, so a
trivial logistic probe reads the attended talker at 0.65 on content-disjoint
folds — above the deep model's own 0.50.  No subject or content split can
remove it, because it is a property of the target ROLE.

Two constructions here:

  qmatch  — per-window quantile (histogram) matching of the K candidates onto
            a common marginal.  Every marginal statistic the confound imprints
            (kurtosis, skew, Gini, dynamic range, silence fraction) becomes
            identical across candidates BY CONSTRUCTION; only the temporal
            ordering survives, which is exactly the part EEG can track.

  shifted — same-source time-shifted negatives: the positive is the attended
            talker's envelope for this window, the K-1 negatives are the SAME
            talker's envelope from non-overlapping windows of the same trial.
            The candidates are exchangeable by construction, so the audio-only
            Bayes accuracy is exactly chance.  This is the standard
            match-mismatch negative (de Cheveigne; Francart; ICASSP-2024).
"""

import numpy as np

FS = 64
FEATURE_NAMES = ["kurtosis", "skew", "dyn_range_p95_p5", "frac_silence_lt_-0.5",
                 "mod_0.5_4Hz", "mod_4_8Hz", "hf_8_20Hz", "gini_sparsity"]


# ── shortcut removal ───────────────────────────────────────────────────────────

def _zscore_last(x, eps=1e-8):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + eps)


def quantile_match(A, chunk=1000):
    """A: (N,K,T) -> (N,K,T). Forces the K candidates of each window onto a
    common marginal (their average order statistics) while preserving each
    candidate's own temporal ordering."""
    N, K, T = A.shape
    out = np.empty_like(A, dtype=np.float32)
    ar = np.arange(T)
    for s in range(0, N, chunk):
        a = A[s:s + chunk]
        n = a.shape[0]
        order = np.argsort(a, axis=2, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, (n, K, T)), axis=2)
        ref = np.sort(a, axis=2).mean(axis=1, keepdims=True)          # (n,1,T)
        matched = np.take_along_axis(np.broadcast_to(ref, (n, K, T)), ranks, axis=2)
        out[s:s + chunk] = _zscore_last(matched).astype(np.float32)
    return out


def build_shifted_imposters(trial_ids, window_sec, hop_sec, n_neg=3, seed=0):
    """For every window, pick `n_neg` imposter windows from the same trial whose
    time spans do not overlap it.  Same voice, same recording, different
    segment.  Returns (N, n_neg) global window indices.

    Imposters are drawn at RANDOM from the hop-disjoint windows of the trial, so
    a negative never shares any audio with the positive.  (Taking the furthest
    window instead biases imposters toward the trial edges, which have their own
    onset/offset envelope statistics — that alone lifted the audio-only probe to
    0.60 on a chance-0.50 task.)  `n_fallback` counts windows where the trial was
    too short to avoid overlap; it should be 0 for n_neg <= 2 at window/hop = 2.
    """
    gap = max(1, int(np.ceil(window_sec / hop_sec)))
    out = np.full((len(trial_ids), n_neg), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_fallback = 0
    for t in np.unique(trial_ids):
        idx = np.where(trial_ids == t)[0]            # contiguous, temporal order
        n = len(idx)
        for p in range(n):
            valid = [q for q in range(n) if abs(q - p) >= gap]
            rng.shuffle(valid)
            if len(valid) < n_neg:
                n_fallback += 1
                extra = [q for q in range(n) if q != p and q not in valid]
                valid = valid + list(rng.permutation(extra))
            if not valid:                            # single-window trial
                out[idx[p]] = idx[p]
                continue
            pick = (valid * n_neg)[:n_neg]
            out[idx[p]] = idx[np.asarray(pick)]
    assert (out >= 0).all(), "imposter index not assigned"
    return out, n_fallback


# ── audio-only acceptance probe ────────────────────────────────────────────────

def shape_features_batch(E):
    """Level-free shape features of z-scored envelopes. E: (M,T) -> (M,8)."""
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    M, T = E.shape
    p5, p95 = np.percentile(E, [5, 95], axis=1)
    f, P = welch(E, fs=FS, nperseg=min(256, T), axis=1)
    tot = P.sum(1) + 1e-12

    def band(lo, hi):
        return P[:, (f >= lo) & (f < hi)].sum(1) / tot

    a = np.sort(np.abs(E), axis=1)
    w = np.arange(1, T + 1)
    gini = 2 * (a * w).sum(1) / (T * a.sum(1) + 1e-12) - (T + 1) / T
    return np.stack([kurtosis(E, axis=1), skew(E, axis=1), p95 - p5,
                     (E < -0.5).mean(1), band(0.5, 4), band(4, 8),
                     band(8, 20), gini], axis=1).astype(np.float64)


def audio_only_probe(cands, labels, groups, n_splits=5, seed=0):
    """THE acceptance test for shortcut removal.

    cands  : (N,K,T) the candidate envelopes actually fed to the model
    labels : (N,)    index of the attended candidate
    groups : (N,)    content id, so folds are content-disjoint

    Fits a logistic 'attended vs not' on level-free shape features only, then
    takes the per-window argmax over the K candidates.  A shortcut-free
    construction must land at 1/K.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    N, K, T = cands.shape
    X = shape_features_batch(cands.reshape(N * K, T))
    y = np.zeros(N * K, dtype=np.int64)
    y[np.arange(N) * K + labels] = 1
    g = np.repeat(groups, K)

    oof = np.zeros(N * K)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    pred = oof.reshape(N, K).argmax(1)
    return float((pred == labels).mean())
