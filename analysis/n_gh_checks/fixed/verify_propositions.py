"""Numerical verification of the propositions in FIXED_MODEL.md section 7.

Each check is self-contained and prints the measured value beside the value the
proposition predicts.  Run: python verify_propositions.py
"""

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

T, D, K, B = 640, 16, 4, 32


def corr(u, v, eps=1e-9):
    """Per-dimension Pearson correlation over time (the fixed head's score)."""
    u = u - u.mean(0, keepdim=True)
    v = v - v.mean(0, keepdim=True)
    u = u / (u.norm(dim=0, keepdim=True) + eps)
    v = v / (v.norm(dim=0, keepdim=True) + eps)
    return (u * v).sum(0)


def old_head(b, a, w, beta):
    """Upstream scoring: per-timestep normalise, multiply, average, linear+bias."""
    return (w * (F.normalize(b, dim=1) * F.normalize(a, dim=1)).mean(0)).sum() + beta


def p1_gain_invariance():
    """P1: the envelope pipeline is gain-equivariant, the z-score gain-invariant."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    import soundfile as sf
    sys.path.insert(0, "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream/scripts")
    import dataloader as dl

    root = "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset/media/audio"
    tid = sorted(os.listdir(root))[0]
    w, sr = sf.read(sorted(glob.glob(os.path.join(root, tid, "speaker1_*.flac")))[0],
                    dtype="float32")

    def pipe(x):
        e = np.abs(hilbert(x.astype(np.float64)))
        e = sosfiltfilt(butter(4, 20.0 / (sr / 2), btype="low", output="sos"), e)
        e = dl._resample(e.astype(np.float32), sr, 64)
        return (e - e.mean()) / (e.std() + 1e-8)

    base = pipe(w)
    worst = max(np.abs(base - pipe(w * 10 ** (db / 20))).max() for db in (3, 15, -15))
    print(f"P1  max |Δ| over gains ±15 dB          = {worst:.2e}   (predicted: 0, float eps)")


def p5_p6_heads():
    """P5: old head with a constant brain is an audio-only classifier.
       P6: new head with a constant brain is identically zero."""
    b = torch.randn(1, D).expand(T, D).contiguous()      # constant over time
    a = [torch.randn(T, D) for _ in range(K)]
    w, beta = torch.randn(D), torch.randn(1)
    old = torch.stack([old_head(b, x, w, beta) for x in a])
    new = torch.stack([(w * corr(b, x)).sum() for x in a])
    print(f"P5  old head, spread across candidates = {float(old.std()):.4f}   "
          f"(predicted: > 0, a classifier exists)")
    print(f"P6  new head, max |score|              = {float(new.abs().max()):.2e}   "
          f"(predicted: 0, chance is forced)")


def p7_affine():
    """P7: the correlation score is invariant to affine rescaling of a candidate."""
    b, a = torch.randn(T, D), torch.randn(T, D)
    d = (corr(b, 3 * a + 7) - corr(b, a)).abs().max()
    print(f"P7  |corr(b,3a+7) − corr(b,a)|max      = {float(d):.2e}   (predicted: 0)")


def p9_receptive_field():
    """P9: RF = 1 + (k-1)Σd_i, and the padded fraction of a past-only window."""
    rf = lambda k, L, base: 1 + (k - 1) * sum(base ** i for i in range(L))
    old, new = rf(3, 7, 3), rf(3, 5, 2)
    real = (T + 1) / (2 * old)
    print(f"P9  RF old = {old} ({old/64:.1f} s), RF new = {new} ({new/64:.2f} s); "
          f"mean real fraction = {real:.3f}   (predicted: 3^7=2187, 2^6-1=63)")


def p10_rectified_cosine():
    """P10: non-negative embeddings are near-parallel; centred ones are not."""
    out = {}
    for name, f in (("rectified", F.relu), ("centred", lambda z: z)):
        U = F.normalize(f(torch.randn(2000, D) + 0.5), dim=1)
        n = len(U)
        out[name] = float(((U @ U.t()).sum() - n) / (n * (n - 1)))
    print(f"P10 mean pairwise cosine: rectified = {out['rectified']:.4f}, "
          f"centred = {out['centred']:.4f}   (predicted: rectified ≥ 0 and larger)")


def p11_infonce():
    """P11: a collapsed encoder cannot beat log B on the contrastive term."""
    c = torch.randn(B)
    S = c.unsqueeze(0).expand(B, B)                     # every row identical
    tgt = torch.arange(B)
    l1, l2 = F.cross_entropy(S, tgt), F.cross_entropy(S.t(), tgt)
    print(f"P11 collapsed InfoNCE: CE(S,I) = {float(l1):.4f}, CE(Sᵀ,I) = {float(l2):.4f}, "
          f"log B = {np.log(B):.4f}   (predicted: both ≥ log B)")


def p14_audio_free_null():
    """P14: an audio-free branch nulls at exactly 1/K for ANY classifier bias."""
    rng = np.random.default_rng(0)
    n = 200_000
    for label, pred in (("unbiased", rng.integers(0, 4, n)),
                        ("always speaker 2", np.full(n, 2))):
        y = rng.integers(0, 4, n)                       # balanced, independent window
        print(f"P14 permuted acc, {label:16s}     = {(pred == y).mean():.4f}   "
              f"(predicted: 0.2500)")


if __name__ == "__main__":
    print("Verification of FIXED_MODEL.md §7 propositions\n")
    try:
        p1_gain_invariance()
    except Exception as e:                              # needs the audio corpus
        print(f"P1  skipped ({type(e).__name__}: {e})")
    p5_p6_heads()
    p7_affine()
    p9_receptive_field()
    p10_rectified_cosine()
    p11_infonce()
    p14_audio_free_null()
    print("\nP2, P3, P4, P8, P12, P13, P15, P16 are algebraic identities or "
          "distributional arguments; see the proofs in §7.")
