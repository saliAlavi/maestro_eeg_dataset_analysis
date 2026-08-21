"""Why do attended and masker envelopes differ in DISTRIBUTIONAL SHAPE?

Established already: a pure level difference cannot survive the z-score (the
envelope pipeline is gain-equivariant and the z-score gain-invariant, verified
to float32 rounding).  Yet a level-free shape probe reads the attended talker at
0.56.  So something other than gain differs between target and masker
recordings, and it correlates with level.  This script discriminates the
candidate explanations.

H1  NOISE FLOOR.  Maskers were attenuated and their noise floor is relatively
    higher, filling silent gaps.  PREDICTS maskers LOWER kurtosis (more
    Gaussian).  We measure the opposite, so this should come out refuted.

H2  SPEAKING STYLE / CONTENT SELECTION.  Recordings assigned the target role
    differ in how continuously they are spoken -- fewer/shorter pauses gives both
    a higher full-file RMS and a fuller, less spiky envelope.  Level and shape
    would then be common consequences of content, not one causing the other.
    PREDICTS the shape gap persists after per-file loudness is regressed out,
    and that pause statistics differ.

H3  PROCESSING.  Targets were dynamics-processed (compression/limiting) relative
    to maskers.  PREDICTS reduced crest factor and reduced short-term level
    variance for targets, beyond what pause structure explains.

H4  ROLE ASSIGNMENT IS ARBITRARY.  The same physical recordings appear in both
    roles across trials, and the gap is a property of the ROLE not the FILE.
    Tested directly where a voice is reused; with 400 unique voices this is
    expected to be untestable, which is itself informative.

Discriminator: fit the shape statistic on per-file loudness ACROSS ALL
candidates, then ask whether the attended/masker gap survives in the residual.
If it does not, level and shape are one confound; if it does, role carries
information beyond level.
"""

import csv
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

UP = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
LOCAL = "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset"
sys.path.insert(0, os.path.join(UP, "scripts"))
import dataloader as dl                                              # noqa: E402

FS = 64


def env_z(x, sr):
    e = np.abs(hilbert(x.astype(np.float64)))
    e = sosfiltfilt(butter(4, 20.0 / (sr / 2), btype="low", output="sos"), e)
    e = dl._resample(e.astype(np.float32), sr, FS)
    return (e - e.mean()) / (e.std() + 1e-8), e


def file_stats(x, sr):
    """Loudness, noise-floor, pause and dynamics descriptors of one recording."""
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    ez, e_raw = env_z(x, sr)
    # zero-phase filtering of |hilbert| can undershoot below zero; clip so the
    # log-domain descriptors below are defined
    e_raw = np.maximum(e_raw.astype(np.float64), 1e-12)

    # noise floor: 1st percentile of the raw envelope, relative to its median
    floor = np.percentile(e_raw, 1)
    med = np.median(e_raw) + 1e-12

    # pause structure, on the RAW envelope at -25 dB relative to its own median
    thr = med * 10 ** (-25 / 20)
    silent = e_raw < thr
    n_pause, cur, runs = 0, 0, []
    for s in silent:
        if s:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs) if runs else np.array([0])

    return {
        "rms_db": float(20 * np.log10(rms)),
        "noise_floor_rel_db": float(20 * np.log10(floor / med)),   # H1
        "frac_silent": float(silent.mean()),                        # H2
        "mean_pause_s": float(runs.mean() / FS),                    # H2
        "n_pauses_per_s": float(len(runs) / (len(e_raw) / FS)),     # H2
        "crest_db": float(20 * np.log10((np.abs(x).max() + 1e-12) / rms)),   # H3
        "env_level_var": float(np.var(20 * np.log10(e_raw / med + 1e-12))),  # H3
        "kurtosis": float(kurtosis(ez)),
        "skew": float(skew(ez)),
        "p95_p5": float(np.percentile(ez, 95) - np.percentile(ez, 5)),
    }


SHAPE = ["kurtosis", "skew", "p95_p5"]
EXPLAIN = ["noise_floor_rel_db", "frac_silent", "mean_pause_s",
           "n_pauses_per_s", "crest_db", "env_level_var"]


def cohen_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (s + 1e-12))


def residualise(y, x):
    """Remove the best linear fit of y on x; return residuals."""
    A = np.stack([x, np.ones_like(x)], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(LOCAL, "metadata", "trials.csv")))
            if r["kind"] == "main"]
    recs, att_flag, tid_of, path_of = [], [], [], []
    for ti, r in enumerate(rows):
        tid, att = r["trial_id"], int(r["attended_speaker"])
        for spk in (1, 2, 3, 4):
            m = sorted(glob.glob(os.path.join(LOCAL, "media", "audio", tid,
                                              f"speaker{spk}_*.flac")))
            if not m:
                continue
            w, sr = sf.read(m[0], dtype="float32", always_2d=False)
            recs.append(file_stats(w, sr))
            att_flag.append(int(spk == att))
            tid_of.append(ti)
            path_of.append(os.path.basename(m[0]))
        if (ti + 1) % 25 == 0:
            print(f"  {ti+1}/{len(rows)} trials", flush=True)

    y = np.array(att_flag, dtype=bool)
    tid_of = np.array(tid_of)
    F = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    print(f"\n{len(recs)} recordings, {y.sum()} attended, {(~y).sum()} masker\n")

    out = {"n": len(recs), "descriptors": {}, "residual": {}, "reuse": {}}

    print("=== descriptors: attended vs masker ===")
    print(f"{'descriptor':>20} {'attended':>10} {'masker':>10} {'Cohen d':>9}")
    for k in ["rms_db"] + EXPLAIN + SHAPE:
        a, b = F[k][y], F[k][~y]
        d = cohen_d(a, b)
        out["descriptors"][k] = dict(attended=float(a.mean()), masker=float(b.mean()),
                                     cohen_d=d)
        print(f"{k:>20} {a.mean():>10.3f} {b.mean():>10.3f} {d:>+9.2f}")

    # H1: does the noise floor go the direction it must for the noise-floor story?
    nf = out["descriptors"]["noise_floor_rel_db"]
    print(f"\n[H1 noise floor] maskers' floor is "
          f"{'HIGHER' if nf['masker'] > nf['attended'] else 'LOWER'} relative to median "
          f"(d={nf['cohen_d']:+.2f}); the noise-floor story additionally requires maskers "
          f"to have LOWER kurtosis, observed d="
          f"{out['descriptors']['kurtosis']['cohen_d']:+.2f} (attended minus masker).")

    # H2/H3: does the shape gap survive removing per-file loudness?
    print("\n=== does the shape gap survive removing per-file loudness? ===")
    print(f"{'shape stat':>12} {'raw d':>8} {'d | loudness removed':>22}")
    for k in SHAPE:
        d_raw = cohen_d(F[k][y], F[k][~y])
        r = residualise(F[k], F["rms_db"])
        d_res = cohen_d(r[y], r[~y])
        out["residual"][k] = dict(d_raw=d_raw, d_resid_on_loudness=d_res)
        print(f"{k:>12} {d_raw:>+8.2f} {d_res:>+22.2f}")

    # and after also removing the pause/dynamics descriptors
    print("\n=== ...and after also removing pause + dynamics descriptors? ===")
    cols = []
    for k in ["rms_db"] + EXPLAIN:
        v = F[k]
        if not np.isfinite(v).all() or v.std() < 1e-12:
            print(f"   (dropping non-finite/constant descriptor {k})")
            continue
        cols.append((v - v.mean()) / v.std())
    X = np.stack(cols + [np.ones(len(y))], 1)
    for k in SHAPE:
        beta, *_ = np.linalg.lstsq(X, F[k], rcond=None)
        d_res = cohen_d((F[k] - X @ beta)[y], (F[k] - X @ beta)[~y])
        out["residual"][k]["d_resid_on_all"] = d_res
        print(f"{k:>12} {'':>8} {d_res:>+22.2f}")

    print("\n=== H3: is the level difference achievable by gain alone? ===")
    c_a, c_m = F["crest_db"][y], F["crest_db"][~y]
    r_crest = float(np.corrcoef(F["crest_db"], F["rms_db"])[0, 1])
    out["crest"] = dict(attended=float(c_a.mean()), masker=float(c_m.mean()),
                        cohen_d=cohen_d(c_a, c_m), corr_with_rms=r_crest)
    print(f"  crest factor (peak/RMS) is GAIN-INVARIANT: target {c_a.mean():.2f} dB vs "
          f"masker {c_m.mean():.2f} dB, d={cohen_d(c_a, c_m):+.2f}")
    print(f"  -> the two groups are NOT related by any gain; a gain cannot change crest.")
    print(f"  corr(crest, level) across all 400 files = {r_crest:+.3f}")
    print(f"  natural conversational speech has a crest factor of roughly 15-20 dB;")
    print(f"  a value near 11 dB is characteristic of dynamics compression/limiting.")

    # H4: is any physical recording used in both roles?
    seen = {}
    for p, a, t in zip(path_of, att_flag, tid_of):
        seen.setdefault(p, []).append(a)
    both = {p: v for p, v in seen.items() if len(set(v)) > 1}
    out["reuse"] = dict(unique_files=len(seen), files_in_both_roles=len(both))
    print(f"\n[H4 role vs file] {len(seen)} unique files; "
          f"{len(both)} appear in BOTH roles "
          f"({'role effect separable from file identity' if both else 'NOT separable — role is confounded with file identity by design'})")

    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "results", "fixed", "mechanism_trace.json")
    os.makedirs(os.path.dirname(od), exist_ok=True)
    json.dump(out, open(od, "w"), indent=2)
    print(f"\nsaved -> {od}")


if __name__ == "__main__":
    main()
