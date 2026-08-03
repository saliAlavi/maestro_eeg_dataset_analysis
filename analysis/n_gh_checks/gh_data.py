"""Leakage-safe data adapter feeding the github MAESTRO models from our own
5 s / 0.5-overlap, per-device-aligned, loudness-matched dataloader (src.data).

What this fixes vs. the github ``dataloader.py``:
  * 5 s windows @ 0.5 overlap (github: 30 s, no overlap) — from WindowSpec.
  * trial-level splits so overlapping windows never straddle train/test.
  * per-device perfect audio alignment + table-power loudness equalisation
    (baked into the cached envelopes) — kills the +3..18 dB attended-loudness
    confound that let the github 4-class decode from audio energy alone.
  * a real held-out test set: early stopping / checkpoint selection happen on an
    inner-val split carved from *train*, never on the reported test split.

Envelope note: the github audio encoder takes a single broadband envelope per
speaker. Our cache stores 28 loudness-matched gammatone bands; we collapse them
to one broadband channel (mean over bands) so the github 1-channel audio encoder
is reproduced exactly while still using the loudness-matched, aligned envelope.

EEG band: we load the full-band EEG cache (``elp0`` = no 10 Hz EEG lowpass) so
alpha/beta lateralisation survives for the hemisphere/eccentricity models, while
the speaker envelopes remain 10 Hz low-passed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

# --- make `src` importable from analysis/n_gh_checks/ -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.windows import (WindowSpec, TrialRecord, cache_path,  # noqa: E402
                              load_subject_records)

# --------------------------------------------------------------------------- #
# Fixed spec + cache location (matches the pre-built win5 full-band EEG cache)
# --------------------------------------------------------------------------- #
CACHE_DIR = Path("/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_spec")
SPEC = WindowSpec(win_s=5.0, sr=64.0, n_bands=28, overlap=0.5,
                  audio_norm="table_power", lp_hz=10.0, eeg_lp_hz=0.0,
                  perfect_align=True)   # tag -> win5_sr64_b28_pwT_lp10_elp0_pa2

SUBJECTS = list(range(1, 17))
MODALITIES = ("eeg", "gaze", "imu")     # video is not materialised by our loader

# github label maps (0-indexed attended speaker -> binary class)
HEMI_MAP = {0: 0, 1: 0, 2: 1, 3: 1}          # {S1,S2}=left(0), {S3,S4}=right(1)
ECC_MAP = {0: 1, 1: 0, 2: 0, 3: 1}           # {S2,S3}=inner(0), {S1,S4}=outer(1)

TASKS = ("hemisphere", "eccentricity", "speaker4", "reconstruction")
CLASSIF_TASKS = ("hemisphere", "eccentricity", "speaker4")
N_SPEAKERS = {"hemisphere": 2, "eccentricity": 2, "speaker4": 4}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_subjects(subjects: List[int]) -> dict:
    out = {}
    for s in subjects:
        p = cache_path(CACHE_DIR, s, SPEC, "main")
        if not p.exists():
            raise FileNotFoundError(f"missing cache for S{s}: {p}")
        out[s] = load_subject_records(p)
    return out


def _znorm(x: np.ndarray, axis: int) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _broadband(rec: TrialRecord) -> np.ndarray:
    """(6, T) broadband envelope per speaker = mean over gammatone bands."""
    return rec.env.mean(axis=1).astype(np.float32)     # (6, T)


# --------------------------------------------------------------------------- #
# Window materialisation
# --------------------------------------------------------------------------- #
def _win_idx(records: List[TrialRecord], win_len: int, hop: int):
    """(rec_ptr, start) window index list for an explicit window/hop."""
    idx = []
    for p, r in enumerate(records):
        T = r.eeg.shape[1]
        for s in range(0, T - win_len + 1, hop):
            idx.append((p, s))
    return idx


def _window_indices(records: List[TrialRecord], spec: WindowSpec):
    return _win_idx(records, spec.win_len, spec.hop_len)


def whole_trial_len(records: List[TrialRecord]) -> int:
    """Length of one github-style whole-trial (~30 s) window: the shortest trial
    across the loaded records, so a single fixed-length window fits every trial
    (github used 30 s = 1920 samples, ~1 window/trial; our trials are ~1908)."""
    return int(min(r.eeg.shape[1] for r in records))


def materialize_classif(records: List[TrialRecord], task: str,
                        modalities, spec: WindowSpec = SPEC,
                        win_len: int | None = None, hop: int | None = None) -> dict:
    """Stack all windows of ``records`` for a classification task.

    Returns dict with per-active-modality arrays (N,W,C), an audio list of
    ``n_speakers`` arrays (N,W,1), and integer labels (N,). Speaker order for the
    4-class task is permuted per-window with a deterministic seed (subject,
    trial, start) so the attended slot varies window-to-window — no fixed
    slot->direction shortcut, and eval is reproducible.

    ``win_len``/``hop`` override the spec (used for the github 30 s whole-trial
    method vs. our 5 s / 0.5-overlap default).
    """
    W = win_len or spec.win_len
    idx = _win_idx(records, W, hop or spec.hop_len)
    bb = [_broadband(r) for r in records]          # per-record (6, T)
    n_spk = N_SPEAKERS[task]
    use = [m for m in ("eeg", "gaze", "imu") if m in modalities]

    mod_arrs = {m: [] for m in use}
    audio = [[] for _ in range(n_spk)]
    labels = []
    for (p, s) in idx:
        r = records[p]
        sl = slice(s, s + W)
        if "eeg" in use:
            mod_arrs["eeg"].append(_znorm(r.eeg[:, sl].T, axis=0))          # (W,32)
        if "gaze" in use:
            g = r.gaze[sl] if r.present_gaze else np.zeros((W, 3), np.float32)
            mod_arrs["gaze"].append(_znorm(g, axis=0))                      # (W,3)
        if "imu" in use:
            im = r.imu[sl] if r.present_imu else np.zeros((W, 6), np.float32)
            mod_arrs["imu"].append(_znorm(im, axis=0))                      # (W,6)

        e = bb[p][:, sl]                                                    # (6,W)
        att = r.attended - 1
        if task == "speaker4":
            seed = (int(r.subject) * 1_000_003 + int(r.trial_k) * 9_973 + s) & 0x7FFFFFFF
            perm = np.random.default_rng(seed).permutation(4)
            for j, spk in enumerate(perm):
                audio[j].append(_znorm(e[spk], axis=0)[:, None])           # (W,1)
            labels.append(int(np.where(perm == att)[0][0]))
        elif task == "hemisphere":
            left = 0.5 * (e[0] + e[1]); right = 0.5 * (e[2] + e[3])
            audio[0].append(_znorm(left, axis=0)[:, None])
            audio[1].append(_znorm(right, axis=0)[:, None])
            labels.append(HEMI_MAP[att])
        elif task == "eccentricity":
            inner = 0.5 * (e[1] + e[2]); outer = 0.5 * (e[0] + e[3])
            audio[0].append(_znorm(inner, axis=0)[:, None])
            audio[1].append(_znorm(outer, axis=0)[:, None])
            labels.append(ECC_MAP[att])
        else:
            raise ValueError(task)

    out = {"labels": np.asarray(labels, np.int64),
           "audio": [np.stack(a).astype(np.float32) for a in audio],
           "n_speakers": n_spk}
    for m in use:
        out[m] = np.stack(mod_arrs[m]).astype(np.float32)
    return out


def materialize_recon(records: List[TrialRecord], modalities,
                      spec: WindowSpec = SPEC,
                      win_len: int | None = None, hop: int | None = None) -> dict:
    """Concatenated-channel input X (N,W,C) in fixed order eeg,gaze,imu and the
    attended broadband envelope target y (N,W,1). ``win_len``/``hop`` override
    the spec (github 30 s whole-trial vs. our 5 s / 0.5-overlap)."""
    W = win_len or spec.win_len
    idx = _win_idx(records, W, hop or spec.hop_len)
    bb = [_broadband(r) for r in records]
    use = [m for m in ("eeg", "gaze", "imu") if m in modalities]

    X, Y = [], []
    for (p, s) in idx:
        r = records[p]
        sl = slice(s, s + W)
        parts = []
        if "eeg" in use:
            parts.append(_znorm(r.eeg[:, sl].T, axis=0))
        if "gaze" in use:
            g = r.gaze[sl] if r.present_gaze else np.zeros((W, 3), np.float32)
            parts.append(_znorm(g, axis=0))
        if "imu" in use:
            im = r.imu[sl] if r.present_imu else np.zeros((W, 6), np.float32)
            parts.append(_znorm(im, axis=0))
        X.append(np.concatenate(parts, axis=1))                # (W, C)
        att = r.attended - 1
        Y.append(_znorm(bb[p][att, sl], axis=0)[:, None])      # (W,1)
    return {"X": np.stack(X).astype(np.float32), "y": np.stack(Y).astype(np.float32)}


# --------------------------------------------------------------------------- #
# Leakage-safe split protocols (trial-level, with inner-val for early stopping)
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    name: str
    protocol: str            # "within" | "loso"
    test_subject: int
    fold: int | None
    train: List[TrialRecord]
    val: List[TrialRecord]
    test: List[TrialRecord]


def _chrono_order(recs: List[TrialRecord]) -> np.ndarray:
    return np.argsort([r.trial_k for r in recs])


def _carve_val_chrono(train_recs: List[TrialRecord], val_frac: float):
    """Inner-val = chronological tail of train (no look-ahead)."""
    order = _chrono_order(train_recs)
    n_val = max(1, int(round(val_frac * len(order))))
    val_ids = set(order[-n_val:].tolist())
    tr = [train_recs[i] for i in order if i not in val_ids]
    va = [train_recs[i] for i in order if i in val_ids]
    return tr, va


def _carve_val_stratified(train_recs: List[TrialRecord], val_frac: float, seed: int):
    """Inner-val = seeded, attended-stratified fraction of train trials (LOSO)."""
    rng = np.random.default_rng(seed)
    by_att = {}
    for i, r in enumerate(train_recs):
        by_att.setdefault(r.attended, []).append(i)
    val_ids = set()
    for att, ids in by_att.items():
        ids = list(ids); rng.shuffle(ids)
        k = max(1, int(round(val_frac * len(ids))))
        val_ids.update(ids[:k])
    tr = [r for i, r in enumerate(train_recs) if i not in val_ids]
    va = [r for i, r in enumerate(train_recs) if i in val_ids]
    return tr, va


def within_splits(by_subject: dict, subject: int, n_folds: int = 5,
                  val_frac: float = 0.2) -> Iterator[Split]:
    """Per-subject chrono-forward CV: fold f trains on time-blocks [0..f],
    tests on block f+1 (train strictly precedes test)."""
    recs = by_subject[subject]
    if len(recs) < n_folds + 1:
        return
    order = _chrono_order(recs)
    blocks = np.array_split(order, n_folds + 1)
    for f in range(n_folds):
        tr_idx = np.concatenate(blocks[: f + 1])
        te_idx = blocks[f + 1]
        train_recs = [recs[i] for i in tr_idx]
        test_recs = [recs[i] for i in te_idx]
        tr, va = _carve_val_chrono(train_recs, val_frac)
        if not tr or not va or not test_recs:
            continue
        yield Split(f"within_s{subject}_fold{f}", "within", subject, f, tr, va, test_recs)


def loso_split(by_subject: dict, test_subject: int, subjects=SUBJECTS,
               val_frac: float = 0.15, seed: int = 42) -> Split:
    """Leave-one-subject-out: train on the other 15 subjects, test on the
    held-out one. Inner-val is a seeded stratified fraction of the pooled train."""
    train_recs = []
    for s in subjects:
        if s != test_subject:
            train_recs.extend(by_subject[s])
    tr, va = _carve_val_stratified(train_recs, val_frac, seed + test_subject)
    return Split(f"loso_test_s{test_subject}", "loso", test_subject, None,
                 tr, va, by_subject[test_subject])


# --------------------------------------------------------------------------- #
# GITHUB-METHOD split protocols (the leaky repo methodology, for A/B comparison)
#   * pooled StratifiedKFold over per-(subject,trial) instances  -> subjects are
#     NOT held out (same subjects in train & test = the pooled leak)
#   * val == test (no inner-val): the reported number is max-over-epochs on the
#     very fold used for early stopping / checkpoint selection.
# Reproduces train_pooled / train_hemisphere / train_eccentricity / train_loso_hot.
# --------------------------------------------------------------------------- #
def task_label(rec: TrialRecord, task: str) -> int:
    att = rec.attended - 1
    if task == "hemisphere":
        return HEMI_MAP[att]
    if task == "eccentricity":
        return ECC_MAP[att]
    return att                                   # speaker4: 0..3


def github_pooled_splits(by_subject: dict, task: str, subjects=SUBJECTS,
                         n_folds: int = 5, seed: int = 42) -> Iterator[Split]:
    """github pooled StratifiedKFold(shuffle, random_state=42), val==test."""
    from sklearn.model_selection import StratifiedKFold
    recs, labs = [], []
    for s in subjects:
        for r in by_subject[s]:
            recs.append(r); labs.append(task_label(r, task))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(recs)), labs)):
        train = [recs[i] for i in tr]; test = [recs[i] for i in te]
        yield Split(f"ghpool_{task}_fold{f}", "pooled", None, f, train, test, test)


def github_loso_splits(by_subject: dict, task: str, subjects=SUBJECTS) -> Iterator[Split]:
    """github train_loso_hot-style: subject-disjoint, but val==test on the held-out
    subject with best-epoch selection (no inner val)."""
    for test_s in subjects:
        train = []
        for s in subjects:
            if s != test_s:
                train.extend(by_subject[s])
        test = by_subject[test_s]
        yield Split(f"ghloso_{task}_test_s{test_s}", "loso", test_s, None, train, test, test)
