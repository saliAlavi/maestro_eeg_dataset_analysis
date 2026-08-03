"""Leakage-safe data for the backward-decoder AAD baseline.

This layer reuses the SAME properly-aligned dataset the method paper
(docs/draft/attended_speaker_decoding.pdf) is built on, so we never re-derive the
cross-modal alignment ourselves. Every stream (EEG, per-talker audio, gaze, IMU)
has a *different* recording lag; those lags are resolved once, at cache-build time,
by the project's data-processing module and frozen into the `*_pa2_af64.npz` cache
(`pa2` = proper-alignment v2, `af64` = audio features at 64 Hz). We read that cache
directly and do no lag correction of our own.

Four-way task (chance = 0.25 at EVERY window). The four candidates are the four
REAL co-present attendable talkers (loudspeakers 1-4), z-scored per band. Because a
window always contains exactly those four real streams, the number of choices is 4
regardless of window length -> theoretical chance is 0.25 at 5 s and at the whole
~30 s trial alike. (The earlier time-shift-spoiler construction is what made the
null drift with window length; real co-present candidates fix that by design.)

Two confounds are defused the way the method paper defuses them -- architecturally,
not by spoilers:
  * loudness  -> the decision is a SCALE-FREE correlation of the EEG reconstruction
    with each candidate (backward.mm_scores); an uninformative reconstruction
    correlates ~0 with every talker, so amplitude cannot help.
  * clip identity / the deterministic attended schedule (attended==((k-1)%4)+1,
    same stimuli replayed to every subject) -> the four candidate streams are
    PERMUTED per (subject,trial) with one running RNG, and the label becomes the
    attended talker's permuted SLOT. The same trial_k lands on different slots for
    different subjects, so trial_k->slot cannot be memorised; the EEG-shuffle null
    then verifies chance == 0.25.

Splits (train/val trial-disjoint in BOTH protocols, per project requirement):
  * within: per-subject StratifiedKFold(5) over that subject's trials; the inner
    validation set is a further trial-disjoint, attended-stratified hold-out of the
    training trials. Test fold, val, and train share no trial.
  * loso:   test = the held-out subject's trials; train/val come from the other 15
    subjects and are split by trial_k CONTENT (a val trial_k never appears in train),
    so the same stimulus never sits in both train and val.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Properly-aligned cache built by the project data-processing module (see module docstring).
CACHE = Path("/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__aad_recon/aad_trials")
SUBJECTS = list(range(1, 17))
SR = 64.0
N_BANDS = 28
N_TALKERS = 4                 # four attendable talkers -> four-way, chance 0.25 at any window
PERM_SEED = 20260619          # per-(subject,trial) candidate permutation (matches the draft)

# DECISION-WINDOW length (s) used for training views. The cache stores FULL-TRIAL
# arrays, so the window is applied at materialise time; WIN_S can change without
# rebuilding the cache. Fully-convolutional decoders trained at WIN_S evaluate at
# any window (run_curve).
WIN_S = 5.0
OVERLAP = 0.5
SPOILER_MODE = "real"         # retained for API compatibility; candidates are 4 real talkers


# --------------------------------------------------------------------------- #
# Lightweight records / views (self-contained; no src.data dependency)
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    subject: int
    trial_k: int
    attended: int             # 0-based PERMUTED slot of the attended talker (the label)
    attended_phys: int        # 0-based physical talker (1..4 -> 0..3), pre-permutation
    eeg: np.ndarray           # (32, T) z-scored per channel over time
    cand: np.ndarray          # (4, 28, T) four REAL talkers, z-scored per band, PERMUTED


@dataclass
class WindowIndex:
    rec_ptr: int
    start: int


@dataclass
class SplitDesc:
    name: str
    kind: str                 # "within" | "loso"
    test_subject: Optional[int]
    fold: Optional[int]


class View:
    """A set of decision windows over a list of records; materialises to numpy."""

    def __init__(self, records: List[Record], indices: List[WindowIndex], win: int, name: str):
        self.records = records
        self.indices = indices
        self.win = win
        self.name = name

    def __len__(self):
        return len(self.indices)

    def as_numpy(self) -> dict:
        W, n = self.win, len(self.indices)
        C = self.records[0].cand.shape[1] if self.records else N_BANDS
        eeg = np.empty((n, 32, W), np.float32)
        cand = np.empty((n, N_TALKERS, C, W), np.float32)
        att = np.empty(n, np.int64)
        for j, wi in enumerate(self.indices):
            r = self.records[wi.rec_ptr]; s = wi.start
            eeg[j] = r.eeg[:, s:s + W]
            cand[j] = r.cand[:, :, s:s + W]
            att[j] = r.attended
        return dict(eeg=eeg, cand_env=cand, attended=att)


# --------------------------------------------------------------------------- #
# Loading + candidate permutation
# --------------------------------------------------------------------------- #
def _zscore(x: np.ndarray, axis: int, eps: float = 1e-6) -> np.ndarray:
    m = x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32)


def _load_subject_raw(s: int) -> List[Record]:
    """Load one subject's trials from the aligned cache; candidates UN-permuted."""
    files = sorted(glob.glob(str(CACHE / f"s{s}_main_*_pa2_af64.npz")))
    if not files:
        return []
    z = np.load(files[0])
    eeg = _zscore(z["eeg"].astype(np.float32), axis=2)          # (N,32,T) per-channel over time
    env = z["env"][:, :N_TALKERS].astype(np.float32)           # (N,4,28,T) four real talkers
    cand = _zscore(env, axis=-1)                                # per (trial,talker,band) over time
    att = z["attended"].astype(int) - 1                        # 0-based physical (1..4 -> 0..3)
    tk = z["trial_k"].astype(int)
    return [Record(subject=s, trial_k=int(tk[i]), attended=int(att[i]),
                   attended_phys=int(att[i]), eeg=eeg[i], cand=cand[i])
            for i in range(len(att))]


class DM:
    """Holds per-subject records with candidates permuted per (subject,trial)."""

    def __init__(self, subjects, perm_seed: int = PERM_SEED):
        self.subjects = list(subjects)
        self.by_subject = {s: _load_subject_raw(s) for s in self.subjects}
        self.by_subject = {s: r for s, r in self.by_subject.items() if r}
        self.subjects = [s for s in self.subjects if s in self.by_subject]
        self._permute(perm_seed)

    def _permute(self, seed: int):
        """One running RNG over all (subject,trial) rows in a fixed order -> the same
        trial_k gets DIFFERENT slot assignments across subjects, breaking the
        deterministic attended schedule (see module docstring)."""
        rng = np.random.default_rng(seed)
        allr = sorted((r for s in self.subjects for r in self.by_subject[s]),
                      key=lambda r: (r.subject, r.trial_k))
        for r in allr:
            p = rng.permutation(N_TALKERS)
            r.cand = r.cand[p]
            r.attended = int(np.flatnonzero(p == r.attended_phys)[0])


def build_dm(subjects) -> DM:
    return DM(subjects)


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def _win_len(win_s: float, T: int) -> int:
    return min(int(round(win_s * SR)), T)


def _windows_for(records: List[Record], W: int, overlap=None) -> List[WindowIndex]:
    if overlap is None:
        overlap = OVERLAP                                  # module global (settable for denser TRAIN windows)
    hop = max(1, int(round(W * (1.0 - overlap))))
    return [WindowIndex(p, s) for p, r in enumerate(records)
            for s in range(0, r.eeg.shape[1] - W + 1, hop)]


def _view(records: List[Record], win_s: float, name: str, overlap=None) -> View:
    if not records:
        return View(records, [], 1, name)
    T = min(r.eeg.shape[1] for r in records)
    W = _win_len(win_s, T)
    return View(records, _windows_for(records, W, overlap), W, name)


def test_view(records: List[Record], win_s: float) -> View:
    """Test view at an arbitrary decision window (fixed 0.5 overlap regardless of the training
    overlap). Candidates are the four real talkers, so chance is 0.25 at every win_s."""
    return _view(records, win_s, f"te{win_s:g}", overlap=0.5)


# --------------------------------------------------------------------------- #
# Trial-disjoint splits
# --------------------------------------------------------------------------- #
def _strat_holdout(y: np.ndarray, frac: float, rng) -> np.ndarray:
    """Boolean mask holding out `frac` of the rows, stratified by label y."""
    mask = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c); rng.shuffle(idx)
        k = max(1, int(round(frac * len(idx))))
        mask[idx[:k]] = True
    return mask


def _intra_folds(recs: List[Record], seed: int, n_folds: int = 5, val_frac: float = 0.15):
    """StratifiedKFold(5) over one subject's trials; inner val is a trial-disjoint,
    attended-stratified hold-out of the training trials -> train/val/test share no trial."""
    y = np.array([r.attended_phys for r in recs])
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(recs)), y)):
        te = [recs[i] for i in te_idx]
        tr_recs = [recs[i] for i in tr_idx]
        rng = np.random.default_rng(seed + 1 + fold)
        vmask = _strat_holdout(y[tr_idx], val_frac, rng)
        tr = [r for r, m in zip(tr_recs, vmask) if not m]
        va = [r for r, m in zip(tr_recs, vmask) if m]
        yield fold, tr, va, te


def _loso(dm: DM, test_subject: int, seed: int, val_frac: float = 0.15):
    """Test = held-out subject's trials. Train/val from the other 15 subjects, split by
    trial_k CONTENT (a val trial_k never appears in train). Stratify the content split
    by the deterministic attended schedule so train and val stay attended-balanced."""
    test_recs = dm.by_subject[test_subject]
    train_subj = [s for s in dm.subjects if s != test_subject]
    tks = sorted({r.trial_k for s in train_subj for r in dm.by_subject[s]})
    strata = np.array([(k - 1) % N_TALKERS for k in tks])       # deterministic attended stratum
    rng = np.random.default_rng(seed)
    vmask = _strat_holdout(strata, val_frac, rng)
    val_c = {k for k, m in zip(tks, vmask) if m}
    train_recs, val_recs = [], []
    for s in train_subj:
        for r in dm.by_subject[s]:
            (val_recs if r.trial_k in val_c else train_recs).append(r)
    desc = SplitDesc(f"loso_test_s{test_subject}", "loso", test_subject, None)
    return desc, train_recs, val_recs, test_recs


def splits(dm: DM, protocol: str, val_frac: float = 0.15, seed: int = 42
           ) -> Iterator[Tuple[SplitDesc, View, View, View]]:
    """Yield (desc, train_view, val_view, test_view). Train/val trial-disjoint in both."""
    if protocol == "within":
        for s in dm.subjects:
            for fold, tr, va, te in _intra_folds(dm.by_subject[s], seed):
                if tr and va and te:
                    desc = SplitDesc(f"within_s{s}_f{fold}", "within", s, fold)
                    yield desc, _view(tr, WIN_S, "tr"), _view(va, WIN_S, "val"), _view(te, WIN_S, "te")
    elif protocol == "loso":
        for s in dm.subjects:
            desc, tr, va, te = _loso(dm, s, seed, val_frac)
            if tr and va and te:
                yield desc, _view(tr, WIN_S, "tr"), _view(va, WIN_S, "val"), _view(te, WIN_S, "te")
    else:
        raise ValueError(protocol)


def loso_curve_split(dm: DM, test_subject: int, seed: int = 42):
    """LOSO split returning train/val VIEWS at WIN_S and the raw held-out TEST RECORDS,
    so the caller can build test views at a curve of decision windows."""
    desc, tr, va, test_recs = _loso(dm, test_subject, seed)
    return desc, _view(tr, WIN_S, "tr"), _view(va, WIN_S, "val"), test_recs


def trial_groups(view: View):
    """Each window index -> (subject, trial_k, attended_slot) for trial-level pooling."""
    return [(int(view.records[wi.rec_ptr].subject),
             int(view.records[wi.rec_ptr].trial_k),
             int(view.records[wi.rec_ptr].attended)) for wi in view.indices]
