"""The constant data<->model interface.

The runner and every model only ever see two things:
  * ``AADView``  -- a materialisable set of decision windows that exposes BOTH
    a numpy view (``as_numpy``) for classical decoders and a torch
    ``DataLoader`` (``as_torch_loader``) for neural decoders. A model picks the
    representation it wants; the data layer is free to change how windows are
    produced/cached underneath without touching any model.
  * ``AbstractDataModule`` -- yields protocol splits (within-subject CV / LOSO)
    as ``(train_view, test_view)`` pairs.

A single window sample is a dict with fixed keys/shapes:
    eeg       (C, W)            float32
    cand_env  (6, n_bands, W)   float32   the 6 candidate speaker envelopes
    cand_mask (6,)              bool       attendable speakers (1..4 -> True)
    attended  ()                int64      0..3
    gaze      (8,)  imu (12,)  video (16,) float32   per-window context features
    present   (3,)             float32    [gaze, imu, video] real-vs-padding
    subject   ()               int64      1..16
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from .windows import (GAZE_DIM, IMU_DIM, N_SPEAKERS, VIDEO_DIM, TrialRecord,
                      WindowSpec, gaze_window_features, gaze_traj_features,
                      imu_window_features)

CAND_MASK = np.array([True, True, True, True, False, False])  # speakers 1..4 attendable


def _znorm(x: np.ndarray, axis: int) -> np.ndarray:
    """Zero-mean / unit-std along ``axis`` (per channel/band over time)."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _shifted_candidates(rec: TrialRecord, s: int, spec: WindowSpec):
    """Build (n_cand, n_bands, W) candidates = the attended speaker's aligned
    envelope + n_neg time-shifted spoilers of the SAME speaker. Returns
    (candidates, target_index). All candidates are the same talker, so audio
    cannot reveal the match -- only EEG-envelope temporal alignment can.
    """
    W = spec.win_len
    att = rec.attended - 1                              # attended speaker 0..3
    full = rec.env[att]                                 # (n_bands, T) attended envelope
    T = full.shape[1]
    matched = full[:, s:s + W]
    # deterministic per-window RNG so eval is stable
    seed = (int(rec.subject) * 1_000_003 + int(rec.trial_k) * 9_973 + s) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    min_sep = int(round(spec.min_shift_s * spec.sr))
    mode = getattr(spec, "spoiler_mode", "grid")
    grid = list(range(0, T - W + 1, max(1, W // 2)))
    pool = [o for o in grid if abs(o - s) >= min_sep]
    spoilers = []
    for _ in range(spec.n_neg):
        if mode == "circular":                         # circular copy (marginal-preserving)
            o = int(rng.integers(min_sep, max(min_sep + 1, T - min_sep)))
            spoilers.append(np.roll(full, o, axis=1)[:, s:s + W])
        elif pool:
            o = int(rng.choice(pool))
            spoilers.append(full[:, o:o + W])
        else:                                          # degenerate short trial -> roll
            spoilers.append(np.roll(matched, W // 2, axis=1))
    cands = np.stack([matched] + spoilers, 0)          # idx 0 = matched
    perm = rng.permutation(len(cands))
    cands = cands[perm]
    target = int(np.where(perm == 0)[0][0])
    return np.ascontiguousarray(cands, np.float32), target


def _match_candidates(rec: TrialRecord, s: int, spec: WindowSpec):
    """Build PERMUTED real-envelope candidates for the match-mismatch task.

    Candidates = the band-envelopes of the 4 attendable, simultaneously-playing
    talkers (speakers 1..4) at this window, in a per-window RANDOM order. The label
    is the permuted index of the attended talker. Permuting per window destroys any
    fixed candidate-slot <-> spatial-direction mapping, so the decoder cannot solve
    the task by reading attended DIRECTION from the EEG (alpha lateralisation) -- it
    must match the EEG to the attended talker's envelope CONTENT. This is what makes
    the EEG-alone number an honest measure of cortical envelope tracking.
    """
    W = spec.win_len
    att = rec.attended - 1                              # attended talker 0..3
    seed = (int(rec.subject) * 1_000_003 + int(rec.trial_k) * 9_973 + s) & 0x7FFFFFFF
    perm = np.random.default_rng(seed).permutation(4)  # deterministic -> stable eval
    target = int(np.where(perm == att)[0][0])
    cands = np.ascontiguousarray(rec.env[:4, :, s:s + W][perm], np.float32)
    # cand_pos[j] = PHYSICAL speaker index (0..3) sitting in candidate slot j.
    # Content branches stay position-blind (permuted candidates), but a directional
    # branch can decode physical position from EEG-alpha then re-index into slot order.
    extra = {"cand_pos": np.ascontiguousarray(perm, np.int64)}
    if rec.w2v is not None:                            # rich audio-feature candidates (same perm)
        extra["cand_w2v"] = np.ascontiguousarray(rec.w2v[:4, :, s:s + W][perm], np.float32)
        extra["cand_sem"] = np.ascontiguousarray(rec.sem[:4, :, s:s + W][perm], np.float32)
    return cands, target, np.ones(4, bool), extra


def _materialize_window(rec: TrialRecord, s: int, spec: WindowSpec) -> dict:
    W = spec.win_len
    sl = slice(s, s + W)
    eeg = np.ascontiguousarray(rec.eeg[:, sl], np.float32)
    if getattr(spec, "norm_eeg", True):
        eeg = _znorm(eeg, axis=1)                 # per-channel over time

    task = getattr(spec, "mm_task", "speaker")
    extra = {}
    if task == "shifted":
        cand, target = _shifted_candidates(rec, s, spec)
        cand_mask = np.ones(cand.shape[0], bool)
    elif task == "match":                          # permuted real talkers 1..4
        cand, target, cand_mask, extra = _match_candidates(rec, s, spec)
    else:                                          # "speaker": 6 fixed speakers
        cand = np.ascontiguousarray(rec.env[:, :, sl], np.float32)
        cand_mask = CAND_MASK.copy()
        target = rec.attended - 1
    if getattr(spec, "norm_cand", True):
        cand = _znorm(cand, axis=cand.ndim - 1)    # per-(cand,band) over time
        if "cand_w2v" in extra:                    # z-norm auditory per-(cand,dim); keep sem raw
            extra["cand_w2v"] = _znorm(extra["cand_w2v"], axis=extra["cand_w2v"].ndim - 1)

    tl = int(getattr(spec, "gaze_traj_len", 16) or 0)
    gaze = gaze_window_features(rec.gaze[sl]) if rec.present_gaze else np.zeros(GAZE_DIM, np.float32)
    gaze_traj = (gaze_traj_features(rec.gaze[sl], tl) if rec.present_gaze
                 else np.zeros(2 * tl, np.float32))
    imu = imu_window_features(rec.imu[sl]) if rec.present_imu else np.zeros(IMU_DIM, np.float32)
    video = np.zeros(VIDEO_DIM, np.float32)
    present = np.array(
        [float(rec.present_gaze), float(rec.present_imu), float(rec.present_video)], np.float32
    )
    out = dict(
        eeg=eeg, cand_env=cand, cand_mask=cand_mask,
        attended=np.int64(target), gaze=gaze, gaze_traj=gaze_traj, imu=imu, video=video,
        present=present, subject=np.int64(rec.subject),
    )
    out.update(extra)                              # cand_w2v (4,Dw,W), cand_sem (4,3,W) when present
    return out


@dataclass
class WindowIndex:
    rec_ptr: int
    start: int


class AADView:
    """A set of decision windows over a list of TrialRecords."""

    def __init__(self, records: List[TrialRecord], indices: List[WindowIndex],
                 spec: WindowSpec, name: str = ""):
        self.records = records
        self.indices = indices
        self.spec = spec
        self.name = name

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def task_type(self) -> str:
        return getattr(self.spec, "mm_task", "speaker")

    @property
    def n_candidates(self) -> int:
        return (self.spec.n_neg + 1) if self.task_type == "shifted" else 4

    def materialize(self, i: int) -> dict:
        wi = self.indices[i]
        return _materialize_window(self.records[wi.rec_ptr], wi.start, self.spec)

    # ---- classical interface -------------------------------------------------
    def as_numpy(self) -> dict:
        """Stack every window into dense arrays (good for small/medium views)."""
        samples = [self.materialize(i) for i in range(len(self))]
        out = {}
        keys = ["eeg", "cand_env", "gaze", "gaze_traj", "imu", "video", "present", "cand_mask"]
        if "cand_pos" in samples[0]:
            keys += ["cand_pos"]
        if "cand_w2v" in samples[0]:
            keys += ["cand_w2v", "cand_sem"]
        for key in keys:
            out[key] = np.stack([s[key] for s in samples], 0)
        out["attended"] = np.array([int(s["attended"]) for s in samples], np.int64)
        out["subject"] = np.array([int(s["subject"]) for s in samples], np.int64)
        return out

    # ---- neural interface ----------------------------------------------------
    def as_torch_dataset(self):
        import torch

        view = self

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(view)

            def __getitem__(self, i):
                s = view.materialize(i)
                return {k: torch.as_tensor(v) for k, v in s.items()}

        return _DS()

    def as_torch_loader(self, batch_size: int = 64, shuffle: bool = False,
                        num_workers: int = 0):
        import torch

        return torch.utils.data.DataLoader(
            self.as_torch_dataset(), batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=False, pin_memory=False,
        )

    @property
    def subjects(self) -> List[int]:
        return sorted({r.subject for r in self.records})


@dataclass
class SplitDesc:
    name: str                # human-readable, e.g. "within_s3_fold2" / "loso_test_s3"
    protocol: str            # "within" | "loso"
    test_subject: int | None
    fold: int | None


class AbstractDataModule(ABC):
    """Constant interface the runner depends on."""

    spec: WindowSpec

    @abstractmethod
    def prepare(self) -> None:
        """Build/load caches so that ``splits`` can run with no heavy I/O."""

    @abstractmethod
    def splits(self, protocol: str) -> Iterator[Tuple[SplitDesc, AADView, AADView]]:
        """Yield (descriptor, train_view, test_view) for a protocol."""

    @abstractmethod
    def feature_dims(self) -> dict:
        ...
