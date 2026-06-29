"""``load_aad`` / ``get_dataloaders`` — the public entry points.

A :class:`MaestroDataset` is a map-style dataset over ``(subject, trial,
segment)`` windows. Each item is a dict of cross-modal-aligned arrays plus
labels. Trials are aligned lazily (LRU-cached) on first access, so construction
is cheap even when streaming from HuggingFace.

* ``load_aad(...)`` → one dataset, or — when ``split=...`` is given — a
  ``(train, test)`` pair that is guaranteed trial-disjoint.
* ``get_dataloaders(setting=..., fold=...)`` → ``(train_loader, test_loader)``
  PyTorch ``DataLoader``s for the LOSO or intra-subject protocol.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .align import AlignedTrial, load_aligned_trial
from .paths import DEFAULT_REPO_ID, DatasetPaths
from .segments import make_segments
from . import splits as _splits
from . import video as _video

ALL_MODALITIES = ("eeg", "gaze", "imu", "audio", "video")
_FILE_RE = re.compile(r"data/eeg/subject=(S\d+)/trial=([a-z]+_\d+)\.parquet")


# --------------------------------------------------------------------------- #
# Discovery / selection
# --------------------------------------------------------------------------- #
def _discover(paths: DatasetPaths) -> dict[str, list[str]]:
    """Map subject_id -> sorted list of trial_ids that exist on disk/hub."""
    out: dict[str, list[str]] = {}
    if paths.local_path is not None:
        root = paths.local_path / "data" / "eeg"
        for sdir in sorted(root.glob("subject=*")):
            sid = sdir.name.split("=", 1)[1]
            out[sid] = sorted(p.name.split("=", 1)[1][:-8]
                              for p in sdir.glob("trial=*.parquet"))
    else:
        for f in paths.list_repo_files():
            m = _FILE_RE.match(f)
            if m:
                out.setdefault(m.group(1), []).append(m.group(2))
        for sid in out:
            out[sid] = sorted(out[sid])
    return out


def _norm_subjects(spec, available: list[str]) -> list[str]:
    if spec == "all" or spec is None:
        return sorted(available)
    if isinstance(spec, (int, str)):
        spec = [spec]
    out = []
    for s in spec:
        sid = s if isinstance(s, str) and s.startswith("S") else f"S{int(s):02d}"
        out.append(sid)
    keep = sorted(set(out) & set(available))
    return keep or sorted(out)


def _kind_of(tid: str) -> str:
    return "main" if tid.startswith("eval") else "training"


def _filter_trials(trials: list[str], spec) -> list[str]:
    if spec == "all":
        return trials
    if spec == "main":
        return [t for t in trials if t.startswith("eval")]
    if spec == "training":
        return [t for t in trials if t.startswith("training")]
    if isinstance(spec, (int, str)):
        spec = [spec]
    wanted = set()
    for s in spec:
        wanted.add(f"eval_{s:03d}" if isinstance(s, int) else s)
    return [t for t in trials if t in wanted]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MaestroDataset:
    """Map-style dataset over aligned ``(subject, trial, segment)`` windows."""

    def __init__(self, paths, units, trials_meta, *, modalities, segment_length,
                 overlap, drop_last, preprocess, target_sfreq, audio_feature,
                 eeg_cfg, normalize, video_frames, video_max_frames,
                 return_format, trial_cache):
        self.paths = paths
        self.trials_meta = trials_meta
        self.modalities = list(modalities)
        self.segment_length = segment_length
        self.overlap = overlap
        self.drop_last = drop_last
        self.preprocess = preprocess
        self.target_sfreq = target_sfreq
        self.audio_feature = audio_feature
        self.eeg_cfg = eeg_cfg
        self.normalize = normalize
        self.video_frames = video_frames
        self.video_max_frames = video_max_frames
        self.return_format = return_format
        self._cache: "OrderedDict[tuple, AlignedTrial]" = OrderedDict()
        self._cache_max = max(1, trial_cache)
        self.units = list(units)                       # [(sid, tid), ...]
        self.index = self._build_index()               # [(unit_i, Segment), ...]

    def _build_index(self):
        idx = []
        for ui, (sid, tid) in enumerate(self.units):
            try:
                tj = json.loads(Path(self.paths.timing(sid, tid)).read_text())
                dur = float(tj["align"]["overlap_sec"])
            except Exception:
                continue
            for seg in make_segments(dur, self.segment_length, self.overlap,
                                     self.drop_last):
                idx.append((ui, seg))
        return idx

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _aligned(self, sid, tid) -> AlignedTrial | None:
        key = (sid, tid)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        at = load_aligned_trial(
            self.paths, sid, tid, modalities=self.modalities,
            trials_meta=self.trials_meta, preprocess=self.preprocess,
            target_sfreq=self.target_sfreq, audio_feature=self.audio_feature,
            eeg_cfg=self.eeg_cfg)
        self._cache[key] = at
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return at

    def __getitem__(self, i):
        ui, seg = self.index[i]
        sid, tid = self.units[ui]
        at = self._aligned(sid, tid)
        out: dict[str, Any] = {
            "subject_id": sid, "trial_id": tid, "kind": _kind_of(tid),
            "segment_idx": seg.idx,
            "t_start_sec": seg.t_start_sec, "t_end_sec": seg.t_end_sec,
        }
        if at is None:
            return _format(out, self.return_format)
        out.update({k: at.labels.get(k) for k in at.labels})
        for name, msig in at.signals.items():
            arr = np.ascontiguousarray(msig.slice_seconds(
                seg.t_start_sec, seg.t_end_sec, at.anchor_unix))
            if self.normalize == "zscore" and name in ("eeg", "gaze", "imu", "audio"):
                arr = _zscore(arr)
            out[name] = arr
            if msig.columns is not None:
                out[f"{name}_columns"] = msig.columns
            out[f"{name}_sfreq"] = msig.rate
        if "video" in self.modalities and at.video_path:
            self._attach_video(out, at, seg)
        return _format(out, self.return_format)

    def _attach_video(self, out, at, seg):
        rs = at.meta.get("recording_start_unix", at.anchor_unix)
        f0, f1 = _video.frame_index_range(
            rs, at.video_fps or 0.0,
            at.anchor_unix + seg.t_start_sec, at.anchor_unix + seg.t_end_sec)
        out["video_path"] = at.video_path
        out["video_fps"] = at.video_fps
        out["video_frame_range"] = (f0, f1)
        if self.video_frames:
            step = 1
            if self.video_max_frames and (f1 - f0) > self.video_max_frames:
                step = int(np.ceil((f1 - f0) / self.video_max_frames))
            out["video"] = _video.read_frames(
                at.video_path, f0, f1, step=step, max_frames=self.video_max_frames)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _format(out: dict, fmt: str) -> dict:
    if fmt in ("numpy", "dict"):
        return out
    if fmt == "torch":
        import torch
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray) and v.dtype != object:
                out[k] = torch.from_numpy(np.ascontiguousarray(v))
        return out
    raise ValueError(f"unknown return_format: {fmt!r}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_aad(
    *,
    subjects: Any = "all",
    trials: Any = "main",
    modalities: Any = ("eeg",),
    segment_length: float | None = 5.0,
    overlap: float = 0.0,
    drop_last: bool = True,
    preprocess: bool | dict = False,
    target_sfreq: float | None = None,
    audio_feature: str = "auto",
    eeg_cfg: dict | None = None,
    normalize: str | None = None,
    video_frames: bool = False,
    video_max_frames: int | None = None,
    return_format: str = "numpy",
    split: dict | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    local_path: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    trial_cache: int = 8,
):
    """Build a dataset of perfectly aligned segments.

    Parameters
    ----------
    modalities : 'all' or any subset of {eeg, gaze, imu, audio, video}.
    segment_length, overlap : window length (s) and fractional overlap [0,1).
        ``segment_length=None`` → one segment spanning the whole aligned trial.
    preprocess : ``False`` ships aligned near-raw signals; ``True`` applies the
        decoder's EEG pipeline (notch + band-pass + robust reference + bad-channel
        interpolation) and resamples every stream onto a common grid
        (``target_sfreq`` default 64 Hz). A dict overrides the EEG pipeline keys.
    target_sfreq : common rate to resample all streams to. ``None`` keeps native
        rates (each stream is still clipped to the same aligned window).
    audio_feature : 'auto' | 'waveform' | 'mel' (28-band) | 'envelope'.
    normalize : None | 'zscore' (per-segment, per-channel over time).
    split : ``None`` → single dataset. Otherwise e.g.
        ``{"setting": "loso", "fold": 3}`` or
        ``{"setting": "intra", "fold": 0, "n_folds": 5, "scheme": "chrono"}``
        → returns ``(train_ds, test_ds)``, guaranteed trial-disjoint.

    Returns
    -------
    MaestroDataset, or ``(train, test)`` when ``split`` is given.
    """
    paths = DatasetPaths(
        local_path=Path(local_path) if local_path else None,
        repo_id=repo_id, revision=revision,
        cache_dir=Path(cache_dir) if cache_dir else None)

    trials_meta = pd.read_csv(paths.metadata("trials.csv"))
    available = _discover(paths)
    sel_subjects = _norm_subjects(subjects, list(available))
    mods = list(ALL_MODALITIES) if modalities == "all" else (
        [modalities] if isinstance(modalities, str) else list(modalities))
    for m in mods:
        if m not in ALL_MODALITIES:
            raise ValueError(f"unknown modality {m!r}")

    trials_by_subject = {
        s: _filter_trials(available.get(s, []), trials) for s in sel_subjects}

    common = dict(
        modalities=mods, segment_length=segment_length, overlap=overlap,
        drop_last=drop_last, preprocess=preprocess, target_sfreq=target_sfreq,
        audio_feature=audio_feature, eeg_cfg=eeg_cfg, normalize=normalize,
        video_frames=video_frames, video_max_frames=video_max_frames,
        return_format=return_format, trial_cache=trial_cache)

    if split is None:
        units = [(s, t) for s in sel_subjects for t in trials_by_subject[s]]
        return MaestroDataset(paths, units, trials_meta, **common)

    setting = split.get("setting", "loso")
    fold = int(split.get("fold", 0))
    train_u, test_u = _splits.make_split(
        setting, fold, subjects=sel_subjects, trials_by_subject=trials_by_subject,
        n_folds=int(split.get("n_folds", 5)), scheme=split.get("scheme", "chrono"),
        seed=int(split.get("seed", 0)))
    _splits.assert_trial_disjoint(train_u, test_u)
    train = MaestroDataset(paths, train_u, trials_meta, **common)
    test = MaestroDataset(paths, test_u, trials_meta, **common)
    return train, test


def get_dataloaders(
    *,
    setting: str = "loso",
    fold: int = 0,
    n_folds: int = 5,
    scheme: str = "chrono",
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    collate_fn=None,
    **load_kwargs,
):
    """Return ``(train_loader, test_loader)`` for the requested protocol.

    ``setting='loso'`` → ``fold`` in 0..15 (held-out subject). ``setting='intra'``
    → ``fold`` in 0..n_folds-1 (within-subject, trial-disjoint). Extra keyword
    args are forwarded to :func:`load_aad` (modalities, segment_length,
    preprocess, target_sfreq, normalize, ...).
    """
    from torch.utils.data import DataLoader

    train_ds, test_ds = load_aad(
        split={"setting": setting, "fold": fold, "n_folds": n_folds,
               "scheme": scheme},
        return_format=load_kwargs.pop("return_format", "torch"),
        **load_kwargs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                          num_workers=num_workers, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_fn)
    return train_dl, test_dl
