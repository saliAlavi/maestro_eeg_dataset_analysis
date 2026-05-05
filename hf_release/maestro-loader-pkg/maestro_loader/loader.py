"""Main entry point: ``load_aad(...)`` returns a configurable Dataset.

The dataset can be backed by:
- a local directory (``local_path=...``), or
- a Hugging Face dataset repo, downloaded lazily via ``huggingface_hub``.

Each item returned is a dict with the requested modalities + labels +
segment metadata.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maestro_loader.badchans import apply_bad_channels
from maestro_loader.normalize import normalize_array
from maestro_loader.segments import Segment, make_segments, slice_signal

DEFAULT_REPO_ID = "aspire-osu/maestro-eeg-dataset"
EEG_SFREQ_DEFAULT = 500.0
ALL_MODALITIES = ("eeg", "gaze", "imu", "audio", "video")


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
@dataclass
class DatasetPaths:
    """Resolves dataset files either from a local directory or HF cache.

    When ``local_path`` is given, files are read directly. Otherwise we use
    ``huggingface_hub`` to lazily download per-file as needed.
    """
    local_path: Path | None = None
    repo_id: str = DEFAULT_REPO_ID
    cache_dir: Path | None = None
    _hf_api: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.local_path is not None:
            self.local_path = Path(self.local_path)
            return
        # Lazy import — only needed when streaming from HF.
        from huggingface_hub import HfApi
        self._hf_api = HfApi()

    def _resolve(self, rel: str) -> Path:
        if self.local_path is not None:
            return self.local_path / rel
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=rel,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        ))

    def metadata(self, name: str) -> Path:
        return self._resolve(f"metadata/{name}")

    def parquet(self, modality: str, subject_id: str, trial_id: str) -> Path:
        return self._resolve(f"data/{modality}/subject={subject_id}/trial={trial_id}.parquet")

    def audio_speaker(self, trial_id: str, filename: str) -> Path:
        return self._resolve(f"media/audio/{trial_id}/{filename}")

    def video(self, subject_id: str, trial_id: str) -> Path:
        return self._resolve(f"media/video/subject={subject_id}/{trial_id}.mp4")

    def timing(self, subject_id: str, trial_id: str) -> Path:
        return self._resolve(f"media/timing/subject={subject_id}/trial={trial_id}.json")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MaestroDataset:
    """Map-style dataset; works with ``torch.utils.data.DataLoader``."""

    def __init__(
        self,
        paths: DatasetPaths,
        items: list[dict[str, Any]],
        modalities: list[str],
        normalize: str | None,
        normalize_scope: str,
        bad_channels: str,
        eeg_channels_full: list[str],
        bad_channels_map: dict[str, list[str]],
        return_format: str,
        per_subject_norm_stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] | None,
        global_norm_stats: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        audio_layout: list[dict],
        audio_manifest: dict[str, dict[str, str]] | None,
    ) -> None:
        self.paths = paths
        self.items = items
        self.modalities = modalities
        self.normalize = normalize
        self.normalize_scope = normalize_scope
        self.bad_channels = bad_channels
        self.eeg_channels_full = eeg_channels_full
        self.bad_channels_map = bad_channels_map
        self.return_format = return_format
        self._per_subject_norm = per_subject_norm_stats or {}
        self._global_norm = global_norm_stats or {}
        self._audio_layout = audio_layout
        self._audio_manifest = audio_manifest or {}

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def _load_parquet_array(self, modality: str, sid: str, tid: str, value_cols_prefix: str | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return (values (T, C), col_names, t_sec)."""
        path = self.paths.parquet(modality, sid, tid)
        df = pd.read_parquet(path)
        t_col = "t_sec" if "t_sec" in df.columns else "t"
        t_sec = df[t_col].to_numpy()
        if value_cols_prefix:
            cols = [c for c in df.columns if c.startswith(value_cols_prefix)]
            names = [c[len(value_cols_prefix):] for c in cols]
        else:
            cols = [c for c in df.columns if c not in (t_col, "sample_idx")]
            names = list(cols)
        vals = df[cols].to_numpy(dtype=np.float32, copy=False)
        return vals, names, t_sec

    def _maybe_normalize(self, x: np.ndarray, modality: str, sid: str) -> np.ndarray:
        if self.normalize is None:
            return x
        if self.normalize_scope == "per_trial":
            return normalize_array(x, self.normalize)
        if self.normalize_scope == "per_subject":
            stats = self._per_subject_norm.get(sid, {}).get(modality)
            if stats is None:
                return normalize_array(x, self.normalize)
            c, s = stats
            return ((x - c) / s).astype(np.float32, copy=False)
        if self.normalize_scope == "global":
            stats = self._global_norm.get(modality)
            if stats is None:
                return normalize_array(x, self.normalize)
            c, s = stats
            return ((x - c) / s).astype(np.float32, copy=False)
        raise ValueError(f"unknown normalize_scope: {self.normalize_scope!r}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        sid = item["subject_id"]
        tid = item["trial_id"]
        seg: Segment = item["segment"]
        out: dict[str, Any] = {
            "subject_id": sid,
            "trial_id": tid,
            "kind": item["kind"],
            "segment_idx": seg.idx,
            "t_start_sec": seg.t_start_sec,
            "t_end_sec": seg.t_end_sec,
            "attended_speaker": item.get("attended_speaker"),
            "comprehension_correct": item.get("comprehension_correct"),
            "azimuth_attended_deg": item.get("azimuth_attended_deg"),
        }

        if "eeg" in self.modalities:
            vals, ch_names, _ = self._load_parquet_array("eeg", sid, tid, value_cols_prefix="ch_")
            bad = self.bad_channels_map.get(sid, [])
            vals, ch_names = apply_bad_channels(vals, ch_names, bad, mode=self.bad_channels)
            vals = self._maybe_normalize(vals, "eeg", sid)
            seg_vals = slice_signal(vals, EEG_SFREQ_DEFAULT, seg.t_start_sec, seg.t_end_sec)
            out["eeg"] = seg_vals
            out["eeg_channels"] = ch_names
            out["eeg_sfreq"] = EEG_SFREQ_DEFAULT

        for mod in ("gaze", "imu"):
            if mod not in self.modalities:
                continue
            vals, names, t_sec = self._load_parquet_array(mod, sid, tid)
            vals = self._maybe_normalize(vals, mod, sid)
            # Time vector starts at 0 for gaze/imu (Tobii recording clock).
            sf = (len(t_sec) - 1) / max(t_sec[-1] - t_sec[0], 1e-9) if len(t_sec) > 1 else 0.0
            seg_vals = slice_signal(vals, sf, seg.t_start_sec, seg.t_end_sec)
            out[mod] = seg_vals
            out[f"{mod}_columns"] = names
            out[f"{mod}_sfreq"] = float(sf)

        if "audio" in self.modalities:
            import soundfile as sf_lib
            speakers: dict[int, np.ndarray] = {}
            audio_sr = None
            trial_manifest = self._audio_manifest.get(tid) if self._audio_manifest else None
            for spk_meta in self._audio_layout:
                spk_no = spk_meta["speaker"]
                fname: str | None = None
                if trial_manifest:
                    fname = trial_manifest.get(str(spk_no))
                if fname is None and self.paths.local_path is not None:
                    # Fallback: glob the on-disk trial folder.
                    pattern = f"speaker{spk_no}_dev{spk_meta['device']}_{spk_meta['channel']}_"
                    folder = self.paths.local_path / "media" / "audio" / tid
                    matches = sorted(folder.glob(f"{pattern}*.flac"))
                    if matches:
                        fname = matches[0].name
                if fname is None:
                    continue
                wav, sr = sf_lib.read(self.paths.audio_speaker(tid, fname), dtype="float32", always_2d=False)
                audio_sr = sr if audio_sr is None else audio_sr
                seg_wav = slice_signal(wav, sr, seg.t_start_sec, seg.t_end_sec)
                speakers[spk_no] = seg_wav
            out["audio"] = speakers
            out["audio_sfreq"] = audio_sr or 0

        if "video" in self.modalities:
            out["video_path"] = str(self.paths.video(sid, tid))

        if self.return_format == "torch":
            try:
                import torch
            except ImportError as e:
                raise ImportError("return_format='torch' requires PyTorch") from e
            for k, v in list(out.items()):
                if isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(v)
                elif isinstance(v, dict) and all(isinstance(x, np.ndarray) for x in v.values()):
                    out[k] = {kk: torch.from_numpy(vv) for kk, vv in v.items()}
        elif self.return_format == "numpy":
            pass  # already numpy
        elif self.return_format != "dict":
            raise ValueError(f"unknown return_format: {self.return_format!r}")

        return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _norm_subjects(spec) -> list[int]:
    if spec == "all":
        return list(range(1, 17))
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, range):
        return list(spec)
    return [int(s) for s in spec]


def _norm_trials(spec, kinds_avail: dict[str, list[int]]) -> list[tuple[str, int]]:
    """Return list of (kind, k)."""
    if spec == "all":
        return [("training", k) for k in kinds_avail["training"]] + [("main", k) for k in kinds_avail["main"]]
    if spec == "main":
        return [("main", k) for k in kinds_avail["main"]]
    if spec == "training":
        return [("training", k) for k in kinds_avail["training"]]
    if isinstance(spec, int):
        return [("main", spec)]
    if isinstance(spec, range):
        return [("main", k) for k in spec]
    return [("main", int(s)) for s in spec]


def _norm_modalities(spec) -> list[str]:
    if spec == "all":
        return list(ALL_MODALITIES)
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def load_aad(
    *,
    subjects: Any = "all",
    trials: Any = "main",
    modalities: Any = ("eeg",),
    segment_length: float | None = None,
    overlap: float = 0.0,
    drop_last: bool = True,
    normalize: str | None = None,
    normalize_scope: str = "per_trial",
    bad_channels: str = "raw",
    return_format: str = "dict",
    splits: str | None = None,
    fold: int | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    local_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> MaestroDataset:
    """Build a dataset over (subject, trial, segment) tuples.

    See module README for full docs.
    """
    paths = DatasetPaths(
        local_path=Path(local_path) if local_path else None,
        repo_id=repo_id,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    # ----- metadata -----
    trials_df = pd.read_csv(paths.metadata("trials.csv"))
    per_sub_df = pd.read_csv(paths.metadata("trials_per_subject.csv"))
    bad_df = pd.read_csv(paths.metadata("bad_channels.csv"))
    eeg_meta = json.loads(paths.metadata("eeg_channels.json").read_text())
    audio_meta = json.loads(paths.metadata("audio_layout.json").read_text())
    try:
        audio_manifest = json.loads(paths.metadata("audio_manifest.json").read_text())
    except Exception:
        # Older dataset releases may not ship audio_manifest.json — fall back
        # to filename globbing in local-path mode (HF streaming will skip audio).
        audio_manifest = None

    eeg_channels = eeg_meta["channels"]
    bad_channels_map = {
        r["subject_id"]: [b for b in str(r.get("bad_channels", "") or "").split(";") if b]
        for _, r in bad_df.iterrows()
    }
    # Speaker → azimuth lookup
    spk_az = {s["speaker"]: s["azimuth_deg"] for s in audio_meta["speakers"]}

    # ----- selection -----
    subj_list = _norm_subjects(subjects)
    kinds_avail = {
        "main": sorted(trials_df.loc[trials_df["kind"] == "main", "trial_id"]
                       .str.removeprefix("eval_").astype(int).tolist()),
        "training": sorted(trials_df.loc[trials_df["kind"] == "training", "trial_id"]
                           .str.removeprefix("training_").astype(int).tolist()),
    }
    trial_pairs = _norm_trials(trials, kinds_avail)
    mods = _norm_modalities(modalities)
    for m in mods:
        if m not in ALL_MODALITIES:
            raise ValueError(f"unknown modality: {m!r} (valid: {ALL_MODALITIES})")

    # ----- build per-segment items -----
    items: list[dict[str, Any]] = []
    per_sub_idx = per_sub_df.set_index(["subject_id", "trial_id"])
    trials_idx = trials_df.set_index("trial_id")
    for s in subj_list:
        sid = f"S{s:02d}"
        for kind, k in trial_pairs:
            tid = f"{'training' if kind == 'training' else 'eval'}_{k:03d}"
            if tid not in trials_idx.index:
                continue
            trow = trials_idx.loc[tid]
            attended = int(trow["attended_speaker"])
            try:
                psrow = per_sub_idx.loc[(sid, tid)]
                cc = psrow["comprehension_correct"]
                cc = None if pd.isna(cc) else bool(cc)
            except KeyError:
                cc = None

            # Trial duration: read from EEG parquet to be precise.
            try:
                eeg_t = pd.read_parquet(paths.parquet("eeg", sid, tid), columns=["t_sec"])["t_sec"]
                duration = float(eeg_t.iloc[-1] - eeg_t.iloc[0])
            except FileNotFoundError:
                continue

            for seg in make_segments(duration, segment_length, overlap, drop_last):
                items.append({
                    "subject_id": sid,
                    "trial_id": tid,
                    "kind": kind,
                    "segment": seg,
                    "attended_speaker": attended,
                    "comprehension_correct": cc,
                    "azimuth_attended_deg": spk_az.get(attended),
                })

    # ----- splits -----
    if splits == "loso":
        if fold is None:
            raise ValueError("splits='loso' requires fold (0..15)")
        held_out = f"S{fold + 1:02d}"
        items = [it for it in items if it["subject_id"] == held_out]
    elif splits == "within":
        if fold is None:
            raise ValueError("splits='within' requires fold")
        # 80/20 within subject; fold ∈ {0=train, 1=test}
        items_train, items_test = [], []
        for it in items:
            n = int(it["trial_id"].rsplit("_", 1)[1])
            (items_test if n > 80 else items_train).append(it)
        items = items_train if fold == 0 else items_test

    return MaestroDataset(
        paths=paths,
        items=items,
        modalities=mods,
        normalize=normalize,
        normalize_scope=normalize_scope,
        bad_channels=bad_channels,
        eeg_channels_full=eeg_channels,
        bad_channels_map=bad_channels_map,
        return_format=return_format,
        per_subject_norm_stats=None,   # computed lazily on first access — TODO
        global_norm_stats=None,
        audio_layout=audio_meta["speakers"],
        audio_manifest=audio_manifest,
    )
