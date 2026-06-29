"""Load one trial with every modality cross-modal aligned to a common window.

``load_aligned_trial`` returns an :class:`AlignedTrial` whose streams are all
clipped to the *perfect-alignment window* ``[anchor_unix, end_unix]`` (the span
during which EEG, audio playback and the Tobii recording are simultaneously
live). With ``target_sfreq`` set, every continuous stream is additionally
resampled onto one shared time grid so all modalities have identical length and
are sample-for-sample aligned — exactly the representation the decoder consumes.

Each modality is stored as a :class:`ModalitySignal` ``(data, rate, t0_unix)``
so a segment expressed in seconds-from-anchor slices every stream consistently,
whether streams are on the common grid or at native rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import preprocess as pp
from .paths import DatasetPaths
from .timing import load_timing, TrialTiming

HEMI = {1: 0, 2: 0, 3: 1, 4: 1}          # speaker -> Left(0)/Right(1)
INOUT = {1: 1, 2: 0, 3: 0, 4: 1}         # speaker -> outer(1)/inner(0)
DEFAULT_TARGET_SFREQ = 64.0
N_MELS = 28


@dataclass
class ModalitySignal:
    data: np.ndarray          # time axis is the LAST axis
    rate: float               # samples / sec
    t0_unix: float            # unix time of data[..., 0]
    columns: list[str] | None = None

    def slice_seconds(self, t0: float, t1: float, anchor_unix: float) -> np.ndarray:
        """Slice [t0, t1) seconds-from-anchor along the time axis."""
        i0 = int(round((anchor_unix + t0 - self.t0_unix) * self.rate))
        i1 = int(round((anchor_unix + t1 - self.t0_unix) * self.rate))
        i0 = max(0, i0)
        return self.data[..., i0:i1]


@dataclass
class AlignedTrial:
    subject_id: str
    trial_id: str
    kind: str
    anchor_unix: float
    end_unix: float
    target_sfreq: float | None
    signals: dict[str, ModalitySignal]
    labels: dict[str, Any]
    video_path: str | None = None
    video_fps: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.end_unix - self.anchor_unix


# --------------------------------------------------------------------------- #
def _anti_alias(x, sr_in, target, axis=-1):
    """Low-pass before downsampling raw continuous signals onto a coarser grid."""
    if target >= sr_in:
        return x
    from scipy import signal as _s
    wn = min(0.45 * target / (sr_in / 2.0), 0.99)
    b, a = _s.butter(4, wn, btype="low")
    return _s.filtfilt(b, a, x, axis=axis)


def _to_grid(t_unix, data_time_last, t_grid, sr_in, target, smooth):
    """Resample a (C, T) (time-last) stream onto t_grid via optional AA + interp."""
    x = np.asarray(data_time_last, dtype=np.float64)
    if smooth and target < sr_in and x.shape[-1] > 12:
        x = _anti_alias(x, sr_in, target, axis=-1)
    # interp expects time-first
    xf = x.T if x.ndim == 2 else x
    g = pp.interp_to_grid(t_unix, xf, t_grid)
    return (g.T if g.ndim == 2 else g).astype(np.float32)


def load_aligned_trial(
    paths: DatasetPaths,
    subject_id: str,
    trial_id: str,
    *,
    modalities: list[str],
    trials_meta: pd.DataFrame,
    preprocess: bool | dict = False,
    target_sfreq: float | None = None,
    audio_feature: str = "auto",
    eeg_cfg: dict | None = None,
) -> AlignedTrial | None:
    tj = paths.timing(subject_id, trial_id)
    timing: TrialTiming = load_timing(tj)
    anchor, end = timing.anchor_unix, timing.end_unix
    if end - anchor <= 0:
        return None

    do_pp = bool(preprocess) if isinstance(preprocess, bool) else True
    pp_cfg = preprocess if isinstance(preprocess, dict) else (eeg_cfg or None)
    if target_sfreq is None and do_pp:
        target_sfreq = DEFAULT_TARGET_SFREQ
    if audio_feature == "auto":
        audio_feature = "mel" if (do_pp and target_sfreq) else "waveform"

    t_grid = None
    if target_sfreq:
        T = max(1, int(np.floor((end - anchor) * target_sfreq)))
        t_grid = anchor + np.arange(T) / target_sfreq

    signals: dict[str, ModalitySignal] = {}
    kind = "main" if trial_id.startswith("eval") else "training"

    # ---------------- EEG ---------------- #
    if "eeg" in modalities:
        df = pd.read_parquet(paths.parquet("eeg", subject_id, trial_id))
        ch_cols = [c for c in df.columns if c.startswith("ch_")]
        ch_names = [c[3:] for c in ch_cols]
        data = df[ch_cols].to_numpy(np.float64).T          # (C, T)
        t_unix = timing.eeg_unix(df["t_sec"].to_numpy())
        sr = timing.eeg_sfreq
        if do_pp:
            data = pp.preprocess_eeg(data, ch_names, sr, pp_cfg).astype(np.float64)
        if t_grid is not None:
            arr = _to_grid(t_unix, data, t_grid, sr, target_sfreq, smooth=not do_pp)
            signals["eeg"] = ModalitySignal(arr, target_sfreq, anchor, ch_names)
        else:
            m = (t_unix >= anchor) & (t_unix <= end)
            signals["eeg"] = ModalitySignal(
                data[:, m].astype(np.float32), sr, float(t_unix[m][0]), ch_names)

    # ------------- Gaze / IMU (Tobii) ------------- #
    for mod in ("gaze", "imu"):
        if mod not in modalities:
            continue
        if not paths.exists(f"data/{mod}/subject={subject_id}/trial={trial_id}.parquet"):
            continue
        df = pd.read_parquet(paths.parquet(mod, subject_id, trial_id))
        cols = [c for c in df.columns if c != "t"]
        vals = df[cols].to_numpy(np.float64).T             # (C, T)
        t_unix = timing.tobii_unix(df["t"].to_numpy(), mod)
        sr = (len(t_unix) - 1) / max(t_unix[-1] - t_unix[0], 1e-9) if len(t_unix) > 1 else 0.0
        if t_grid is not None:
            arr = _to_grid(t_unix, vals, t_grid, sr or target_sfreq, target_sfreq,
                           smooth=False)
            signals[mod] = ModalitySignal(arr, target_sfreq, anchor, cols)
        else:
            m = (t_unix >= anchor) & (t_unix <= end)
            if m.sum() == 0:
                continue
            signals[mod] = ModalitySignal(
                vals[:, m].astype(np.float32), sr, float(t_unix[m][0]), cols)

    # ---------------- Audio ---------------- #
    if "audio" in modalities:
        sig = _load_audio(paths, trial_id, timing, anchor, end, t_grid,
                          target_sfreq, audio_feature)
        if sig is not None:
            signals["audio"] = sig

    # ---------------- Video (lazy) ---------------- #
    video_path = None
    if "video" in modalities and paths.exists(
            f"media/video/subject={subject_id}/{trial_id}.mp4"):
        video_path = str(paths.video(subject_id, trial_id))

    # ---------------- Labels ---------------- #
    row = trials_meta.loc[trials_meta["trial_id"] == trial_id]
    att = int(row.iloc[0]["attended_speaker"]) if len(row) else -1
    labels = {
        "attended_speaker": att,                       # 1..4
        "hemisphere": HEMI.get(att),                   # 0=L 1=R
        "inout": INOUT.get(att),                       # 0=inner 1=outer
        "speaker4": att - 1 if att >= 1 else -1,       # 0..3
    }

    return AlignedTrial(
        subject_id=subject_id, trial_id=trial_id, kind=kind,
        anchor_unix=anchor, end_unix=end, target_sfreq=target_sfreq,
        signals=signals, labels=labels,
        video_path=video_path, video_fps=timing.video_fps,
        meta={"overlap_sec": end - anchor,
              "recording_start_unix": timing.recording_start_unix},
    )


def _load_audio(paths, trial_id, timing, anchor, end, t_grid, target_sfreq,
                audio_feature):
    import soundfile as sf
    import json
    try:
        manifest = json.loads(paths.metadata("audio_manifest.json").read_text())
    except Exception:
        manifest = {}
    tman = manifest.get(trial_id, {})
    if not tman:
        return None
    t0 = timing.audio_t0_unix
    per_spk = []
    cols = []
    sr_native = None
    for spk in range(1, 7):
        fname = tman.get(str(spk))
        if fname is None:
            continue
        wav, sr = sf.read(str(paths.audio_file(trial_id, fname)),
                          dtype="float32", always_2d=False)
        sr_native = sr
        i0 = max(0, int(round((anchor - t0) * sr)))
        i1 = int(round((end - t0) * sr))
        clip = np.asarray(wav[i0:i1], dtype=np.float32)
        if audio_feature == "mel":
            feat = pp.mel_envelope(clip, sr, target_sfreq or DEFAULT_TARGET_SFREQ, N_MELS)
            if t_grid is not None and feat.shape[-1] != len(t_grid):
                feat = _fit_len(feat, len(t_grid))
            per_spk.append(feat)                                  # (n_mels, T)
        elif audio_feature == "envelope":
            from scipy.signal import hilbert
            envv = np.abs(hilbert(clip)).astype(np.float32)
            if target_sfreq:
                envv = pp.resample_to(envv, sr, target_sfreq)
                if t_grid is not None:
                    envv = _fit_len(envv, len(t_grid))
            per_spk.append(envv)                                  # (T,)
        else:  # waveform
            if target_sfreq:
                clip = pp.resample_to(clip, sr, target_sfreq)
            per_spk.append(clip)
        cols.append(f"speaker{spk}")
    if not per_spk:
        return None
    if audio_feature == "waveform" and not target_sfreq:
        rate, t0_out = float(sr_native), anchor
    else:
        rate, t0_out = float(target_sfreq or sr_native), anchor
        per_spk = [_fit_len(a, max(s.shape[-1] for s in per_spk)) for a in per_spk]
    arr = np.stack(per_spk, axis=0)                               # (n_spk, [n_mels,] T)
    return ModalitySignal(arr, rate, t0_out, cols)


def _fit_len(a: np.ndarray, T: int) -> np.ndarray:
    """Pad/truncate the last axis to length T."""
    n = a.shape[-1]
    if n == T:
        return a
    if n > T:
        return a[..., :T]
    pad = [(0, 0)] * (a.ndim - 1) + [(0, T - n)]
    return np.pad(a, pad)
