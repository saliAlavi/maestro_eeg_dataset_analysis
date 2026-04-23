"""IO helpers for loading every modality of the AAD dataset."""
from __future__ import annotations

import gzip
import json
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import (
    AUDIO_DIR,
    EXPERIMENT_DIR,
    PAIRS_DIR,
    TRIALS_CSV,
    VIDEO_DIR,
    N_SUBJECTS,
    N_TRAIN_TRIALS,
    N_MAIN_TRIALS,
)


# --------------------------------------------------------------------------- #
# Static metadata
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_trials_csv() -> pd.DataFrame:
    """Return the canonical trial metadata as a DataFrame.

    Main trials in the underlying CSV are named as bare integers ('1'..'100');
    this helper rewrites them to the 'Trial-N' form so downstream lookups can
    use either convention without surprises. A ``Trial No. Raw`` column
    preserves the original CSV string.
    """
    df = pd.read_csv(TRIALS_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["Trial No. Raw"] = df["Trial No."].astype(str)
    df["Trial No."] = df["Trial No. Raw"].str.replace(
        r"^(\d+)$", r"Trial-\1", regex=True
    )
    return df


def trial_row(trial_name: str | int):
    """Look up a single trial row by name, accepting multiple conventions.

    Accepts: 'Trial-1', '1', 1, 'Training-1'. Returns a pandas Series (the
    first matching row) or raises KeyError if no trial matches.
    """
    df = load_trials_csv()
    name = str(trial_name)
    candidates = {name}
    if name.isdigit():
        candidates.add(f"Trial-{name}")
    if name.startswith("Trial-") and name[6:].isdigit():
        candidates.add(name[6:])
    mask = df["Trial No."].isin(candidates) | df["Trial No. Raw"].isin(candidates)
    hits = df[mask]
    if not len(hits):
        raise KeyError(f"Trial {trial_name!r} not found in trials.csv")
    return hits.iloc[0]


def load_answers(subject: int) -> pd.DataFrame:
    path = EXPERIMENT_DIR / f"Subject {subject}" / "answers.json"
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["subject"] = subject
    return df


def load_demographics(subject: int) -> dict:
    path = EXPERIMENT_DIR / f"Subject {subject}" / "demographic.json"
    with open(path) as f:
        return json.load(f)


def list_subjects() -> list[int]:
    """Return sorted list of subject indices that actually exist on disk."""
    out = []
    for p in EXPERIMENT_DIR.iterdir():
        m = re.match(r"Subject (\d+)", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


# --------------------------------------------------------------------------- #
# EEG / gaze-2D / audio timestamps (experiment_data)
# --------------------------------------------------------------------------- #
def _eval_dir(subject: int, trial_idx_1based: int) -> Path:
    """Main trial K (1..100) lives at ``Subject N/Eval-K``.

    The 5 training trials are in separate ``Training-K`` folders and are
    accessed via the ``kind='training'`` argument on the loaders below.
    """
    return EXPERIMENT_DIR / f"Subject {subject}" / f"Eval-{trial_idx_1based}"


def _training_dir(subject: int, training_idx_1based: int) -> Path:
    """Training trial K (1..5) lives at ``Subject N/Training-K``."""
    return EXPERIMENT_DIR / f"Subject {subject}" / f"Training-{training_idx_1based}"


def _trial_dir(subject: int, k: int, kind: str = "main") -> Path:
    """Unified resolver.

    Parameters
    ----------
    k : int
        Trial index (1-based). 1..100 for main, 1..5 for training.
    kind : {'main', 'training'}
        Which block to index into.
    """
    if kind == "main":
        return _eval_dir(subject, k)
    if kind == "training":
        return _training_dir(subject, k)
    raise ValueError(f"kind must be 'main' or 'training', got {kind!r}")


def trial_name(k: int, kind: str = "main") -> str:
    """Canonical ``Trial No.`` string for a (k, kind) pair.

    Matches the normalized ``Trial No.`` column produced by ``load_trials_csv``.
    """
    if kind == "main":
        return f"Trial-{k}"
    if kind == "training":
        return f"Training-{k}"
    raise ValueError(kind)


def load_eeg_trial(subject: int, trial_idx_1based: int, kind: str = "main") -> tuple[np.ndarray, np.ndarray]:
    """Load one trial's EEG.

    Parameters
    ----------
    subject : int
    trial_idx_1based : int
        1..100 when ``kind='main'``; 1..5 when ``kind='training'``.
    kind : {'main', 'training'}

    Returns
    -------
    data : (n_times, 32) float array of EEG samples.
    ts : (n_times,) internal monotonic eeg_ts timestamps in seconds.
    """
    path = _trial_dir(subject, trial_idx_1based, kind) / "eeg_data.p"
    with open(path, "rb") as f:
        rows = pickle.load(f)
    ts = np.asarray([r["eeg_ts"] for r in rows], dtype=float)
    data = np.asarray([r["sample"][:32] for r in rows], dtype=float)
    return data, ts


def load_eeg_time(subject: int, trial_idx_1based: int, kind: str = "main") -> dict:
    path = _trial_dir(subject, trial_idx_1based, kind) / "eeg_time_data.p"
    with open(path, "rb") as f:
        rows = pickle.load(f)
    return rows[0]


def load_gaze_trial_2d(subject: int, trial_idx_1based: int, kind: str = "main") -> pd.DataFrame:
    path = _trial_dir(subject, trial_idx_1based, kind) / "gaze_data.p"
    with open(path, "rb") as f:
        rows = pickle.load(f)
    gx, gy, gts = [], [], []
    for r in rows:
        g2 = r.get("gaze2d", [np.nan, np.nan])
        if g2 is None or (hasattr(g2, "__len__") and len(g2) < 2):
            gx.append(np.nan); gy.append(np.nan)
        else:
            gx.append(float(g2[0])); gy.append(float(g2[1]))
        gts.append(float(r["gaze_ts"]))
    return pd.DataFrame(dict(gaze_ts=gts, gaze_x=gx, gaze_y=gy))


def load_gaze_time(subject: int, trial_idx_1based: int, kind: str = "main") -> dict:
    path = _trial_dir(subject, trial_idx_1based, kind) / "gaze_time_data.p"
    with open(path, "rb") as f:
        rows = pickle.load(f)
    return rows[0]


def load_audio_timestamps(subject: int, trial_idx_1based: int, kind: str = "main") -> list[dict]:
    path = _trial_dir(subject, trial_idx_1based, kind) / "audio_timestamps.json"
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Video-recording bundle (Tobii Glasses 3 per-eye gaze, IMU, scene video)
# --------------------------------------------------------------------------- #
def resolve_video_trial_mapping(subject: int) -> dict[int, Path]:
    """Return {video_folder_number: Path} for a subject.

    Semantics of the returned key ``K`` (1..105):
        K = 1..5   → training trial (K) — same order as experiment_data/Training-K
        K = 6..105 → main trial (K-5)   — same order as experiment_data/Eval-(K-5)

    Handles three naming conventions observed in the dataset:
        (a) Numeric names ``'1'`` .. ``'105'`` — used directly. If any are
            missing (e.g. S2 lacks folders 82, 83), those keys are absent
            from the output (caller sees ``.get(K) → None``).
        (b) Numeric names with a strict prefix (rare) — treated as (a).
        (c) Timestamp names (e.g. ``20250410T221008Z``, S3+) — mtime-sort,
            oldest first ≡ recording-earliest ≡ training-1 .. trial-100.
    """
    root = VIDEO_DIR / f"Subject {subject}"
    if not root.exists():
        return {}
    folders = [p for p in root.iterdir() if p.is_dir()]
    # Case (a)/(b): every folder name parses as an int — use the integer
    # directly. Missing integers (like S2's 82, 83) simply don't appear in
    # the mapping.
    try:
        numeric = {int(p.name): p for p in folders}
        if numeric:
            return {k: numeric[k] for k in sorted(numeric)}
    except ValueError:
        pass
    # Case (c): timestamp names. Parse the embedded recording-start time from
    # each name; sort ASCENDING by that time. Empirically (16 subjects) this
    # ordering is monotonically matched to the audio-playback clock
    # (corr ≥ 0.99 with trial-to-trial audio deltas), while mtime-sort is not
    # reliable because files were rsync'd in bulk and mtime == copy time.
    import re as _re
    _TS = _re.compile(r"^(\d{8})T(\d{6})Z$")
    parsed: list[tuple[str, Path]] = []
    unparsed: list[Path] = []
    for p in folders:
        m = _TS.match(p.name)
        if m:
            parsed.append((m.group(1) + m.group(2), p))
        else:
            unparsed.append(p)
    parsed.sort(key=lambda t: t[0])
    # Unparsed fall to the end in mtime order (rare edge case).
    unparsed.sort(key=lambda q: q.stat().st_mtime)
    ordered = [p for _, p in parsed] + unparsed

    expected = N_TRAIN_TRIALS + N_MAIN_TRIALS  # 105
    if len(ordered) > expected:
        # Anomaly: more folders than trials. For any surplus we drop the
        # folders whose gaze duration is most anomalous (extreme deviation
        # from 30 s). Caller should double-check such subjects manually.
        import gzip as _gz, json as _j
        def _dur(folder: Path) -> float:
            gp = folder / "gazedata.gz"
            if not gp.exists(): return 0.0
            first = last = None
            try:
                with _gz.open(gp, "rt") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln: continue
                        r = _j.loads(ln)
                        if r.get("type") != "gaze": continue
                        t = r.get("timestamp")
                        if t is None: continue
                        if first is None: first = t
                        last = t
            except Exception: return 0.0
            return (last - first) if first is not None else 0.0

        durs = [_dur(f) for f in ordered]
        # Rank by |dur - 32| (30 s audio + ~2 s pre-roll); largest surplus is
        # dropped.
        import numpy as _np
        dev = _np.abs(_np.array(durs) - 32.0)
        keep = _np.argsort(dev)[:expected]
        ordered = [ordered[i] for i in sorted(keep.tolist())]

    return {i + 1: p for i, p in enumerate(ordered)}


def video_folder_for_trial(trial_idx_1based: int, kind: str = "main") -> int:
    """Translate (trial index, kind) → Video-Recordings folder number K ∈ [1, 105].

    Training-K (K∈[1,5]) lives in video folder K.
    Main Trial-K (K∈[1,100]) lives in video folder K + 5.
    """
    if kind == "training":
        if not 1 <= trial_idx_1based <= N_TRAIN_TRIALS:
            raise ValueError(f"training trial index out of range: {trial_idx_1based}")
        return trial_idx_1based
    if kind == "main":
        if not 1 <= trial_idx_1based <= N_MAIN_TRIALS:
            raise ValueError(f"main trial index out of range: {trial_idx_1based}")
        return trial_idx_1based + N_TRAIN_TRIALS
    raise ValueError(f"kind must be 'main' or 'training', got {kind!r}")


def video_trial_dir(subject: int, trial_idx_1based: int, kind: str = "main") -> Path | None:
    """Return the Path of the Video-Recordings folder for a given (trial, kind)."""
    folder_num = video_folder_for_trial(trial_idx_1based, kind)
    return resolve_video_trial_mapping(subject).get(folder_num)


def _read_gz_jsonl(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_raw_gaze(subject: int, trial_idx_1based: int, kind: str = "main") -> pd.DataFrame:
    """Per-eye gaze stream from Tobii Glasses 3 ``gazedata.gz``.

    Parameters
    ----------
    subject, trial_idx_1based, kind :
        See :func:`load_eeg_trial`. Video-Recordings folders are offset from
        ``Eval-K`` by +5 (folder 6 = Eval-1 = Trial-1); training folders are
        1:1 (folder 1 = Training-1).
    """
    d = video_trial_dir(subject, trial_idx_1based, kind)
    if d is None:
        return pd.DataFrame()
    path = d / "gazedata.gz"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for rec in _read_gz_jsonl(path):
        if rec.get("type") != "gaze":
            continue
        t = rec.get("timestamp", np.nan)
        data = rec.get("data", {})
        g2 = data.get("gaze2d", [np.nan, np.nan]) or [np.nan, np.nan]
        g3 = data.get("gaze3d", [np.nan, np.nan, np.nan]) or [np.nan, np.nan, np.nan]
        eL = data.get("eyeleft") or {}
        eR = data.get("eyeright") or {}
        def _vec(d, k, n):
            v = d.get(k)
            if not v or len(v) != n:
                return [np.nan] * n
            return list(v)
        lorig = _vec(eL, "gazeorigin", 3)
        ldir  = _vec(eL, "gazedirection", 3)
        rorig = _vec(eR, "gazeorigin", 3)
        rdir  = _vec(eR, "gazedirection", 3)
        rows.append((
            t, g2[0], g2[1], g3[0], g3[1], g3[2],
            *lorig, *ldir, eL.get("pupildiameter", np.nan),
            *rorig, *rdir, eR.get("pupildiameter", np.nan),
        ))
    cols = [
        "t", "gaze2d_x", "gaze2d_y", "gaze3d_x", "gaze3d_y", "gaze3d_z",
        "L_ox", "L_oy", "L_oz", "L_dx", "L_dy", "L_dz", "L_pupil",
        "R_ox", "R_oy", "R_oz", "R_dx", "R_dy", "R_dz", "R_pupil",
    ]
    return pd.DataFrame(rows, columns=cols)


def load_raw_imu(subject: int, trial_idx_1based: int, kind: str = "main") -> pd.DataFrame:
    d = video_trial_dir(subject, trial_idx_1based, kind)
    if d is None:
        return pd.DataFrame()
    path = d / "imudata.gz"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for rec in _read_gz_jsonl(path):
        if rec.get("type") != "imu":
            continue
        t = rec.get("timestamp", np.nan)
        data = rec.get("data", {})
        acc = data.get("accelerometer", [np.nan] * 3) or [np.nan] * 3
        gyr = data.get("gyroscope", [np.nan] * 3) or [np.nan] * 3
        rows.append((t, *acc, *gyr))
    return pd.DataFrame(rows, columns=["t", "ax", "ay", "az", "gx", "gy", "gz"])


def load_audio_file(filename: str) -> tuple[np.ndarray, int]:
    """Load a stimulus pair by filename (e.g., '1195_2592.flac')."""
    import soundfile as sf
    path = PAIRS_DIR / filename
    if not path.exists():
        # Training/practice files may live under AUDIO_DIR directly or elsewhere.
        for alt in AUDIO_DIR.rglob(filename):
            path = alt
            break
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)
