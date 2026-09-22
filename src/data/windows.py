"""Build + cache per-subject windowed AAD data from the raw corpus.

We cache at the *trial* level (not the window level) on scratch: overlapping
decision windows are materialised on the fly, so the cache stays small
(~per-trial 30 s arrays) and the window/hop can change without rebuilding.

Per trial we store, all on a shared ``sr`` Hz grid aligned to the audio
playback window:
    eeg   : (n_chans, T)          float32   preprocessed, re-referenced
    env   : (6, n_bands, T)       float32   gammatone envelopes, 6 fixed speakers
    gaze  : (T, 3)                float32   [gaze2d_x, gaze2d_y, pupil]
    imu   : (T, 6)                float32   [ax, ay, az, gx, gy, gz]
    attended : int (1..4)         the attended speaker (speakers 5/6 never attended)
    present  : (gaze, imu, video) bool      whether each context modality is real

Video is intentionally left out of the first materialisation (egocentric mp4
decode is expensive and -- per the dataset's own diagnostics -- mostly
redundant with gaze/IMU); ``present_video`` is False and the model receives a
zero video token. Turn it on later via the data config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import aad_compat as C

log = logging.getLogger("data.windows")

N_SPEAKERS = 6
N_ATTENDABLE = 4
GAZE_DIM = 8
IMU_DIM = 12
VIDEO_DIM = 16


@dataclass
class WindowSpec:
    win_s: float = 5.0
    sr: float = 64.0
    n_bands: int = 28
    overlap: float = 0.5            # window overlap fraction -> hop = win*(1-overlap)
    # --- cache-time (baked into the cached envelopes; affects cache tag) ---
    # audio_level_norm: scale each speaker's audio to unit RMS BEFORE the
    # gammatone, equalising the 6 speakers' loudness at source so the attended
    # (louder, +3..18 dB) speaker can't be picked by energy alone.
    audio_level_norm: bool = True
    # audio_norm: how the 6 speakers are level-equalised before the gammatone.
    #   "rms"         -- divide each channel by its empirical RMS (== audio_level_norm).
    #   "table_power" -- divide each channel by sqrt(documented presented power)
    #                    from trials.csv ("Device-k {Left,Right} Power"). Uses the
    #                    GROUND-TRUTH mixing gains rather than a measured estimate,
    #                    so all 6 candidates are exactly loudness-matched and the
    #                    4-way speaker decision can only be solved from EEG.
    #   "none"        -- leave native FLAC levels (re-introduces the confound).
    audio_norm: str = "rms"
    # lp_hz: lowpass cutoff (Hz) applied to the speaker envelopes (and, by default,
    # the EEG) after resampling -- restricting to the ~1-10 Hz speech-tracking band.
    # 0 disables. EEG is already 1 Hz high-passed in preprocess_eeg.
    lp_hz: float = 0.0
    hp_hz: float = 1.0             # informational; EEG HPF is done in preprocess_eeg
    # eeg_lp_hz: separate EEG lowpass. -1 => inherit lp_hz. Use a HIGHER cutoff
    # (e.g. 0/off or 30) for spectral/lateralisation models that need alpha (8-12 Hz)
    # and beta, which a 10 Hz envelope-band lowpass would destroy.
    eeg_lp_hz: float = -1.0
    # perfect_align (COMPLETE alignment, fixed 2026-06-12): valid data begins only once
    # BOTH (a) the EEG has started recording (eeg_unix[0], derived from first_sample_time)
    # AND (b) all 3 stereo devices are sounding (max playback_start). The anchor is the
    # later of the two: anchor = max(eeg_start, max(playback_start)). EEG is kept from the
    # anchor; each device's FLAC is sliced from (anchor - its playback_start); the window is
    # bounded by the earliest device's end AND the EEG end. This fixes the EEG-late trials
    # (e.g. S1/Eval-32, EEG starts ~4 s after audio) that the old code mis-sliced by
    # assuming EEG always started first. The pre-fix behaviour (anchor=min playback_start,
    # every envelope at index 0; and the interim max-playback-only anchor) is GONE.
    perfect_align: bool = True
    # gaze_traj_len: length of the raw (subject-relative, NOT z-scored) gaze x/y
    # trajectory exposed per window as the extra ``gaze_traj`` feature (2*len dims).
    # The absolute horizontal gaze position is the azimuth cue that separates the 4
    # source positions WITHIN a hemisphere -- what full 4-way source-ID needs.
    # Materialise-time only (raw gaze is cached), so changing it needs no rebuild.
    gaze_traj_len: int = 16
    # audio_feats: also load the rich per-speaker audio features (HuBERT layer-9 +
    # GPT-2 surprisal, from src/data/audio_features.py) aligned/sliced exactly like
    # the envelope, for content-based mixture matching. Adds `_af` to the cache tag
    # (separate cache from the envelope-only _pa2). w2v_pca_dim = HuBERT PCA target.
    audio_feats: bool = False
    w2v_pca_dim: int = 64
    # --- materialise-time (not cached; no rebuild needed to change) ---
    norm_cand: bool = True         # per-(speaker,band) z-score over the window
    norm_eeg: bool = True          # per-channel z-score over the window
    # Match-mismatch task framing:
    #   "speaker" -- candidates are the 6 fixed speakers; pick the attended one.
    #               (CONFOUNDED on this corpus: target is louder + spectrally
    #               distinct from maskers, so audio alone decodes it.)
    #   "shifted" -- candidates are the ATTENDED speaker's aligned envelope plus
    #               n_neg time-shifted spoilers of the SAME speaker. Audio is
    #               identical across candidates, so only genuine EEG-envelope
    #               temporal alignment can pick the match. (Accou/Francart MM.)
    mm_task: str = "speaker"
    n_neg: int = 1                 # number of shifted spoilers (n_candidates = n_neg+1)
    min_shift_s: float = 3.0       # min temporal separation of a spoiler from the matched window
    # spoiler_mode: how "shifted" spoilers are drawn.
    #   "grid"     -- distinct within-trial window positions >= min_shift_s from the match
    #                 (default; spoiler diversity shrinks at long windows).
    #   "circular" -- circularly-rolled copies of the FULL attended envelope at the same
    #                 window position. A circular copy preserves the envelope's marginal
    #                 statistics, so a candidate-only classifier stays at chance EVEN at long
    #                 decision windows -- the honest way to grow the decision window.
    spoiler_mode: str = "grid"

    @property
    def win_len(self) -> int:
        return int(round(self.win_s * self.sr))

    @property
    def hop_len(self) -> int:
        return max(1, int(round(self.win_len * (1.0 - self.overlap))))

    @property
    def eeg_cut(self) -> float:
        """Effective EEG lowpass cutoff (inherits lp_hz when eeg_lp_hz < 0)."""
        return self.lp_hz if self.eeg_lp_hz < 0 else self.eeg_lp_hz

    def tag(self) -> str:
        # Only cache-affecting params belong in the tag (overlap/norm_* are
        # applied at window time, not stored).
        if self.audio_norm == "table_power":
            a = "pwT"
        elif self.audio_norm == "none":
            a = "an0"
        else:                                  # "rms"
            a = "an1" if self.audio_level_norm else "an0"
        t = f"win{self.win_s:g}_sr{self.sr:g}_b{self.n_bands}_{a}"
        if self.lp_hz and self.lp_hz > 0:
            t += f"_lp{self.lp_hz:g}"
        if self.eeg_lp_hz >= 0 and self.eeg_lp_hz != self.lp_hz:
            t += f"_elp{self.eeg_lp_hz:g}"     # distinct EEG band (spectral models)
        if self.perfect_align:
            t += "_pa2"                        # COMPLETE alignment (eeg_start+device anchor);
                                               # _pa2 != stale _pa (max-playback-only) != old
        if self.audio_feats:
            t += f"_af{self.w2v_pca_dim}"      # + HuBERT(PCA) & GPT-2 surprisal candidates
        return t


@dataclass
class TrialRecord:
    eeg: np.ndarray      # (C, T)
    env: np.ndarray      # (6, n_bands, T)
    gaze: np.ndarray     # (T, 3)
    imu: np.ndarray      # (T, 6)
    attended: int        # 1..4
    subject: int
    trial_k: int
    present_gaze: bool
    present_imu: bool
    present_video: bool = False
    w2v: np.ndarray | None = None    # (4, Dw, T)  HuBERT(PCA) for talkers 1..4, time-aligned
    sem: np.ndarray | None = None    # (4, Ds, T)  GPT-2 surprisal/entropy/onset, time-aligned


# ----------------------------------------------------------------------------
# Signal helpers
# ----------------------------------------------------------------------------
def _resample_last(x: np.ndarray, sr_out: float, sr_in: float) -> np.ndarray:
    from fractions import Fraction

    from scipy.signal import resample_poly

    fr = Fraction(int(round(sr_out)), int(round(sr_in))).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1).astype(np.float32)


def _lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth lowpass along the last axis (the time axis).

    Used to confine the EEG and the speaker envelopes to the ~10 Hz cortical
    speech-tracking band. No-op if ``cutoff`` is unset or above Nyquist, or if
    the signal is too short for ``filtfilt``'s default padding.
    """
    if not cutoff or cutoff <= 0 or cutoff >= 0.5 * fs:
        return x.astype(np.float32)
    from scipy.signal import butter, filtfilt

    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    if x.shape[-1] <= 3 * max(len(a), len(b)):
        return x.astype(np.float32)
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def _interp_to_grid(t_src: np.ndarray, vals: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear-interp each column of ``vals`` (len(t_src), k) onto ``t_grid``."""
    if vals.ndim == 1:
        vals = vals[:, None]
    out = np.zeros((len(t_grid), vals.shape[1]), np.float32)
    if len(t_src) < 2:
        return out
    order = np.argsort(t_src)
    t_src = np.asarray(t_src)[order]
    for j in range(vals.shape[1]):
        col = np.asarray(vals[order, j], float)
        good = np.isfinite(col) & np.isfinite(t_src)
        if good.sum() < 2:
            continue
        out[:, j] = np.interp(t_grid, t_src[good], col[good]).astype(np.float32)
    return out


def _first_col(df, names: list[str]):
    for n in names:
        if n in df.columns:
            return np.asarray(df[n].values, float)
    return None


# ----------------------------------------------------------------------------
# Rich audio features (HuBERT layer-9 + GPT-2 surprisal) -- shared per-trial cache
# ----------------------------------------------------------------------------
AUDIOFEAT_DIR = Path("/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__audiofeat")
AUDIO_FEAT_SR = 64.0
_W2V_PCA: dict = {}


def _w2v_pca(dim: int):
    """Load (components (dim,768), mean (768,)) for HuBERT PCA; cached per dim."""
    if dim not in _W2V_PCA:
        p = AUDIOFEAT_DIR / f"pca_w2v{dim}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"missing HuBERT PCA {p}; run `python -m src.data.audio_features --fit-pca {dim}`")
        z = np.load(p)
        _W2V_PCA[dim] = (z["components"].astype(np.float32), z["mean"].astype(np.float32))
    return _W2V_PCA[dim]


def _load_trial_audiofeat(k: int, dim: int):
    """(w2v (4,T64,dim), sem (4,T64,3)) for the 4 attendable talkers, full 30 s @64 Hz."""
    z = np.load(AUDIOFEAT_DIR / f"trial{k}.npz")     # MAIN trials only
    w = z["w2v"].astype(np.float32)                  # (4, T64, 768)
    sem = z["sem"].astype(np.float32)                # (4, T64, 3)
    comp, mean = _w2v_pca(dim)
    w = (w - mean) @ comp.T                          # (4, T64, dim)
    return w.astype(np.float32), sem


# ----------------------------------------------------------------------------
# One trial -> TrialRecord
# ----------------------------------------------------------------------------
def build_trial(subject: int, k: int, spec: WindowSpec, *, kind: str = "main") -> TrialRecord | None:
    csv = C.load_trials_csv()
    tno = C.trial_name(k, kind)
    row = csv[csv["Trial No."] == tno]
    if not len(row):
        return None
    row = row.iloc[0]
    att = int(row["Attended Speaker"])
    if att not in (1, 2, 3, 4):
        return None

    # --- EEG ---
    data, ts = C.load_eeg_trial(subject, k, kind=kind)
    meta = C.load_eeg_time(subject, k, kind=kind)
    gaze2d = C.load_gaze_trial_2d(subject, k, kind=kind)
    ats = C.load_audio_timestamps(subject, k, kind=kind)
    raw_gaze = C.load_raw_gaze(subject, k, kind=kind)
    raw_imu = C.load_raw_imu(subject, k, kind=kind)

    aln = C.align_modalities_to_trial(
        eeg=data, eeg_ts=ts, eeg_time_meta=meta, gaze2d=gaze2d,
        audio_timestamps=ats, raw_gaze=raw_gaze, raw_imu=raw_imu,
    )
    raw = C.eeg_raw_to_mne(aln["eeg"])
    proc = C.preprocess_eeg(raw, reference="auto", apply_ica=False)
    eeg500 = proc.get_data().astype(np.float32)          # (32, T@500)
    eeg = _resample_last(eeg500, spec.sr, C.EEG_SFREQ)   # (32, T@sr)
    if spec.eeg_cut and spec.eeg_cut > 0:               # 1 Hz HPF already applied in preprocess_eeg
        eeg = _lowpass(eeg, spec.sr, spec.eeg_cut)       # envelope models: ~1-10 Hz; spectral: keep alpha/beta
    T = eeg.shape[1]
    eeg_unix = np.asarray(aln["eeg_unix"], float)
    # shared sr-grid spanning the EEG window
    t_grid = np.linspace(eeg_unix[0], eeg_unix[-1], T) if len(eeg_unix) >= 2 else np.arange(T) / spec.sr

    # --- COMPLETE per-device alignment (anchor = when EEG *and* all devices are live) ---
    # ps[i] is the playback_start of the i-th audio device (list order = Device-1,2,3), in
    # the same unix clock as t_grid (eeg_unix is built from first_sample_time). Valid data
    # begins only once BOTH the EEG is recording AND every device is sounding:
    #     anchor = max(eeg_start, max(playback_start))
    # We keep EEG from the anchor and slice each device's FLAC from (anchor - ps[device]),
    # so staggered onsets are removed AND EEG-late trials (eeg_start > max(ps), e.g.
    # S1/Eval-32) are sliced from the EEG onset rather than the audio onset. The window is
    # bounded by the earliest-ending device (min_ps + FLAC_DUR) AND the EEG end (t_grid[-1]).
    ps = [float(a["playback_start_time"]) for a in ats]
    FLAC_DUR = 30.0
    if getattr(spec, "perfect_align", True) and len(ps) >= 1:
        eeg_start = float(t_grid[0]); eeg_end = float(t_grid[-1])
        min_ps = min(ps)
        anchor = max(max(ps), eeg_start)                       # both EEG and all devices live
        overlap_s = min(min_ps + FLAC_DUR, eeg_end) - anchor   # bound by earliest device end & EEG end
        if overlap_s <= 0:
            log.warning("S%d trial %d: no EEG/audio overlap (anchor=%.3f, audio_end=%.3f, eeg_end=%.3f)",
                        subject, k, anchor, min_ps + FLAC_DUR, eeg_end)
            return None
        n_ov = max(1, int(round(overlap_s * spec.sr)))
        a_idx = int(np.argmin(np.abs(t_grid - anchor)))
        eeg = eeg[:, a_idx:a_idx + n_ov]
        t_grid = t_grid[a_idx:a_idx + n_ov]
        T = eeg.shape[1]
    else:
        anchor = None; overlap_s = None

    # --- Audio envelopes: 3 stereo device FLACs -> 6 speakers ---
    import soundfile as sf

    chans = []
    for di, dev in enumerate(("Device-1", "Device-2", "Device-3")):
        fn = row[dev]
        audio, sr = sf.read(str(C.PAIRS_DIR / fn))
        audio = np.asarray(audio, np.float32)
        if audio.ndim == 1:                              # mono fallback -> duplicate
            audio = np.stack([audio, audio], axis=1)
        if anchor is not None and di < len(ps):          # slice this device's FLAC from its onset offset
            o0 = max(0, int(round((anchor - ps[di]) * sr)))
            o1 = o0 + int(round(overlap_s * sr))
            audio = audio[o0:o1]
        for ch, side in ((0, "Left"), (1, "Right")):     # ch0=L, ch1=R
            x = audio[:, ch].astype(np.float64)
            if spec.audio_norm == "table_power":         # ground-truth gain equalise
                p = float(row[f"{dev} {side} Power"])
                x = x / (np.sqrt(p) + 1e-12)
            elif spec.audio_norm == "none":
                pass
            elif spec.audio_level_norm:                  # "rms": empirical equalise
                x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
            env = C.gammatone_envelope(x.astype(np.float32), sr,
                                       n_bands=spec.n_bands, sr_out=spec.sr)
            chans.append(env.T)                          # (n_bands, Te)
    Tenv = min(c.shape[1] for c in chans)
    env = np.stack([c[:, :Tenv] for c in chans], axis=0).astype(np.float32)  # (6, n_bands, Tenv)
    if spec.lp_hz and spec.lp_hz > 0:                    # confine envelopes to speech-tracking band
        env = _lowpass(env, spec.sr, spec.lp_hz)

    # --- rich audio features (HuBERT layer-9 + GPT-2 surprisal), sliced like the env ---
    # Each talker's full-FLAC 64 Hz feature stream is sliced from its device's onset
    # offset (anchor - playback_start[device]) -- exactly the envelope alignment -- so
    # w2v/sem are mutually time-aligned with the EEG and the envelope.
    rec_w2v = rec_sem = None
    if getattr(spec, "audio_feats", False) and anchor is not None and kind == "main":
        assert abs(spec.sr - AUDIO_FEAT_SR) < 1e-6, "audio_feats require sr=64 Hz"
        wf, smf = _load_trial_audiofeat(k, int(spec.w2v_pca_dim))      # (4,T64,D),(4,T64,3)
        Tf = wf.shape[1]; w_sl, s_sl = [], []
        for spk in range(4):                                          # talker -> its device
            o0 = max(0, int(round((anchor - ps[spk // 2]) * AUDIO_FEAT_SR)))
            o1 = min(Tf, o0 + n_ov)
            w_sl.append(wf[spk, o0:o1].T); s_sl.append(smf[spk, o0:o1].T)
        Tw = min(c.shape[1] for c in w_sl)
        rec_w2v = np.stack([c[:, :Tw] for c in w_sl]).astype(np.float32)  # (4,D,Tw)
        rec_sem = np.stack([c[:, :Tw] for c in s_sl]).astype(np.float32)  # (4,3,Tw)

    # Common-length trim across EEG, envelope (and audio features when present) so all
    # streams are EXACTLY the same number of samples.
    Tc = min(T, Tenv)
    if rec_w2v is not None:
        Tc = min(Tc, rec_w2v.shape[2])
        rec_w2v = rec_w2v[:, :, :Tc]; rec_sem = rec_sem[:, :, :Tc]
    if T != Tenv:
        log.debug("len align: EEG=%d env=%d -> %d (S%d trial %d)", T, Tenv, Tc, subject, k)
    eeg = eeg[:, :Tc]
    env = env[:, :, :Tc]
    t_grid = t_grid[:Tc]
    assert eeg.shape[1] == env.shape[2] == Tc

    # --- Gaze (per-eye + 2d + pupil) onto the grid ---
    present_gaze = isinstance(raw_gaze, type(gaze2d)) and len(aln.get("raw_gaze", [])) > 0
    gaze = np.zeros((Tc, 3), np.float32)
    rg = aln.get("raw_gaze")
    if rg is not None and len(rg) > 1:
        tcol = _first_col(rg, ["t_unix", "unix", "t"])
        gx = _first_col(rg, ["gaze2d_x"])
        gy = _first_col(rg, ["gaze2d_y"])
        lp = _first_col(rg, ["L_pupil"])
        rp = _first_col(rg, ["R_pupil"])
        if tcol is not None and gx is not None:
            pupil = np.nanmean(np.stack([c for c in (lp, rp) if c is not None], 0), 0) \
                if (lp is not None or rp is not None) else np.zeros_like(gx)
            stacked = np.stack([gx, gy if gy is not None else np.zeros_like(gx), pupil], 1)
            gaze = _interp_to_grid(tcol, stacked, t_grid)
            present_gaze = True
        else:
            present_gaze = False
    else:
        present_gaze = False

    # --- IMU onto the grid ---
    imu = np.zeros((Tc, 6), np.float32)
    present_imu = False
    ri = aln.get("raw_imu")
    if ri is not None and len(ri) > 1:
        tcol = _first_col(ri, ["t_unix", "unix", "t"])
        cols = [_first_col(ri, [c]) for c in ("ax", "ay", "az", "gx", "gy", "gz")]
        if tcol is not None and all(c is not None for c in cols):
            imu = _interp_to_grid(tcol, np.stack(cols, 1), t_grid)
            present_imu = True

    return TrialRecord(
        eeg=eeg, env=env, gaze=gaze, imu=imu, attended=att, subject=subject,
        trial_k=k, present_gaze=present_gaze, present_imu=present_imu, present_video=False,
        w2v=rec_w2v, sem=rec_sem,
    )


# ----------------------------------------------------------------------------
# Subject cache (atomic write to scratch)
# ----------------------------------------------------------------------------
def cache_path(cache_dir: Path, subject: int, spec: WindowSpec, kind: str) -> Path:
    return cache_dir / "aad_trials" / f"s{subject}_{kind}_{spec.tag()}.npz"


def build_and_cache_subject(
    subject: int, spec: WindowSpec, cache_dir: Path, *, kind: str = "main",
    n_trials: int = 100, overwrite: bool = False, progress: bool = True,
) -> Path:
    out = cache_path(cache_dir, subject, spec, kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        log.info("subject %d cache exists: %s", subject, out)
        return out

    import gc

    from tqdm import tqdm

    recs: list[TrialRecord] = []
    it = range(1, n_trials + 1)
    if progress:
        it = tqdm(it, desc=f"S{subject} trials", unit="trial")
    for k in it:
        try:
            r = build_trial(subject, k, spec, kind=kind)
        except Exception as exc:
            log.warning("S%d trial %d failed: %r", subject, k, exc)
            r = None
        if r is not None:
            recs.append(r)
        gc.collect()                       # MNE/librosa objects -> free each trial
    # Drop trials shorter than one window so the common-length trim below can't
    # fall under win_len (which would yield zero windows for the whole subject).
    recs = [r for r in recs if r.eeg.shape[1] >= spec.win_len]
    if not recs:
        raise RuntimeError(f"subject {subject}: no usable trials >= win_len")

    # Trim every trial to a COMMON length so we can store dense (non-object)
    # stacked arrays: fast to load and cheap to save (object-array pickling was
    # the source of the prepare-time OOM).
    L = min(r.eeg.shape[1] for r in recs)
    eeg = np.stack([r.eeg[:, :L] for r in recs]).astype(np.float32)        # (N,C,L)
    env = np.stack([r.env[:, :, :L] for r in recs]).astype(np.float32)    # (N,6,B,L)
    gaze = np.stack([r.gaze[:L] for r in recs]).astype(np.float32)        # (N,L,3)
    imu = np.stack([r.imu[:L] for r in recs]).astype(np.float32)          # (N,L,6)
    packed = dict(
        eeg=eeg, env=env, gaze=gaze, imu=imu,
        attended=np.array([r.attended for r in recs], np.int64),
        subject=np.array([r.subject for r in recs], np.int64),
        trial_k=np.array([r.trial_k for r in recs], np.int64),
        present=np.array([[r.present_gaze, r.present_imu, r.present_video]
                          for r in recs], bool),
        spec=np.array([spec.win_s, spec.overlap, spec.sr, spec.n_bands], np.float64),
        length=np.int64(L),
    )
    if recs[0].w2v is not None:                                       # rich audio features
        packed["w2v"] = np.stack([r.w2v[:, :, :L] for r in recs]).astype(np.float16)  # (N,4,D,L)
        packed["sem"] = np.stack([r.sem[:, :, :L] for r in recs]).astype(np.float32)  # (N,4,3,L)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **packed)
    tmp.replace(out)
    log.info("subject %d cached %d trials (L=%d) -> %s", subject, len(recs), L, out)
    return out


def load_subject_records(path: Path) -> list[TrialRecord]:
    z = np.load(path, allow_pickle=False)
    eeg, env, gaze, imu = z["eeg"], z["env"], z["gaze"], z["imu"]
    att, subj, tk, pres = z["attended"], z["subject"], z["trial_k"], z["present"]
    has_af = "w2v" in z.files
    w2v_a = z["w2v"] if has_af else None
    sem_a = z["sem"] if has_af else None
    recs = []
    for i in range(len(att)):
        recs.append(TrialRecord(
            eeg=eeg[i], env=env[i], gaze=gaze[i], imu=imu[i],
            attended=int(att[i]), subject=int(subj[i]), trial_k=int(tk[i]),
            present_gaze=bool(pres[i][0]), present_imu=bool(pres[i][1]),
            present_video=bool(pres[i][2]),
            w2v=(w2v_a[i].astype(np.float32) if has_af else None),
            sem=(sem_a[i].astype(np.float32) if has_af else None),
        ))
    return recs


# ----------------------------------------------------------------------------
# Per-window feature extraction (shared by numpy + torch views)
# ----------------------------------------------------------------------------
def gaze_window_features(gaze_win: np.ndarray) -> np.ndarray:
    """(W,3)->(8): mean/std x,y, mean/std pupil, mean/max speed."""
    x, y, pup = gaze_win[:, 0], gaze_win[:, 1], gaze_win[:, 2]
    dx = np.diff(x, prepend=x[:1]); dy = np.diff(y, prepend=y[:1])
    speed = np.sqrt(dx * dx + dy * dy)
    return np.array([
        x.mean(), y.mean(), x.std(), y.std(),
        pup.mean(), pup.std(), speed.mean(), speed.max(),
    ], np.float32)


def imu_window_features(imu_win: np.ndarray) -> np.ndarray:
    """(W,6)->(12): mean+std of ax,ay,az,gx,gy,gz."""
    return np.concatenate([imu_win.mean(0), imu_win.std(0)]).astype(np.float32)


def gaze_traj_features(gaze_win: np.ndarray, tl: int) -> np.ndarray:
    """(W,3)->(2*tl): raw subject-relative x then y resampled to ``tl`` points.

    Deliberately NOT z-scored: the absolute horizontal position is the azimuth cue
    that distinguishes the 4 source positions within a hemisphere. Intra-subject
    training handles the uncalibrated per-subject gaze offset/scale.
    """
    if tl <= 0:
        return np.zeros(0, np.float32)
    n = len(gaze_win)
    if n < 2:
        return np.zeros(2 * tl, np.float32)
    idx = np.linspace(0, n - 1, tl)
    ar = np.arange(n)
    x = np.interp(idx, ar, gaze_win[:, 0])
    y = np.interp(idx, ar, gaze_win[:, 1])
    return np.concatenate([x, y]).astype(np.float32)
