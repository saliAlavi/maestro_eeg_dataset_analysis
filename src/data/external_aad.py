"""Public two-talker AAD corpora, loaded for the acceptance probe and for the
coupling branch.

The point of this module is a claim the paper makes and should have to defend:
that the two parts of \\merit{} which do not need behavioural streams --- the
audio-only acceptance probe and the time-centred coupling score --- transfer to
any match--mismatch corpus.  Three public corpora are wired up here.  None of
them records gaze, head motion or scene video, so only the EEG branch applies;
that is the point, not a limitation.

  KUL   Biesmans et al. / Das et al., 16 listeners, 20 trials, 64-ch @ 128 Hz.
        Two Dutch stories, one per ear, dichotic or HRTF-filtered.  Envelopes
        are taken from the *dry* stimuli as the dataset authors instruct.
  DTU   Fuglsang, Wong & Hjortkjaer, 18 listeners, 60 two-talker trials, 64-ch.
        One male and one female talker in simulated rooms.
  NJU   32 trials per listener, 32-ch @ 128 Hz, two talkers at known azimuths.
        Ships its own 128 Hz envelopes, which we use rather than recomputing.

Every corpus is reduced to the same object: per trial, the attended and the
unattended envelope on a common 64 Hz grid, plus a content id so that folds can
be made content-disjoint.  One envelope pipeline is used everywhere, including
for MAESTRO, so that a probe value is comparable across corpora.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("data.external")

FS = 64                       # common analysis rate, as in the main pipeline
ROOT = "/fs/ess/PAS2301/Data/EEG"
_POWER = 0.6                  # power-law compression, Biesmans et al. (2016)


# -- envelope ---------------------------------------------------------------
def envelope_from_wav(path, fs_out=FS, cache={}):
    """Broadband power-law envelope, anti-alias filtered and decimated.

    Deliberately the same for every corpus.  The probe reads only
    affine-invariant statistics, so the compression exponent and the cutoff are
    shared constants rather than per-corpus choices.
    """
    if path in cache:
        return cache[path]
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    e = np.abs(x).astype(np.float64) ** _POWER
    sos = butter(3, min(20.0, 0.45 * sr) / (sr / 2), btype="low", output="sos")
    e = sosfiltfilt(sos, e)
    n = int(round(len(e) * fs_out / sr))
    e = np.interp(np.linspace(0, len(e) - 1, n), np.arange(len(e)), e)
    out = e.astype(np.float32)
    if len(cache) < 512:
        cache[path] = out
    return out


def _resample(x, fs_in, fs_out=FS):
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(round(len(x) * fs_out / fs_in))
    return np.interp(np.linspace(0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)


def _pair_key(a, b):
    """Content group for a trial: the *unordered* pair of recordings.

    Grouping by the attended recording would put the label inside the group --
    a held-out group then contains the same talkers in the opposite role, and a
    shape classifier trained on role scores systematically *below* chance.  The
    pair is role-free, so holding one out is content-disjoint without being
    label-disjoint, and the probe measures role marking rather than talker
    identity.
    """
    return "|".join(sorted([a, b]))


@dataclass
class Trial:
    corpus: str
    subject: str
    trial: int
    content: str          # grouping key for content-disjoint folds
    att: np.ndarray       # attended envelope, 64 Hz
    unatt: np.ndarray     # unattended envelope, 64 Hz
    eeg: np.ndarray | None = None      # (C,T) at 64 Hz, or None for probe-only
    fs_eeg: float = FS


# -- KU Leuven --------------------------------------------------------------
def load_kul(subjects=None, with_eeg=False, root=f"{ROOT}/kuleuven"):
    import scipy.io as sio
    subs = subjects or [f"S{i}" for i in range(1, 17)]
    for s in subs:
        p = os.path.join(root, f"{s}.mat")
        if not os.path.exists(p):
            continue
        m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["trials"]
        for ti, t in enumerate(np.atleast_1d(m)):
            # stimuli[0] is the left-ear file, stimuli[1] the right-ear file;
            # the dataset authors instruct that envelopes come from the dry
            # stimuli even when the presentation was HRTF-filtered.
            stim = [str(x) for x in np.atleast_1d(t.stimuli)]
            if len(stim) != 2:
                continue
            ear = str(t.attended_ear).upper()
            att_name = stim[0] if ear.startswith("L") else stim[1]
            un_name = stim[1] if ear.startswith("L") else stim[0]
            def dry(n):
                return os.path.join(root, "stimuli",
                                    re.sub(r"_(hrtf|dry)\.wav$", "_dry.wav", n))
            if not (os.path.exists(dry(att_name)) and os.path.exists(dry(un_name))):
                continue
            a, u = envelope_from_wav(dry(att_name)), envelope_from_wav(dry(un_name))
            eeg = None
            if with_eeg:
                e = np.asarray(t.RawData.EegData, dtype=np.float32).T   # (C,T)
                eeg = np.stack([_resample(c, float(t.FileHeader.SampleRate))
                                for c in e])
            n = min(len(a), len(u), eeg.shape[1] if eeg is not None else 10**9)
            key = _pair_key(re.sub(r"_(hrtf|dry)\.wav$", "", att_name),
                            re.sub(r"_(hrtf|dry)\.wav$", "", un_name))
            yield Trial("kul", s, ti, key,
                        a[:n], u[:n], None if eeg is None else eeg[:, :n])


# -- DTU --------------------------------------------------------------------
DTU_EEG_CH = 64          # 73 stored channels = 64 scalp + EOG/mastoid extras


def load_dtu(subjects=None, with_eeg=False, root=f"{ROOT}/dtu"):
    """DTU stores one continuous recording per listener.

    Trials are the 50 s spans between event pairs: the onset carries the trial's
    trigger value and the offset the constant 191.  ``samples`` from the
    metadata is that event stream, so reshaping it to (n_trials, 2) gives the
    segmentation directly and it agrees with ``trigger`` element for element.
    """
    import scipy.io as sio
    subs = subjects or [f"S{i}" for i in range(1, 19)]
    for s in subs:
        mp = os.path.join(root, "metadata", f"{s}_metadata.mat")
        if not os.path.exists(mp):
            continue
        md = sio.loadmat(mp, squeeze_me=True, struct_as_record=False)
        eeg_all = fs_eeg = spans = None
        if with_eeg:
            ep = os.path.join(root, "eeg", f"{s}.mat")
            if not os.path.exists(ep):
                continue
            d = sio.loadmat(ep, squeeze_me=True, struct_as_record=False)["data"]
            eeg_all = np.asarray(d.eeg, dtype=np.float32)          # (T, 73)
            fs_eeg = float(np.atleast_1d(d.fsample.eeg).ravel()[0])
            sm = np.atleast_1d(md["samples"]).astype(np.int64)
            if sm.size % 2:
                continue
            spans = sm.reshape(-1, 2)
        nsp = np.atleast_1d(md["n_speakers"])
        amf = np.atleast_1d(md["attend_mf"])
        wm = [str(x) for x in np.atleast_1d(md["wavfile_male"])]
        wf = [str(x) for x in np.atleast_1d(md["wavfile_female"])]
        for ti in range(len(nsp)):
            if int(nsp[ti]) != 2:          # single-talker trials are not the task
                continue
            male_attended = int(amf[ti]) == 1
            an, un = (wm[ti], wf[ti]) if male_attended else (wf[ti], wm[ti])
            pa, pu = (os.path.join(root, "audio", an), os.path.join(root, "audio", un))
            if not (os.path.exists(pa) and os.path.exists(pu)):
                continue
            a, u = envelope_from_wav(pa), envelope_from_wav(pu)
            eeg = None
            if with_eeg:
                if ti >= len(spans):
                    continue
                s0, s1 = spans[ti]
                seg = eeg_all[s0:s1, :DTU_EEG_CH].T                 # (C, T)
                eeg = np.stack([_resample(c, fs_eeg) for c in seg])
            n = min(len(a), len(u), eeg.shape[1] if eeg is not None else 10**9)
            key = _pair_key(re.sub(r"_trial_\d+\.wav$", "", an),
                            re.sub(r"_trial_\d+\.wav$", "", un))
            yield Trial("dtu", s, ti, key, a[:n], u[:n],
                        None if eeg is None else eeg[:, :n])


# -- NJU --------------------------------------------------------------------
def load_nju(subjects=None, with_eeg=False,
             root=f"{ROOT}/nju/NJUNCA_preprocessed_arte_removed"):
    import h5py
    import scipy.io as sio
    info = sio.loadmat(os.path.join(root, "expinfomat_python_readable.mat"),
                       squeeze_me=True, struct_as_record=False)["expinfomat_struct"]
    files = sorted(f for f in os.listdir(root)
                   if re.fullmatch(r"S\d+\.mat", f))
    for fn in files:
        s = fn[:-4]
        # expinfomat_struct is 0-indexed after loading, so subject S{n} is
        # row n-1.  Verified against the file list: the non-empty rows are
        # exactly the subjects present, shifted by one.
        si = int(s[1:]) - 1
        rows = np.atleast_1d(info[si]) if 0 <= si < len(info) else None
        if rows is None or rows.size == 0:
            continue
        with h5py.File(os.path.join(root, fn), "r") as f:
            d = f["data"]
            fs = float(np.array(d["fsample"]["leftEnv"]).ravel()[0])
            n_tr = d["leftEnv"].shape[0]
            for ti in range(min(n_tr, len(rows))):
                r = rows[ti]
                le = np.array(f[d["leftEnv"][ti, 0]]).ravel()
                re_ = np.array(f[d["rightEnv"][ti, 0]]).ravel()
                left_att = str(r.attended_lr).lower().startswith("l")
                a, u = (le, re_) if left_att else (re_, le)
                a, u = _resample(a, fs), _resample(u, fs)
                eeg = None
                if with_eeg:
                    g = np.array(f[d["eeg"][ti, 0]])          # (C,T) or (T,C)
                    if g.ndim != 2 or min(g.shape) < 2:
                        continue                              # empty/short trial
                    if g.shape[0] > g.shape[1]:
                        g = g.T
                    fse = float(np.array(d["fsample"]["eeg"]).ravel()[0])
                    eeg = np.stack([_resample(c, fse) for c in g])
                n = min(len(a), len(u), eeg.shape[1] if eeg is not None else 10**9)
                key = _pair_key(str(r.l_audio), str(r.r_audio))
                yield Trial("nju", s, ti, key, a[:n], u[:n],
                            None if eeg is None else eeg[:, :n])


LOADERS = {"kul": load_kul, "dtu": load_dtu, "nju": load_nju}
__all__ = ["Trial", "LOADERS", "load_kul", "load_dtu", "load_nju",
           "envelope_from_wav", "FS"]
