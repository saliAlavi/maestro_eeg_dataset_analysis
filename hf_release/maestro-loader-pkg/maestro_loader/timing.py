"""Read the per-trial timing JSON and expose unix-clock mappings.

The released dataset ships, for every (subject, trial), a ``timing`` record with
the wall-clock (unix) anchors of each modality. This module turns that record
into helpers that map any modality's native sample index to unix seconds, and
returns the *perfect-alignment window* — the interval during which EEG, the
audio playback, and the Tobii recording are all simultaneously live.

Clocks
------
* **EEG** runs on an internal monotonic clock. ``unix = first_sample_unix +
  (t_internal - t0_internal)``.
* **Audio** speech begins at ``audio.t0_unix`` (= earliest device playback).
* **Tobii** gaze / imu / video share one recording clock anchored to
  ``tobii.recording_start_unix``: ``unix = recording_start_unix + (t - t_first)``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class TrialTiming:
    raw: dict

    # -- alignment window -- #
    @property
    def anchor_unix(self) -> float:
        return float(self.raw["align"]["anchor_unix"])

    @property
    def end_unix(self) -> float:
        return float(self.raw["align"]["end_unix"])

    @property
    def overlap_sec(self) -> float:
        return float(self.raw["align"]["overlap_sec"])

    # -- per-modality unix conversions -- #
    def eeg_unix(self, t_internal: np.ndarray) -> np.ndarray:
        e = self.raw["eeg"]
        return float(e["first_sample_unix"]) + (
            np.asarray(t_internal, float) - float(e["t0_internal_sec"]))

    @property
    def eeg_sfreq(self) -> float:
        return float(self.raw["eeg"]["sfreq"])

    def tobii_unix(self, t_rel: np.ndarray, stream: str) -> np.ndarray:
        """Map a Tobii recording-relative timestamp (gaze/imu) to unix."""
        tob = self.raw["tobii"]
        t_first = float(tob.get(f"{stream}_t_first", 0.0) or 0.0)
        return float(tob["recording_start_unix"]) + (np.asarray(t_rel, float) - t_first)

    @property
    def audio_t0_unix(self) -> float:
        return float(self.raw["audio"]["t0_unix"])

    @property
    def audio_flac_dur(self) -> float:
        return float(self.raw["audio"]["flac_dur_sec"])

    @property
    def video_fps(self) -> float | None:
        v = self.raw["tobii"].get("video_fps")
        return float(v) if v else None

    @property
    def recording_start_unix(self) -> float:
        return float(self.raw["tobii"]["recording_start_unix"])


def load_timing(path) -> TrialTiming:
    with open(path) as f:
        return TrialTiming(json.load(f))
