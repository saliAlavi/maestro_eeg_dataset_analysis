"""Synthetic datamodule -- exercises the full data->model->runner stack on CPU
without touching the corpus. Used by ``mode=selftest`` and unit smoke tests.

Each trial gets a per-speaker spectral signature; the attended speaker's
signature is leaked into the EEG (so a model must read EEG to solve match-
mismatch) and weakly into the gaze features (so leave-one-modality-out shows a
real gaze contribution).
"""
from __future__ import annotations

import numpy as np

from .aad_datamodule import AADDataModule
from .windows import TrialRecord


class SyntheticDataModule(AADDataModule):
    def prepare(self) -> None:
        rng = np.random.default_rng(self.seed)
        spec = self.spec
        T = spec.win_len * 3
        B = spec.n_bands
        sig = rng.standard_normal((6, B)).astype(np.float32)   # per-speaker signature
        af = bool(getattr(spec, "audio_feats", False))
        Dw = int(getattr(spec, "w2v_pca_dim", 64))
        wsig = rng.standard_normal((4, Dw)).astype(np.float32)  # auditory signature (talkers 1..4)
        ssig = rng.standard_normal((4, 3)).astype(np.float32)   # semantic signature
        n_trials = 24
        for s in self.subjects:
            recs = []
            for k in range(1, n_trials + 1):
                att = int(rng.integers(0, 4))
                env = 0.3 * rng.standard_normal((6, B, T)).astype(np.float32)
                env += sig[:, :, None]
                eeg = 0.5 * rng.standard_normal((32, T)).astype(np.float32)
                eeg[:B] += 1.5 * sig[att][:, None]             # leak attended -> EEG (content)
                eeg[28:30] += 0.8 * (att - 1.5)                # leak PHYSICAL position -> EEG (directional)
                gaze = 0.1 * rng.standard_normal((T, 3)).astype(np.float32)
                gaze[:, 0] += 0.5 * att                        # weak gaze cue
                imu = 0.1 * rng.standard_normal((T, 6)).astype(np.float32)
                w2v = sem = None
                if af:
                    w2v = 0.3 * rng.standard_normal((4, Dw, T)).astype(np.float32)
                    w2v += wsig[:, :, None]
                    sem = 0.3 * rng.standard_normal((4, 3, T)).astype(np.float32)
                    sem += ssig[:, :, None]
                    eeg[10:14] += 1.0 * wsig[att][:4, None]     # leak attended -> EEG (auditory)
                recs.append(TrialRecord(
                    eeg=eeg, env=env, gaze=gaze, imu=imu, attended=att + 1,
                    subject=s, trial_k=k, present_gaze=True, present_imu=True,
                    present_video=False, w2v=w2v, sem=sem))
            self.by_subject[s] = recs
