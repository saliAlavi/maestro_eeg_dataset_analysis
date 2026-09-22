"""Concrete AAD datamodule: caches per-subject windows and yields protocol splits.

Splits are made at the **trial** level, never the window level, so overlapping
windows from one trial can't leak between train and test.

Protocols (matching the project's evaluation contract):
  * ``within`` -- per subject, trial-level K-fold CV.
  * ``loso``   -- leave-one-subject-out across the loaded subjects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from .base import AADView, AbstractDataModule, SplitDesc, WindowIndex
from .windows import (GAZE_DIM, IMU_DIM, N_SPEAKERS, VIDEO_DIM, TrialRecord,
                      WindowSpec, build_and_cache_subject, cache_path,
                      load_subject_records)

log = logging.getLogger("data.aad")


def _windows_for(records: List[TrialRecord], spec: WindowSpec) -> List[WindowIndex]:
    W, hop = spec.win_len, spec.hop_len
    idx: List[WindowIndex] = []
    for p, r in enumerate(records):
        T = r.eeg.shape[1]
        for s in range(0, T - W + 1, hop):
            idx.append(WindowIndex(p, s))
    return idx


class AADDataModule(AbstractDataModule):
    def __init__(self, *, subjects: List[int], cache_dir: str | Path,
                 spec: WindowSpec, kind: str = "main", n_folds: int = 5,
                 overwrite_cache: bool = False, seed: int = 0,
                 split_mode: str = "chrono_forward"):
        self.subjects = list(subjects)
        self.cache_dir = Path(cache_dir)
        self.spec = spec
        self.kind = kind
        self.n_folds = n_folds
        self.overwrite_cache = overwrite_cache
        self.seed = seed
        self.split_mode = split_mode
        self.by_subject: dict[int, List[TrialRecord]] = {}

    # -- build/load ----------------------------------------------------------
    def prepare(self) -> None:
        for s in self.subjects:
            path = cache_path(self.cache_dir, s, self.spec, self.kind)
            if not path.exists() or self.overwrite_cache:
                build_and_cache_subject(
                    s, self.spec, self.cache_dir, kind=self.kind,
                    overwrite=self.overwrite_cache,
                )
            self.by_subject[s] = load_subject_records(path)
            log.info("loaded S%d: %d trials, %d windows", s,
                     len(self.by_subject[s]),
                     len(_windows_for(self.by_subject[s], self.spec)))

    def feature_dims(self) -> dict:
        task = getattr(self.spec, "mm_task", "speaker")
        n_cand = (self.spec.n_neg + 1) if task == "shifted" else 4
        return dict(
            n_chans=self.by_subject[self.subjects[0]][0].eeg.shape[0]
            if self.by_subject else 32,
            n_bands=self.spec.n_bands, n_speakers=N_SPEAKERS,
            gaze_dim=GAZE_DIM, imu_dim=IMU_DIM, video_dim=VIDEO_DIM,
            gaze_traj_dim=2 * int(getattr(self.spec, "gaze_traj_len", 16) or 0),
            win_len=self.spec.win_len, sr=self.spec.sr,
            task_type=task, n_candidates=n_cand, chance=1.0 / n_cand,
            audio_feats=bool(getattr(self.spec, "audio_feats", False)),
            w2v_dim=int(getattr(self.spec, "w2v_pca_dim", 64)), sem_dim=3,
        )

    def _view(self, records: List[TrialRecord], name: str) -> AADView:
        return AADView(records, _windows_for(records, self.spec), self.spec, name)

    # -- protocols -----------------------------------------------------------
    def splits(self, protocol: str) -> Iterator[Tuple[SplitDesc, AADView, AADView]]:
        if protocol == "within":
            yield from self._within()
        elif protocol == "loso":
            yield from self._loso()
        elif protocol == "pooled":
            yield from self._pooled()
        else:
            raise ValueError(f"unknown protocol {protocol!r}")

    def _pooled(self, test_frac: float = 0.2):
        """Cross-subject POOLED training. Each subject's trials are split
        chronologically (early = train, late = test) so there is NO leakage and no
        look-ahead; the train trials of ALL subjects are POOLED into one big training
        set (~n_subjects x more data), and likewise the held-out late trials are
        pooled for test. One model trained on everyone -> stronger encoder for a faint,
        data-starved signal. Per-subject accuracy is broken out by the model's eval."""
        train_recs: List[TrialRecord] = []
        test_recs: List[TrialRecord] = []
        for s in self.subjects:
            recs = self.by_subject[s]
            order = np.argsort([r.trial_k for r in recs])      # chronological
            cut = int(round(len(order) * (1.0 - test_frac)))
            for j, i in enumerate(order):
                (train_recs if j < cut else test_recs).append(recs[i])
        if not train_recs or not test_recs:
            return
        desc = SplitDesc("pooled_all", "pooled", None, None)
        yield desc, self._view(train_recs, desc.name + "_tr"), \
            self._view(test_recs, desc.name + "_te")

    def _within(self):
        """Within-subject CV.

        Splits are at the TRIAL level (trials are atomic recordings), so windows
        -- even at 50% overlap -- never straddle the train/test boundary. Trials
        are ordered chronologically (by trial_k = recording order). Modes:
          * ``chrono_forward`` (default): forward-chaining / expanding window --
            train is always STRICTLY earlier than test, so there is no look-ahead.
          * ``chrono_blocked``: contiguous held-out blocks (time-ordered, but a
            middle fold's train contains later trials).
          * ``random``: shuffled k-fold (kept for ablation only).
        """
        for s in self.subjects:
            recs = self.by_subject[s]
            n = len(recs)
            if n < self.n_folds + 1:
                continue
            order = np.argsort([r.trial_k for r in recs])     # chronological
            for f, (tr_ids, te_ids) in enumerate(self._fold_indices(order)):
                train_recs = [recs[i] for i in tr_ids]
                test_recs = [recs[i] for i in te_ids]
                if not train_recs or not test_recs:
                    continue
                desc = SplitDesc(f"within_s{s}_fold{f}", "within", s, f)
                yield desc, self._view(train_recs, desc.name + "_tr"), \
                    self._view(test_recs, desc.name + "_te")

    def _fold_indices(self, order: np.ndarray):
        """Yield (train_idx, test_idx) over the chronologically-ordered ``order``."""
        n, k = len(order), self.n_folds
        if self.split_mode == "random":
            rng = np.random.default_rng(self.seed)
            for te in np.array_split(rng.permutation(order), k):
                te_set = set(te.tolist())
                yield [i for i in order if i not in te_set], list(te)
            return
        blocks = np.array_split(order, k + 1)                 # k+1 contiguous time blocks
        if self.split_mode == "chrono_forward":
            # fold f: train = blocks[0..f], test = block[f+1]  (no look-ahead)
            for f in range(k):
                tr = np.concatenate(blocks[: f + 1])
                yield list(tr), list(blocks[f + 1])
        else:  # chrono_blocked
            allb = np.array_split(order, k)
            for f in range(k):
                te = allb[f]
                te_set = set(te.tolist())
                yield [i for i in order if i not in te_set], list(te)

    def _loso(self):
        if len(self.subjects) < 2:
            return
        for test_s in self.subjects:
            train_recs: List[TrialRecord] = []
            for s in self.subjects:
                if s != test_s:
                    train_recs.extend(self.by_subject[s])
            test_recs = self.by_subject[test_s]
            desc = SplitDesc(f"loso_test_s{test_s}", "loso", test_s, None)
            yield desc, self._view(train_recs, desc.name + "_tr"), \
                self._view(test_recs, desc.name + "_te")
