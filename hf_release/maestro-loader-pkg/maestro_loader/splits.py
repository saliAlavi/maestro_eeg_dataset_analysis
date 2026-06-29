"""Trial-disjoint train/test splits for LOSO and intra-subject protocols.

A *unit* is a ``(subject_id, trial_id)`` pair — the atomic, indivisible element
of every split. Because no trial is ever divided across train and test, every
split is **trial-disjoint by construction**: all windows cut from a trial land
on the same side. This holds in both settings:

* ``loso`` — leave-one-subject-out. Fold *f* tests on the held-out subject's
  trials; trains on every trial of all other subjects.
* ``intra`` — within-subject k-fold. Each selected subject's trials are
  partitioned into ``n_folds`` groups (chronological by default); fold *f* tests
  on group *f*, trains on the rest — independently per subject.

``make_split`` returns ``(train_units, test_units)``; ``assert_trial_disjoint``
verifies the guarantee.
"""
from __future__ import annotations

import numpy as np

Unit = tuple[str, str]


def _trial_num(tid: str) -> int:
    return int(tid.rsplit("_", 1)[1])


def kfold_groups(items: list, n_folds: int, scheme: str = "chrono",
                 seed: int = 0) -> list[list]:
    """Partition ``items`` into ``n_folds`` groups (chronological or shuffled)."""
    arr = list(items)
    if scheme == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(arr))
        arr = [arr[i] for i in idx]
    # contiguous blocks (chrono keeps trial order; random shuffled first)
    return [list(b) for b in np.array_split(np.array(arr, dtype=object), n_folds)]


def make_split(
    setting: str,
    fold: int,
    *,
    subjects: list[str],
    trials_by_subject: dict[str, list[str]],
    n_folds: int = 5,
    scheme: str = "chrono",
    seed: int = 0,
) -> tuple[list[Unit], list[Unit]]:
    """Return (train_units, test_units) for the requested fold."""
    subjects = sorted(subjects)
    if setting == "loso":
        if not 0 <= fold < len(subjects):
            raise ValueError(f"loso fold {fold} out of range 0..{len(subjects)-1}")
        test_s = subjects[fold]
        train, test = [], []
        for s in subjects:
            for t in trials_by_subject.get(s, []):
                (test if s == test_s else train).append((s, t))
        return train, test

    if setting in ("intra", "within"):
        if not 0 <= fold < n_folds:
            raise ValueError(f"intra fold {fold} out of range 0..{n_folds-1}")
        train, test = [], []
        for s in subjects:
            trs = sorted(trials_by_subject.get(s, []), key=_trial_num)
            groups = kfold_groups(trs, n_folds, scheme, seed)
            test_trs = set(groups[fold])
            for t in trs:
                (test if t in test_trs else train).append((s, t))
        return train, test

    raise ValueError(f"unknown setting {setting!r} (use 'loso' or 'intra')")


def assert_trial_disjoint(train: list[Unit], test: list[Unit]) -> None:
    """Raise if any (subject, trial) unit appears in both splits."""
    inter = set(train) & set(test)
    if inter:
        raise AssertionError(
            f"train/test share {len(inter)} (subject,trial) units, e.g. "
            f"{sorted(inter)[:3]} - split is NOT trial-disjoint")


def n_folds_for(setting: str, n_subjects: int, n_folds: int = 5) -> int:
    return n_subjects if setting == "loso" else n_folds
