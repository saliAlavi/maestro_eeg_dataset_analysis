"""maestro-loader — programmatic access to the maestro-eeg-dataset.

Public API
----------
    load_aad(...)        # dataset of perfectly cross-modal-aligned segments,
                         # or (train, test) when split=... is given
    get_dataloaders(...) # PyTorch (train, test) loaders for LOSO / intra splits
    DatasetPaths         # local-or-HF path resolver
    make_split           # trial-disjoint (subject, trial) split units
    assert_trial_disjoint

Every split is trial-disjoint by construction (a trial is never divided across
train and test) in both the LOSO and the intra-subject setting.
"""
from maestro_loader.loader import load_aad, get_dataloaders, MaestroDataset
from maestro_loader.paths import DatasetPaths, DEFAULT_REPO_ID
from maestro_loader.splits import make_split, assert_trial_disjoint
from maestro_loader.align import load_aligned_trial, AlignedTrial

__all__ = [
    "load_aad", "get_dataloaders", "MaestroDataset",
    "DatasetPaths", "DEFAULT_REPO_ID",
    "make_split", "assert_trial_disjoint",
    "load_aligned_trial", "AlignedTrial",
]
__version__ = "0.2.0"
