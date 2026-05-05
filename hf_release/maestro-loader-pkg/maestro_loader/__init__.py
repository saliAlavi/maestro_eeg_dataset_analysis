"""maestro-loader — programmatic access to the maestro-eeg-dataset.

Public API:

    load_aad(...)   # main entry point — returns a torch/numpy/dict dataset
    DatasetPaths    # helper that resolves local-or-HF-cached paths
"""
from maestro_loader.loader import load_aad, DatasetPaths

__all__ = ["load_aad", "DatasetPaths"]
__version__ = "0.1.2"
