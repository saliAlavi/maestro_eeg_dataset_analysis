"""Attended-envelope reconstruction (Pearson r) — mirrors github
train_reconstruction.py / train_rec_all.py. Sweeps the 7 eeg/gaze/imu modality
combinations (linear backward model). --modes is ignored (combos are fixed).
Method/protocol via CLI; see _expcli.py. Default = our leakage-safe control.
"""
from _expcli import main

if __name__ == "__main__":
    main("reconstruction")
