"""Hemisphere (attended L/R, chance 0.5) — mirrors github train_hemisphere.py.
Audio is grouped into 2 streams (left = mean(S1,S2), right = mean(S3,S4)).
Method/protocol via CLI; see _expcli.py. Default = our leakage-safe control.
"""
from _expcli import main

if __name__ == "__main__":
    main("hemisphere")
