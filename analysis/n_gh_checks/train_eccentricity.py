"""Eccentricity (inner/outer, chance 0.5) — mirrors github train_eccentricity.py.
Audio grouped into 2 streams (inner = mean(S2,S3), outer = mean(S1,S4)).
Method/protocol via CLI; see _expcli.py. Default = our leakage-safe control.
"""
from _expcli import main

if __name__ == "__main__":
    main("eccentricity")
