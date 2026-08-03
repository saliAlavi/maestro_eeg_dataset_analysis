"""4-class attended speaker (chance 0.25) — mirrors github train_pooled.py.
Candidates are the 4 talkers' loudness-matched envelopes, permuted per window.
With --data-method github --protocol pooled this is the repo's pooled experiment;
with --data-method github --protocol loso it mirrors train_loso_hot.py.
Method/protocol via CLI; see _expcli.py. Default = our leakage-safe control.
"""
from _expcli import main

if __name__ == "__main__":
    main("speaker4")
