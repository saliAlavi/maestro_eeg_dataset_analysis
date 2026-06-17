# deep_match — lag-robust deep match-mismatch source identification

## What it is
Source identification framed directly as **"which candidate audio envelope best
matches the EEG"**, with a deep model designed to fix the two weaknesses that sank
every prior attempt (classical CCA aad_v2/v3 ≈ 0.50; recon_mm ≈ chance).

- **EEG path:** raw (re-referenced) EEG → in-model **BatchNorm + learned spatial
  filter (differentiable CSP)** → per-component temporal filter → K spatial-temporal
  components. Full time resolution (no 8× pooling bottleneck).
- **Envelope path:** each candidate envelope → K matched components.
- **Lag-robust matching:** score each candidate by a differentiable
  cross-correlation **soft-max-pooled over a ±500 ms lag window**, averaged over
  components. Cross-entropy over the 6 candidates (5/6 masked).

## Motivation / intuition — the two mistakes this fixes
1. **Fixed-lag matching.** CCA/recon_mm align EEG↔envelope at one lag. This corpus
   has per-trial audio↔EEG **jitter** (software timestamps; peak lags scattered
   std ~600 ms — `analysis/scripts/diag_lag_jitter.py`). Matching at each
   candidate's *best lag* is the untested lever that jitter demands.
2. **Per-channel z-scoring** (recon_mm) erases the relative channel amplitudes a
   spatial/CSP filter needs. Here EEG is **not** per-channel z-scored
   (`norm_eeg=false`); whitening is done in-model by BatchNorm + the learned spatial
   filter — answering "is per-channel z-score a good idea?" with: not before a
   spatial filter.

Honest expectation: proper CCA already failed at fixed lag, so this is a real test
of whether **lag-robustness alone** recovers the signal. Either it beats chance
(jitter was the blocker) or it confirms the tracking signal is absent.

## Data
Reuses the `aad_recon` cache (EEG 1–10 Hz, table-power-equalised candidates) via
`configs/data/aad_match.yaml` with `norm_eeg=false` — **no re-caching** (norm is
materialise-time). Intra-subject, trial-level `chrono_forward` CV.

## Config
`configs/model/deep_match.yaml` (`K`, `ksize`, `max_lag`, `lag_temp`).
Smoke: `python -m src.main mode=selftest model=deep_match data=aad_match`.
