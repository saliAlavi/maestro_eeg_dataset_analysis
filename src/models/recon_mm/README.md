# recon_mm — reconstruction-driven match-mismatch (EEG + audio)

## What it is
An intra-subject auditory-attention decoder that, from a 5 s window of ~10 Hz
band-limited EEG, **reconstructs the attended speech band-envelope** and then
**classifies which of the presented sources is attended** by correlating that
reconstruction against each candidate envelope. One shared decoder, two jointly
trained heads:

- **Reconstruction:** EEG → attended 28-band envelope, loss `MSE + (1 − corr)`.
- **Classification:** cross-entropy over per-candidate Pearson-correlation scores
  (the candidate the reconstruction tracks best is the attended speaker).

## Motivation / intuition — why this design
Two prior mistakes on this corpus led here:

1. **The loudness confound.** With native levels the attended talker is +3–18 dB
   louder, so a classifier (even `eegnet_mm` on the `speaker` task) can hit ~1.0
   from audio energy alone — not EEG. Fix: candidates are loudness-equalised by
   the **documented table power** (`Device-k {Left,Right} Power`), and scoring is
   **Pearson correlation**, which is amplitude-invariant. After both, the only
   thing that separates candidates is *temporal envelope tracking*, which must
   come from EEG.
2. **Whole-audio reconstruction is infeasible / unnecessary.** We reconstruct the
   low-dimensional **band envelope**, the quantity EEG actually tracks, not the
   waveform or full spectrogram.

## Why reconstruction *and* classification (not just a learned key match)
`eegnet_mm` learns an opaque EEG-query/envelope-key dot product. `recon_mm`
instead makes the decision *interpretable and physically grounded*: it produces
an explicit reconstructed envelope (inspectable, comparable to the classical
backward model — the field baseline ~0.7) and derives the class from it. This
unifies the classical backward stimulus-reconstruction model with neural
match-mismatch end to end, and is the EEG-only control for `recon_mm_gaze`.

## Leakage control
Intra-subject, **trial-level** `chrono_forward` CV: train trials are strictly
earlier than test trials, so 50%-overlapping windows can never straddle the
split (windows are atomic to a trial; trials are atomic to a fold).

## Config
`configs/model/recon_mm.yaml` — `d_model`, `dropout`, `w_cls`, `w_recon`, `train.*`.
Data: `configs/data/aad_recon.yaml` (`mm_task=speaker`, `audio_norm=table_power`,
`lp_hz=10`). CPU smoke test: `python -m src.main mode=selftest model=recon_mm`.
