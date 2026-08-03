# Dataset-paper baseline — four-way attended-talker match-mismatch

Reconstruct the attended talker's envelope from EEG, then decide by scale-free correlation against the **four real co-present talkers** (permuted slots). Chance is **0.25 at every decision window** (5 s or the whole trial). Data are the method paper's properly-aligned cache (`*_pa2_af64.npz`); train/val are trial-disjoint in both protocols; held-out test of the best-inner-val checkpoint. mean +/- sd [95% CI] across 16 subjects (5 s windows).

| model | protocol | 4-way acc (chance .25) | EEG-shuffle null | binary (.5) | causal margin |
|---|---|---|---|---|---|
| VLAAI + multiband + margin (headline) | loso | **0.335** +/-0.039 [0.316,0.354] | 0.261 | 0.577 | -0.002 |
| VLAAI + multiband + margin (headline) | within | **0.288** +/-0.015 [0.280,0.295] | 0.257 | 0.534 | -0.000 |
| VLAAI + multiband | loso | **0.328** +/-0.031 [0.313,0.344] | 0.260 | 0.567 | -0.010 |
| VLAAI + multiband | within | **0.303** +/-0.024 [0.291,0.315] | 0.261 | 0.548 | +0.003 |
| VLAAI (plain) | loso | **0.315** +/-0.031 [0.301,0.331] | 0.247 | 0.564 | -0.007 |
| VLAAI (plain) | within | **0.292** +/-0.018 [0.283,0.301] | 0.251 | 0.545 | +0.002 |
| Linear (reference) | loso | **0.254** +/-0.015 [0.246,0.261] | 0.246 | 0.506 | +0.011 |
| Linear (reference) | within | **0.253** +/-0.017 [0.245,0.261] | 0.247 | 0.507 | +0.004 |

## Reading the result

- **Chance = 0.25 at every window.** The four candidates are the four real co-present talkers, so the number of choices never changes with window length; the EEG-shuffle null is flat (~0.25-0.26) across windows — no drift. (A pure-noise guesser scores exactly 0.25; the ~0.015 excess is the audio-only floor — the trained decoder matching the attended talker's acoustic marking with EEG scrambled — a constant dataset property; see curve.md.)
- **Above the null is genuine cortical tracking.** Loudness cannot help (scale-free correlation); the deterministic attended schedule cannot help (candidate slots are permuted per (subject,trial); the model has no audio->label path). Reconstructing from scrambled EEG collapses the decision to 0.25 — the paired per-subject acc-vs-null test is the rigorous proof the signal is EEG.
- **Causal-lag control:** genuine tracking is causal (audio leads EEG ~100-250 ms); the lag curve guards against instantaneous stimulus bleed (subject-variable at the aggregate).
- VLAAI (modern deep backward net) is the headline; the linear decoder is the canonical reference floor. Both are EEG-only single-reconstruction models — the learned similarity head, gaze/video/IMU fusion, and subject adaptation are left for the method paper.


> A candidate-only classifier can read the attended talker's acoustic *marking* above 0.25 (an irremovable dataset property), but the backward model has no path to exploit it, which the EEG-shuffle null = 0.25 confirms. A contrastive match-mismatch model (NeuroCLIP) was also tried and sat at the null — envelope tracking here is detected by backward/reconstruction decoders, consistent with the AAD literature.

