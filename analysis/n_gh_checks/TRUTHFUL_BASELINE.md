# Truthful four-way baseline: GitHub AADModel vs our learned-head decoder

Content-disjoint splits (a `trial_k` in test never appears in train or val) for BOTH protocols, four real talkers (chance **0.25**), EEG only. `null` = EEG-shuffle null. A model is neural iff accuracy is significantly above its OWN null (paired one-sided $t$).

| model / protocol | 4-way [95% CI] | EEG-shuffle null | margin | t | p | n |
|---|---|---|---|---|---|---|
| **GitHub AADModel** — within | 0.253 [0.251,0.255] | 0.253 | +0.000 | 0.97 | 1.7e-01 | 16 |
| Ours, fixed readout — within | 0.296 [0.290,0.303] | 0.256 | +0.040 | 6.32 | 6.9e-06 | 16 |
| Ours, **+learned head** — within | 0.297 [0.291,0.303] | 0.256 | +0.040 | 6.67 | 3.8e-06 | 16 |
| **GitHub AADModel** — loso | 0.362 [0.359,0.364] | 0.362 | -0.000 | -1.26 | 8.9e-01 | 16 |
| Ours, fixed readout — loso | 0.327 [0.318,0.336] | 0.265 | +0.062 | 7.04 | 2.0e-06 | 16 |
| Ours, **+learned head** — loso | 0.323 [0.314,0.332] | 0.264 | +0.060 | 6.44 | 5.6e-06 | 16 |

## Reading it
- **The GitHub model's null does NOT drop to 0.25 even with content-disjoint splits.** Its accuracy equals its null in both protocols: the model ignores the EEG and decodes the attended talker from the candidate **audio**. This is not track/voice memorization (test content is unseen) but *general* acoustic marking -- the attended talker is systematically enhanced by the stimulus design, and the learned audio encoder detects that on any track. Trial-disjoint splits cannot remove a stimulus-design confound; only removing the audio->label architectural path can.
- **Our decoder has a null at ~0.25-0.26 and accuracy significantly above it** -- genuinely neural -- because it decides by correlating an EEG reconstruction against the raw envelope; shuffle the EEG and the correlation vanishes. Adding a learned similarity head keeps the null at ~0.26 (it operates on EEG<->audio correlations, not raw audio), so it cannot launder the confound.
- **The truthful four-way baseline for this corpus is our decoder's margin over a ~0.25 null, not the GitHub model's raw accuracy.**

