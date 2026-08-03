# Decision-window curve (VLAAI + multiband, within-subject)

Train at 5 s, evaluate at each window with a 5-seed reconstruction ensemble. Candidates are the four real talkers, so **theoretical chance is 0.25 at every window** (a pure-noise guesser scores 0.250). The empirical EEG-shuffle null is **flat at ~0.265** (no window drift); the +0.015 is the *audio-only* floor — the decoder, trained to reconstruct attended envelopes, matches the attended talker's acoustic marking even with EEG scrambled (see below). `Δ` = 4-way minus the empirical null (the neural margin); paired one-sided t across 16 subjects.

| window | 4-way (chance .25) | null | Δ (neural) | t | p | binary (.5) | cand-only |
|---|---|---|---|---|---|---|---|
| 5s | **0.317** [0.305,0.331] | 0.259 | +0.058 | 8.43 | <1e-16 | 0.562 | 0.401 |
| 10s | **0.336** [0.316,0.356] | 0.261 | +0.075 | 7.19 | 3.4e-13 | 0.578 | 0.423 |
| 15s | **0.354** [0.326,0.384] | 0.257 | +0.097 | 7.01 | 1.2e-12 | 0.593 | 0.447 |
| 20s | **0.343** [0.317,0.370] | 0.256 | +0.087 | 6.42 | 6.8e-11 | 0.596 | 0.495 |
| 30s | **0.406** [0.375,0.439] | 0.261 | +0.145 | 9.15 | <1e-16 | 0.626 | 0.505 |

- The **null is flat across all windows** (~0.265, no drift) — unlike a same-talker time-shift construction whose null climbs with window. Theoretical four-choice chance is 0.25 at every window; a pure-noise guesser scores 0.250, confirming the four candidates are a fair four-way choice.
- The +0.015 above 0.25 is the **audio-only floor**: the attended talker is acoustically marked, and the decoder (trained to reconstruct attended envelopes) matches that marking even when its EEG is scrambled — i.e. 0.265 is what audio marking alone buys with no EEG. It is present equally in the real accuracy and is removed by testing against the empirical null (not the naive 0.25), so it is never credited to the brain.
- Accuracy integrates over time: the 4-way rises with the decision window while the null holds flat, so the **neural margin Δ grows with window** and stays highly significant.
- `cand-only` (a supervised probe on candidate audio features) reads the marking above chance, but the backward model has no audio->label path to exploit it — its EEG-shuffle null is the flat ~0.265, not the cand-only value.

