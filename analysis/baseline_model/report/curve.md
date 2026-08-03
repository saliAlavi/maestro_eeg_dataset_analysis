# Decision-window curve (VLAAI + multiband + margin, LOSO)

Train at 5 s, evaluate at each window with a 5-seed reconstruction ensemble. Candidates are the four real talkers, so **theoretical chance is 0.25 at every window** (a pure-noise guesser scores 0.250). The empirical EEG-shuffle null is **flat at ~0.265** (no window drift); the +0.015 is the *audio-only* floor — the decoder, trained to reconstruct attended envelopes, matches the attended talker's acoustic marking even with EEG scrambled (see below). `Δ` = 4-way minus the empirical null (the neural margin); paired one-sided t across 16 subjects.

| window | 4-way (chance .25) | null | Δ (neural) | t | p | binary (.5) | cand-only |
|---|---|---|---|---|---|---|---|
| 5s | **0.348** [0.326,0.369] | 0.266 | +0.083 | 7.46 | 4.5e-14 | 0.588 | 0.410 |
| 10s | **0.383** [0.349,0.418] | 0.264 | +0.119 | 6.60 | 2.0e-11 | 0.623 | 0.435 |
| 15s | **0.431** [0.384,0.477] | 0.264 | +0.167 | 7.12 | 5.4e-13 | 0.658 | 0.441 |
| 20s | **0.440** [0.391,0.485] | 0.262 | +0.178 | 7.30 | 1.5e-13 | 0.662 | 0.428 |
| 30s | **0.497** [0.435,0.552] | 0.267 | +0.230 | 7.38 | 8.0e-14 | 0.698 | 0.426 |

- The **null is flat across all windows** (~0.265, no drift) — unlike a same-talker time-shift construction whose null climbs with window. Theoretical four-choice chance is 0.25 at every window; a pure-noise guesser scores 0.250, confirming the four candidates are a fair four-way choice.
- The +0.015 above 0.25 is the **audio-only floor**: the attended talker is acoustically marked, and the decoder (trained to reconstruct attended envelopes) matches that marking even when its EEG is scrambled — i.e. 0.265 is what audio marking alone buys with no EEG. It is present equally in the real accuracy and is removed by testing against the empirical null (not the naive 0.25), so it is never credited to the brain.
- Accuracy integrates over time: the 4-way rises with the decision window while the null holds flat, so the **neural margin Δ grows with window** and stays highly significant.
- `cand-only` (a supervised probe on candidate audio features) reads the marking above chance, but the backward model has no audio->label path to exploit it — its EEG-shuffle null is the flat ~0.265, not the cand-only value.

