# Adjudication: is the GitHub model's strict-LOSO 4-class accuracy neural?

Under our strict, leakage-safe protocol (subject-disjoint LOSO, real held-out
test, loudness-matched + per-window-permuted candidates) the ported GitHub
`AADModel` reaches **0.356** four-way (chance 0.25) on EEG — marginally above our
own backward-decoder baseline (0.335). The obvious question: is that edge neural,
or is the model's *learned audio encoder* reading the attended talker's residual
acoustic marking? We answered it with the same controls we apply to our own
baseline, run eval-only on the saved strict-LOSO checkpoints (`adjudicate_gh.py`).

## Result — the accuracy is **not** neural

| quantity | GitHub AADModel (strict LOSO, EEG) | our baseline (VLAAI backward) |
|---|---|---|
| real 4-way | 0.356 | 0.335 |
| **EEG-shuffle null** | **0.356** | 0.25–0.265 |
| real − null (paired) | **−0.000**, $t=-0.79$, $p=0.78$ (n.s.) | **+0.07**, $t{=}7.0$, $p{<}10^{-6}$ |
| causal-lag margin | +0.002 ($p=0.06$, n.s.) | subject-variable |

The EEG-shuffle null **equals the real accuracy for every one of the 16 subjects**
(e.g. s1 0.330/0.330, s6 0.408/0.408, s13 0.414/0.414). Permuting the EEG across
trials — so the brain signal no longer corresponds to the trial's candidates —
does **not** change the decision at all. The model is therefore **not using the
EEG**: it decodes the attended talker entirely from the candidate audio.

## Why the model can do this (and our baseline cannot)

The `AADModel` has a **learned audio encoder** and a **learned similarity head**:
each candidate envelope is passed through its own dilated CNN into a common space,
and a learned linear projection scores it. This gives the model an **audio-only
path to the label**: the attended talker is acoustically *marked* (enhanced to be
attendable), and even after source loudness-matching a residual envelope-shape
signature remains, which the learned audio encoder detects — independently of the
EEG. The per-window candidate permutation does not stop this, because it is the
*intrinsic* audio of the attended talker that is distinguishable, not its position.

Our baseline cannot launder this confound: it decides by correlating an
**EEG reconstruction** against the **raw** candidate envelope. Shuffle the EEG and
the reconstruction no longer correlates with anything, so accuracy collapses to
0.25. That is exactly why our baseline's null is 0.25–0.265 and its margin is
significant, while the GitHub model's null is its own accuracy.

## Consequence for the benchmark

Strict subject- and content-disjoint splits are **necessary but not sufficient**.
A model with a learned stimulus encoder can convert an admissible split back into
an inadmissible result by reading the stimulus itself. The **EEG-shuffle null is
the control that catches it** — and the GitHub model was never evaluated against
one. Its 0.356 under our strict protocol is the **audio-only floor**, not brain
decoding; its published 0.586 adds leaky methodology and uncontrolled data on top
of that floor (see `DATA_PROCESSING_REPORT.md`).

Bottom line: the GitHub model's "edge" over our baseline is not a better readout of
the brain — it is the acoustic-marking confound, made visible only by a null the
repo does not run. Numbers: `results/adjudicate/`.
