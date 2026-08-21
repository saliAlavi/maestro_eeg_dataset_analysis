# Improving the strict-regime four-way accuracy — what helps, what doesn't

All under the **strict** protocol: content-disjoint splits (a `trial_k` in test never appears in
train/val), four real talkers (chance 0.25), and the **EEG-shuffle null as guardrail** — a lever
only counts if accuracy rises *over its own null* (the margin), because raw accuracy can be
inflated by the acoustic-marking floor. Base decoder: VLAAI 28-band backward + match-mismatch
margin, 5-seed reconstruction ensemble. 16 subjects unless noted.

## Confirmed baseline (5 s operating point)
| protocol | 4-way | null | margin (neural) |
|---|---|---|---|
| within | 0.302 | 0.257 | +0.045 (p=7e-5) |
| LOSO | 0.337–0.344 | 0.264–0.276 | +0.069–0.073 (p<1e-5) |

## Levers tried
| lever | result | verdict |
|---|---|---|
| **Decision-window integration** (train 5 s, eval longer) | LOSO 0.34→**0.47**, within 0.30→**0.39** at whole-trial; null flat ~0.26 | **the dominant lever** — accuracy integrates the weak signal over time |
| **Onset / edge envelope target** | within Δmargin **−0.026** (p=0.99); LOSO −0.007 (n.s.) | ✗ no help (a single-subject pilot mislead; refuted at 16 subj) |
| **Env + onset (56-band)** | worse than env | ✗ dilution |
| **Distractor hard-negatives** (speakers 5–6) | ≤ baseline (env+hardneg −0.012; both+hardneg went below null) | ✗ no help |
| **K-shot subject calibration** (LOSO + fine-tune on K held-out-subject trials) | K=20: margin +0.084 vs +0.069 at K=0 → **+0.015 (p=0.03)**, saturating; null stays ~0.26 | ~ small but real, marginally significant |

## Honest assessment
The corpus's envelope-tracking signal is genuinely weak, so **envelope-feature engineering has
plateaued**: onset targets and distractor hard-negatives do not help across subjects, and per-
subject calibration adds only ~+0.015 margin (~20 shots). The two things that reliably raise real
accuracy are **(i) integrating over a longer decision window** — the strict-regime ceiling is
**~0.47 four-way LOSO / ~0.39 within at the whole trial**, over a flat ~0.26 null — and, more
modestly, **(ii) a short per-subject calibration**.

Throughout, the EEG-shuffle null stayed at ~0.25–0.26 for every configuration: unlike the GitHub
model (whose accuracy *is* its null), every accuracy reported here is genuinely neural.

## Not yet tried (candidate next levers)
- **CCA / stimulus-informed spatial-filter (GEVD) front-end** — the classical strong AAD decoder;
  a genuinely different model that could denoise the EEG before reconstruction (uncertain gain).
- **Reliability-weighted band fusion** and **trial-level score pooling** (cheap, small expected).
- **Combined best**: VLAAI 28-band + margin + whole-trial window + 20-shot calibration.
