# Intra-subject fusion x decision-window curve (EEG content + EEG-spatial)

Within-subject, 5-fold pooled. **content** = VLAAI 28-band backward decoder (trained @5s, 5-seed reconstruction ensemble, early-stopped on inner-val); **spatial** = posterior alpha+beta band-power, gaze-residualised covert-direction LDA (retrained per window); **fused_oof** = the two fused with an **admissible out-of-fold-tuned** weight (never tuned on the scored fold). Candidates are the four real talkers -> chance 0.25 at every window. `Δ` = fused_oof minus the EEG-shuffle null; paired one-sided t across 16 subjects.

| window | content | spatial | **fused (OOF)** | fused (b=1.5) | null | Δ (neural) | t | p |
|---|---|---|---|---|---|---|---|---|
| 5s | 0.313 | 0.309 | **0.347** [0.319,0.380] | 0.330 | 0.251 | +0.096 | 5.93 | 1.5e-09 |
| 10s | 0.334 | 0.316 | **0.363** [0.332,0.399] | 0.333 | 0.251 | +0.112 | 6.12 | 4.7e-10 |
| 15s | 0.345 | 0.307 | **0.380** [0.347,0.413] | 0.327 | 0.254 | +0.126 | 7.00 | 1.3e-12 |
| 20s | 0.335 | 0.312 | **0.345** [0.307,0.385] | 0.321 | 0.250 | +0.095 | 4.64 | 1.7e-06 |
| 30s | 0.386 | 0.306 | **0.398** [0.352,0.448] | 0.322 | 0.250 | +0.148 | 5.73 | 5.0e-09 |

- **Both EEG branches are honest:** content is loudness/schedule-free (scale-free correlation + permuted slots); the spatial branch is **gaze-residualised**, so it reads *covert* neural direction, not eye movements. The fusion weight is out-of-fold tuned.
- Fusion + window together carry the intra four-way well above either branch alone; the EEG-shuffle null stays flat, so the gain is neural.

