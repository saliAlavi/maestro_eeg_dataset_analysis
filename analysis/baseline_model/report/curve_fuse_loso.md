# Intra→LOSO fusion x decision-window curve (EEG content + EEG-spatial), cross-subject

Held-out TEST subject, train on the other 15. **content** = VLAAI 28-band (5-seed, margin, content-disjoint early-stop); **spatial** = gaze-residualised covert-direction LDA. **fused (OOF)** = fused with a weight tuned on the OTHER 15 subjects (admissible). Chance 0.25 every window. `Δ` = fused_oof − null; paired one-sided t across 16 subjects.

| window | content | spatial | **fused (OOF)** | fused (b=1.5) | null | Δ (neural) | t | p |
|---|---|---|---|---|---|---|---|---|
| 5s | 0.360 | 0.313 | **0.360** [0.339,0.382] | 0.335 | 0.250 | +0.111 | 10.16 | 0.0e+00 |
| 10s | 0.402 | 0.321 | **0.402** [0.366,0.441] | 0.343 | 0.251 | +0.150 | 7.91 | 1.2e-15 |
| 15s | 0.416 | 0.321 | **0.416** [0.374,0.460] | 0.343 | 0.251 | +0.166 | 7.44 | 4.9e-14 |
| 20s | 0.430 | 0.339 | **0.430** [0.382,0.481] | 0.358 | 0.251 | +0.179 | 6.91 | 2.3e-12 |
| 30s | 0.466 | 0.321 | **0.466** [0.411,0.518] | 0.343 | 0.251 | +0.215 | 7.67 | 8.3e-15 |

- Fusion weight is tuned across subjects (never on the scored subject); the spatial branch is gaze-residualised (covert neural direction). Null stays flat ~0.25.

