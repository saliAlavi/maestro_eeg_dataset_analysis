# riemann_tangent — Riemannian spatial-covariance AAD (classical, strong)

## What it is
Per decision window: EEG spatial covariance → tangent-space projection on the
SPD manifold → standardise → multinomial logistic regression over the four
attended speakers. A pure "where is attention" spatial decoder.

## Motivation / intuition
Spatial auditory attention leaves covariance-structure fingerprints (lateralised
alpha/beta power, topographic shifts). Riemannian geometry is the right metric
for SPD covariance matrices and, across BCI/AAD benchmarks, tangent-space +
linear classifiers are remarkably strong, robust to subject variability, and
work at **short** windows where envelope decoders struggle — with almost no
tuning.

## Why we chose it (what mistakes led here)
The backward envelope decoder (`linear_backward`) is "what"-based and weak at
short windows. Earlier spectral-power baselines (~0.72) under-used the spatial
*structure* of the montage. Riemannian features make that structure first-class
and give us a near-SOTA classical number for free on CPU — the right yardstick
before spending GPU on neural nets. It also decodes attended speaker directly
(4-class), unlike the reconstruction decoder.

## Caveat (the project's central tension)
Spatial decoders can ride on overt orienting: head/eye turns toward the attended
side change the EEG covariance too. So this model's accuracy is reported
*alongside* the gaze/IMU baselines, not as proof of covert neural attention.

## Key hyperparameters (`configs/model/riemann_tangent.yaml`)
- `estimator` covariance estimator (`oas` shrinkage by default)
- `C` logistic-regression inverse regularisation
