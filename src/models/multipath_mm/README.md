# multipath_mm — multi-path match-mismatch decoder with a frozen first stage

## What it is
A 4-class attended-talker decoder built from **independent per-path matchers** whose
scores are combined by a **learned softmax mixture trained on detached (frozen) branch
outputs**. Candidates are the permuted real talkers (`mm_task=match`), so there is no
candidate-slot→direction shortcut, and with trial-disjoint within-subject splits the
frozen audio embeddings cannot leak any audio→label prior.

## Paths (enable any subset via `paths:`)
| path | EEG band fed in | matches (frozen) | neuroscience |
|------|-----------------|------------------|--------------|
| `env` | delta-theta 1–9 Hz | 28-band gammatone envelope | cortical envelope tracking |
| `w2v` | broadband | HuBERT layer-9 (PCA-64) | auditory / phonetic tracking |
| `sem` | delta-theta 1–9 Hz | GPT-2 surprisal/entropy/onset | semantic (N400-scale) |
| `dir` | alpha 8–14 Hz (CSP) | physical position → cand slots | spatial-attention α lateralisation |

Each EEG band is a **fixed FIR filter** (a physiological prior, not learned). Content paths
reconstruct their target representation and score candidates by Pearson correlation. The
directional path decodes the **physical** attended position from alpha-band CSP power, then
re-indexes those logits into candidate-slot order via `cand_pos` (the per-window permutation
exposed by the data layer) — the only path allowed to use direction.

## Two-stage training in one job
- **Stage 1 (matchers):** each path is trained by its own cross-entropy (`w_aux`) plus, for
  content paths, an EEG→attended-representation reconstruction loss (`w_recon`).
- **Stage 2 (fusion):** the softmax mixture `mix_{path}` is trained by a cross-entropy on the
  **detached** per-path scores (`w_fuse`). Because the inputs are detached, fusion gradients
  never reach stage 1 — realising "first stage frozen" without a separate phase. At inference
  the same mixture combines the non-detached scores.

`mix_{path}` and `acc_{path}` are logged every step, so the run directly reports which path
the EEG actually uses and how much each contributes.

## Why (lineage)
`recon_mm` matched EEG only to the envelope (jitter-fragile → near chance here). `recon_mix`
added HuBERT/GPT-2 content spaces with a learned mixture but used **one shared EEG encoder**
for every space and had **no directional path**. `multipath_mm` fixes both: per-band EEG
front-ends matched to each path's physiology, and an explicit alpha-lateralisation directional
path, so we can cleanly attribute 4-class performance to covert content tracking vs spatial
orienting (run the `dir` path on motion-residualised EEG to test the latter honestly).

## Headline metric
Within-subject, trial-disjoint 5-fold, 4-class attended talker (chance 0.25). Honest bar to
beat: EEG-spectral 0.448 and gaze-only 0.547 (both within-5fold @30 s, in
`results/evaluation_protocol/protocol_baseline_results.parquet`). Note: hemisphere/inner-outer
metrics are not meaningful under the permuted-candidate match task (the label is a slot, not a
physical speaker); this model reports 4-class accuracy.

## Run
```bash
# CPU smoke (synthetic): python -m src.main mode=selftest model=multipath_mm data=aad_multipath
python -m src.main mode=train model=multipath_mm data=aad_multipath runner.protocols=[within]
# single-path ablation:
python -m src.main mode=train model=multipath_mm data=aad_multipath \
    runner.protocols=[within] model.paths=[dir] model.tag=-dir
```
