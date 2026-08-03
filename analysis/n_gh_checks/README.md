# n_gh_checks — leakage-safe reproduction of the ASPIRE-OSU/MAESTRO benchmark

This folder re-runs the **github MAESTRO benchmark experiments** (the public
`github.com/ASPIRE-OSU/MAESTRO` release code) using **our own leakage-safe
dataloader** (`src/data`) instead of the repo's `dataloader.py`. The model
*architectures* are ported byte-faithfully; everything that could inflate the
numbers is fixed in the data + training layers. The point is an honest
apples-to-apples answer to "what happens to the headline numbers once the
leakage is removed."

## What is reproduced (models — faithful)
- `gh_models.py`: `DilatedEncoder`, `AADModel` (4-class + the 2-stream
  hemisphere/eccentricity `spatial` variant), `LinearModel` + `pearson_loss`
  (reconstruction), `LateFusionCombiner` — all verbatim from the repo's
  `model_classification.py` / `model_spatial.py` / `model_reconstruction.py` /
  `late_fusion.py`. Same training recipe: Adam(1e-4), ReduceLROnPlateau(max,
  .5, patience 5), grad-clip 1.0, label-smoothing 0.1, batch 32, ≤50 epochs,
  early-stop patience 10.

## What is fixed (data + protocol — the leakage removal)
1. **5 s windows @ 0.5 overlap** (repo: 30 s, no overlap) — `WindowSpec`.
2. **Real held-out test set.** Early stopping / checkpoint selection use an
   **inner-val** split carved from *train* (`gh_data._carve_val_*`); the reported
   number is a single evaluation of the best-val checkpoint on a test split that
   never touched training. (Repo reports max-over-epochs on the same fold it early
   -stops on — val == test.)
3. **Trial-level splits** so overlapping windows never straddle train/test.
4. **Subject-disjoint LOSO** and **content-disjoint within-subject chrono-forward**
   CV (train strictly precedes test).
5. **Loudness-matched envelopes** (`audio_norm="table_power"`, baked into the
   cache) — removes the +3..18 dB attended-loudness confound that let the repo's
   4-class decode from audio energy. 4-class candidates are permuted per window
   (deterministic seed) so no fixed slot→direction shortcut survives.
6. **Per-device perfect audio alignment** (`perfect_align`) and **full-band EEG**
   (`elp0` cache) so alpha/beta lateralisation survives for hemisphere/eccentricity.

## Scope
- Tasks: **hemisphere** (L/R, chance .5), **eccentricity** (inner/outer, .5),
  **speaker4** (4-class attended, .25), **reconstruction** (Pearson r), plus
  **late-fusion** and the **SNR** / **error-complementarity** analyses.
- Modalities: **eeg, gaze, imu** (+ fusions). **Video is out of scope** — our
  dataloader does not materialise optical-flow video (`present_video=False`).
  Gaze is 3-ch here ([x, y, pupil]) vs the repo's 6-ch.
- Protocols: **within-subject** (chrono-forward 5-fold) + **LOSO**.

## Files
| file | role |
|---|---|
| `gh_models.py` | faithful model ports |
| `gh_data.py` | adapter over `src.data`: windows (5s/0.5 **or** 30s whole-trial), proper **and** github splits, task labels |
| `gh_core.py` | shared engine with the `--data-method` switch (proper vs github) |
| `train_hemisphere.py`, `train_eccentricity.py`, `train_pooled.py`, `train_reconstruction.py` | **per-experiment scripts (github-repo format)**, each with `--data-method {proper,github}` |
| `late_fusion.py` | learned softmax fusion, same `--data-method` toggle |
| `train_gh.py` | original batch driver (proper only; produced the first results) |
| `analyze_snr_gh.py`, `analyze_error_gh.py` | downstream analyses |
| `aggregate.py` | collect JSONs → `results/summary.md` + CSVs, vs repo numbers |
| `slurm/` | proper: `within`/`loso`/`post`; github: `github_pooled` (array 1-4), `github_loso` (array 1-16) |

## Two data methods (A/B) — reproducing *both* columns of the table
Every per-experiment script takes `--data-method`. It switches the **entire
train/val methodology** while holding the underlying (controlled) data fixed, so
the delta between the two runs is exactly the leakage.

| `--data-method` | windows | split | selection |
|---|---|---|---|
| **proper** (default) | 5 s @ 0.5 overlap | within chrono-fwd / subject-disjoint LOSO | inner-val early stop → **held-out test** |
| **github** | 30 s whole-trial | pooled StratifiedKFold (subjects *not* held out) / github-LOSO | **val == test, max-over-epochs** |

**Which command makes which table cell:**
```bash
# ── our leakage-safe numbers ("ours" columns) ───────────────────────────────
python train_hemisphere.py   --data-method proper --protocol within --subject K   # K=1..16
python train_hemisphere.py   --data-method proper --protocol loso   --subject K
python train_pooled.py       --data-method proper --protocol loso   --subject K   # 4-class
python train_reconstruction.py --data-method proper --protocol within --subject K
# ── the github repo's methodology ("gh (repo)" column) ──────────────────────
python train_hemisphere.py   --data-method github --protocol pooled              # all subjects
python train_eccentricity.py --data-method github --protocol pooled
python train_pooled.py       --data-method github --protocol pooled              # 4-class pooled
python train_pooled.py       --data-method github --protocol loso   --subject K  # train_loso_hot
python late_fusion.py        --data-method github --protocol pooled
```
Results are tagged by method in scratch: `results/{within,loso}` (proper) vs
`results/{gh_pooled,gh_loso}` (github); `aggregate.py` folds them all into one table.

> Note: the github method reproduces the repo's **evaluation methodology** on our
> controlled cache (loudness-matched envelopes, our EEG preprocessing), which
> isolates the leakage. Matching the repo's *absolute* numbers to the last digit
> would also require its raw-data preprocessing; the direction/magnitude of the
> inflation is reproduced here.

## Run
```bash
cd analysis/n_gh_checks
# 0) fast GPU validation
sbatch slurm/smoke_gpu.sbatch
# 1) training arrays
W=$(sbatch --parsable slurm/within.sbatch)
L=$(sbatch --parsable slurm/loso.sbatch)
# 2) fusion + analyses + aggregate, after both arrays finish
sbatch --dependency=afterok:$W:$L slurm/post.sbatch
```
Results land in `/fs/scratch/PAS2301/alialavi/projects/n_gh_checks/` (per-config
JSONs + checkpoints) and the summary is copied back to `results/` here.
Account `PAS2966` (fall back to `PAS2301` if `AssocGrpSubmitJobsLimit`), A100
partition `nextgen`.
