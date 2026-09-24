# ICLR 2027 submission — *MERIT*

**MERIT** — *Modality Evidence Regularisation and Identifiable Training*.
Title: *MERIT: Making Every Modality Earn Its Contribution in Multimodal
Attention Decoding*.

> **PENDING: code rename.** The manuscript uses `merit` throughout, including the
> reproduction commands in Appendix A. The code package is still
> `src/models/merit/` with configs `configs/*/credit*.yaml` and run directories
> `multimodal_aad__credit__*`, because renaming while jobs are in flight would
> break the result-loading glob in `src/tools/make_macros.py`. Rename once the
> queue is empty, then re-run `make_macros` and `make_figures`.

The paper for MERIT (`src/models/merit/`, to be renamed). Every number in the manuscript is
generated from measured run outputs; nothing is typed by hand.

## Reproduce everything: `notebooks/merit_iclr2027_reproduce.ipynb`

[`notebooks/merit_iclr2027_reproduce.ipynb`](../../notebooks/merit_iclr2027_reproduce.ipynb)
rebuilds every figure, table and number in the manuscript, lists the training
jobs behind them (and can submit them), loads all 26 released checkpoints from
[`weights/merit/`](../../weights/merit/) with `strict=True`, and re-scores them
on their exact test folds. It runs from the repository alone — see the next
section.

## How numbers get into the paper

```
run  ──►  /fs/scratch/PAS2301/alialavi/projects/multimodal_aad__credit__<tag>__<stamp>/results.parquet
          │                                        │
          │      (snapshotted into the repo by python -m src.tools.snapshot_results)
          ▼                                        ▼
          docs/iclr2027/results/runs/<run>/results.parquet
          │
          ├─ python -m src.tools.make_macros --out docs/iclr2027/results_macros.tex
          │      emits  \defres{<tag>/<variant>/<protocol>/<metric>}{value}
          │      the manuscript writes  \R{streams/all/within/acc}
          │      a key with no measurement renders as a red ?? in the PDF
          │
          └─ python -m src.tools.make_figures --out docs/iclr2027/figs
```

Scratch is purged by OSC, so `make_macros`/`make_figures` read the live run
directories when they exist and the tracked snapshot `docs/iclr2027/results/`
otherwise (duplicate rows collapse by run name). After new runs finish, refresh
the snapshot with `python -m src.tools.snapshot_results` and commit it.

Inspect results from the console with `python -m src.tools.report [--tag hinge] [--streams]`.

## Build

```bash
python -m src.tools.make_macros --out docs/iclr2027/results_macros.tex
python -m src.tools.make_figures --out docs/iclr2027/figs
cd docs/iclr2027 && latexmk -pdf main.tex
```

## Experiments and their tags

| sbatch | tag(s) | table / figure |
|---|---|---|
| `merit_certify.sbatch` | — | Tab. audio-only probe, Tab. role separation |
| `merit_within.sbatch EXP=ladder_raw` | `ladder_raw` | ablation ladder, confounded candidates |
| `merit_within.sbatch EXP=ladder_qm` | `ladder_qm` | ablation ladder, matched candidates |
| `merit_within.sbatch EXP=streams` | `streams` | per-stream credit, within-subject |
| `merit_within.sbatch EXP=hinge` | `hinge` | **the freeloading experiment** |
| `merit_within.sbatch EXP=lag` | `lag_qm`, `lag_mm` | neural directionality |
| `merit_within.sbatch EXP=orientfree` | `mm3`, `mm2` | same-talker negatives |
| `merit_hinge2loso.sbatch`, `merit_v2.sbatch` | `v2hinge2`, `v2hinge2loso`, `v2onset`, … | **headline 2×2, LOSO, EEG-only** |
| `merit_loso.sbatch` (array 0–15) | `loso` | leave-one-listener-out |
| `merit_windows.sbatch` (array 0–4) | `w5`…`w30` | decision-window sweep |
| `merit_external*.sbatch` | `extkul*`, `extdtu*`, `extnju*` | public-corpus transfer |
| `merit_release.sbatch` (array 0–2) | `release`, `releaseloso`, `releaseeeg` | released checkpoints, `weights/merit/` |

The tag is recovered from the run-directory name, so two experiments that share
variant names (the ladder on raw vs. matched candidates; the lag control at
K=4 vs. K=2) never merge.

## Released checkpoints

`weights/merit/` (tracked in git, 11 MB) holds one file per fold for the three
models the paper calls MERIT: fused within-listener (5 folds), fused LOSO
(16 folds) and EEG-only within-listener (5 folds), each with `state_dict` +
full model config. `weights/merit/README.md` documents loading;
`notebooks/merit_iclr2027_reproduce.ipynb` §5–6 verifies all 26 load with `strict=True` and re-scores
them on their exact test folds.

## Open items before submission

- `refs.bib` → `maestro2026`: fill in the real venue / year / arXiv id.
  Entries `rotaru2024`, `wu2022greedy`, `assran2025vjepa2` are web-verified; the
  rest were written from memory and should be checked.
- Author block and acknowledgements (currently the anonymous placeholder).
- Trim the main text to 9 pages once all result tables are populated.
- `% VERIFY` comments in `sections/06_results.tex` mark prose whose direction
  must be re-read against the final numbers.
