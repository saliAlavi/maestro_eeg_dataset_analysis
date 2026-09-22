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

## How numbers get into the paper

```
run  ──►  /fs/scratch/PAS2301/alialavi/projects/multimodal_aad__credit__<tag>__<stamp>/results.parquet
          │
          ├─ python -m src.tools.make_macros --out docs/iclr2027/results_macros.tex
          │      emits  \defres{<tag>/<variant>/<protocol>/<metric>}{value}
          │      the manuscript writes  \R{streams/all/within/acc}
          │      a key with no measurement renders as a red ?? in the PDF
          │
          └─ python -m src.tools.make_figures --out docs/iclr2027/figs
```

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
| `credit_certify.sbatch` | — | Tab. audio-only probe, Tab. role separation |
| `credit_within.sbatch EXP=ladder_raw` | `ladder_raw` | ablation ladder, confounded candidates |
| `credit_within.sbatch EXP=ladder_qm` | `ladder_qm` | ablation ladder, matched candidates |
| `credit_within.sbatch EXP=streams` | `streams` | per-stream credit, within-subject |
| `credit_within.sbatch EXP=hinge` | `hinge` | **the freeloading experiment** |
| `credit_within.sbatch EXP=lag` | `lag_qm`, `lag_mm` | neural directionality |
| `credit_within.sbatch EXP=orientfree` | `mm3`, `mm2` | same-talker negatives |
| `credit_loso.sbatch` (array 0–15) | `loso` | leave-one-listener-out |
| `credit_windows.sbatch` (array 0–4) | `w5`…`w30` | decision-window sweep |

The tag is recovered from the run-directory name, so two experiments that share
variant names (the ladder on raw vs. matched candidates; the lag control at
K=4 vs. K=2) never merge.

## Open items before submission

- `refs.bib` → `maestro2026`: fill in the real venue / year / arXiv id.
  Entries `rotaru2024`, `wu2022greedy`, `assran2025vjepa2` are web-verified; the
  rest were written from memory and should be checked.
- Author block and acknowledgements (currently the anonymous placeholder).
- Trim the main text to 9 pages once all result tables are populated.
- `% VERIFY` comments in `sections/06_results.tex` mark prose whose direction
  must be re-read against the final numbers.
