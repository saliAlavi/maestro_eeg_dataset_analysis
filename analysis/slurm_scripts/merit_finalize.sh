#!/bin/bash
# Wait for every MERIT/MERIT job to drain, then regenerate macros, figures, PDF.
set -uo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PAT='^(cr_|credit_|mv2_|merit_)'
while squeue -u alialavi -h -o "%j" | grep -qE "$PAT"; do
  echo "[$(date +%H:%M:%S)] $(squeue -u alialavi -h -o '%j' | grep -cE "$PAT") jobs running"
  sleep 300
done
echo "[$(date +%H:%M:%S)] drained; regenerating"
$PY -m src.tools.make_macros --out docs/iclr2027/results_macros.tex
$PY -m src.tools.make_figures --out docs/iclr2027/figs
$PY -m src.tools.report > docs/iclr2027/RESULTS.txt 2>&1
$PY -m src.tools.report --streams >> docs/iclr2027/RESULTS.txt 2>&1
cd docs/iclr2027 && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "ALL DONE: unresolved cells $(pdftotext main.pdf - | grep -o '??' | wc -l)"
