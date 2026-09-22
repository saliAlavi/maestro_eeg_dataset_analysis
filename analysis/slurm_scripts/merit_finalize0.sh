#!/bin/bash
# Wait for the MERIT jobs to drain, then regenerate macros, figures and the PDF.
set -uo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while squeue -u alialavi -h -o "%j" | grep -qE '^(cr_|credit_)'; do
  n=$(squeue -u alialavi -h -o "%j" | grep -cE '^(cr_|credit_)')
  echo "[$(date +%H:%M:%S)] $n MERIT jobs still queued/running"
  sleep 300
done
echo "[$(date +%H:%M:%S)] queue drained; regenerating"
$PY -m src.tools.make_macros --out docs/iclr2027/results_macros.tex
$PY -m src.tools.make_figures --out docs/iclr2027/figs
$PY -m src.tools.report > docs/iclr2027/RESULTS.txt
$PY -m src.tools.report --streams >> docs/iclr2027/RESULTS.txt
cd docs/iclr2027 && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "remaining unresolved cells in the PDF: $(pdftotext main.pdf - | grep -o '??' | wc -l)"
pdfinfo main.pdf | grep Pages
