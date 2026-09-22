# TASLP draft — handoff

Last worked: 2026-08-29. Everything below was verified from files on disk at that
date, not from memory. Where a number appears here it can be re-derived from the
paths given.

---

## 1. What this is

`docs/taslp_draft/bare_jrnl_new_sample4.tex` is the MAESTRO dataset paper for IEEE
TASLP. Sections IV, V and VI were rewritten because the numbers they carried came
from a benchmark implementation whose reported accuracy was **not attributable to
the physiological recordings**: permuting the recordings across test windows changed
T1 accuracy by 0.0009, and substituting zeros changed it by nothing.

Sections I, II and III are the original authors' text. Edits outside IV–VI, all
factual corrections: the attended-source rotation in §II (see §5 below); the SNR
range corrected from 0–18 dB to **3–18 dB** in the abstract-adjacent intro, §II and
the comparison table (verified against `experiment_data/trials.csv`: main trials
3–18, training 8–16 — no trial is below 3 dB); and the Fig. 3 caption now states
that its 82.0 % dashed line is the mean over all responses while the text's 83.2 %
excludes "I could not pay attention" responses (both verified exactly from
`answers.json`). Template leftovers (running head, received dates, pubid year) were
also replaced.

**Build** (the module load is required; without it `stfloats.sty` is missing):

```bash
cd docs/taslp_draft
source /etc/profile.d/lmod.sh; module load texlive/2024
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
bibtex   bare_jrnl_new_sample4
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex   # x2 more
```

Current state: **22 pages, 0 errors, 0 undefined references, 0 undefined citations,
0 overfull boxes, 0 multiply-defined labels, and no `!` lines in the log at all.**
The `subcaption`/`subfigure`/`subfig` clash is resolved: `subcaption` and `subfig`
were removed (neither's macros were used); `subfigure` alone remains and is what
the `\subfigure` calls in §III use.

---

## 2. The FACT\_NEEDED placeholder — resolved 2026-08-30

The former placeholder (KUL/DTU decision-window lengths for the binary-AAD
comparison in §VI-B) was filled from the primary sources, not from memory:

- **KUL (biesmans2016):** the paper's own Table I gives 74.5–81.5 % across envelope
  extraction methods at **30 s** trials (best method `p-law sub` = 81.5 %), and
  87.5 % at **60 s**; the text notes "Most studies on AAD employ 60 second trials."
  Verified from the author PDF at
  `homes.esat.kuleuven.be/~abertran/reports/2017-05-16 Biesmans.pdf`.
- **DTU (fuglsang2017):** the paper's abstract states decoding was "equally high
  (80–90 % correct)" using "40–50 s long blocks" — verified via the DTU Orbit
  record for doi 10.1016/j.neuroimage.2017.04.026.

§VI-B's first paragraph now states these numbers with their windows; the second
paragraph's "70–85 %" four-dataset range was widened to 70–90 % so it brackets the
verified KUL/DTU figures (ESAA's baselines, 84.6 %/84.3 %, sit inside it). The
authors should still eyeball the sentence, but nothing is guessed.

---

## 3. Where the results live

| What | Path |
|---|---|
| Code (branch `fixes`) | `/fs/scratch/PAS2301/alialavi/MAESTRO_fixbranch` |
| All result JSONs | `/fs/scratch/PAS2301/alialavi/fixbranch_results` |
| Generated tables, CSVs, `stats.json` | `.../fixbranch_results/paper_tables` |
| Generated figures | `.../fixbranch_results/paper_figures` |
| Per-cell significance | `.../fixbranch_results/significance` (450 files) |
| Superseded 20 s results | `.../fixbranch_results/_stale_w20` — do not use |
| Superseded host-path significance | `.../significance_hostpath` — do not use |

Regenerate everything:

```bash
cd /fs/scratch/PAS2301/alialavi/MAESTRO_fixbranch/scripts
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
$PY make_paper_tables.py  /fs/scratch/PAS2301/alialavi/fixbranch_results \
      --out /fs/scratch/PAS2301/alialavi/fixbranch_results/paper_tables
$PY make_paper_figures.py /fs/scratch/PAS2301/alialavi/fixbranch_results \
      --out /fs/scratch/PAS2301/alialavi/fixbranch_results/paper_figures
```

The table bodies are pasted into the `.tex`, not `\input`. If numbers change,
regenerate and re-paste the six grid tables (§4).

---

## 4. What was run

474 trained cells / 4,955 fold models, plus 75 evaluation-only passes. Each "cell"
is one (task, modality set, window, split) and is estimated from 5 folds
(within-subject) or 16 (LOSO).

| Experiment | Cells | Composition |
|---|---|---|
| T1 attended source, 4-way | 150 | 15 modality sets x 5 windows x 2 splits |
| T2 hemisphere, binary | 150 | 15 x 5 x 2 splits |
| T3 eccentricity, binary | 150 | 15 x 5 x 2 splits |
| Objective ablation (`--w_null 0`) | 20 | 5 sets x 2 windows x 2 splits |
| Candidate construction, raw vs matched | 4 | 2 tasks x 2 constructions |
| SNR-stratified (**no training**) | 75 | 15 x 5, LOSO |

Every training run also computes, internally: the audio-only acceptance probe, the
zeros ablation, the decision-flip rate, the embedding-collapse measure, and three
permutation nulls (global, position-stratified, trial-stratified).

Total compute: 227 GPU-hours over 235 array tasks on one A100 PCIe 40 GB per task,
`preemptible-nextgen`. No task failed or was preempted.

---

## 5. Findings that changed the paper's claims

These are the places where the rewritten text says something different from the
original draft. Each is load-bearing.

**Hemisphere decoding inverts.** EEG alone reaches 53.4–58.1 % under LOSO; gaze alone
reaches 68.9–70.9 %. (Earlier drafts of the rewrite said 43.4–58.1 and 68.9–72.2 —
those extremes came from the pre-fix 20 s cells and were purged from the prose on
2026-08-29.) The original draft reported EEG at 68.5–78.5 % and argued it was
"directly comparable to binary AAD benchmarks". §VI-B now argues the opposite and
explains why: free head and eye movement plus a permutation null.

**The SNR climb disappears.** The old figure showed EEG accuracy rising 17.7 % -> 80.0 %
across SNR bins. Under the fixed benchmark it is 53.6 / 42.1 / 53.0 / 52.5, and
**no** modality of the fifteen shows a strictly increasing curve. The bin-wise nulls
are not flat and track accuracy, which is what produced the old monotone appearance.

**Accuracy rises monotonically with modality count** (51.4 -> 56.4 -> 59.2 -> 61.1 %
within-subject and 52.7 -> 56.4 -> 58.2 -> 59.0 % LOSO, EEG-containing configurations
only — verified against the final tables). The original draft claimed the opposite and
built a limitation and a future-work item on it; both are gone.

**Seven of the fifteen modality rows are a different task.** With no EEG there is no
coupling branch, `couple_mod is None`, and the candidate envelopes never enter the
forward pass — verified by execution, the logits are bit-identical for any audio or
none. Those rows classify the attended class index; they do not match it to speech.
They are marked `†` throughout and must not be pooled with the EEG rows into a single
ranking.

**The attended-source rotation is deterministic.** `attended == ((k-1) mod 4)+1` for
all 100 main trials, identically for every participant (verified against
`metadata/trials.csv`). §II said only that the attended speaker was "balanced across
positions", which reads as counterbalancing. **This is the one edit outside IV–VI.**
Its consequence, stated in §V-F: a positive contribution establishes dependence on the
recording, not on auditory attention specifically.

---

## 6. Two defects found and fixed — check these first if numbers look wrong

**The 20 s window count.** Trials are 1908 samples (29.81 s) after the streams are
clipped to their shared span. `load_trial` padded each trial to the next whole
multiple of the *window* length. At 5/10/15/30 s that padded 1908 -> 1920 and the hop
rule held; at 20 s the next multiple is 2560, beyond the one-second allowance, so no
padding was applied and only one 20 s window fitted. The 20 s column was therefore
trained on a third of the windows of the 15 s column, which is the entire unexplained
"20 s dip" that ran through both splits and all three tasks. Trials now pad to the
nominal 30 s at every window size; counts are 11 / 5 / 3 / 2 / 1 as documented, and
20 s LOSO now has 40 test windows per fold rather than 20. All 60 affected cells were
retrained. Commit `c956d5f`.

**The permutation p-value was saturated.** The training runs use 20 permutations. That
p-value cannot fall below 1/(n+1) = 0.0476, and 1,299 of 1,575 T1 fold models sat
exactly on that floor, so it reported the permutation count and not the effect.
`recompute_significance.py` re-derives the same test for every cell from its saved
checkpoints at 10,000 permutations. **It trains nothing** — it loads state dicts under
`no_grad` — a point worth remembering if someone asks why it is running.

---

## 7. Significance, as now reported

Two levels, both in `.../significance/sig_*.json` and summarised into
`paper_tables/stats.json` under `"significance"`:

- **per fold** — accuracy against that fold's own permuted accuracies;
- **per cell** — the fold-averaged accuracy against the distribution of fold-averaged
  accuracies under permutation. This tests the quantity the tables print, against the
  null they are printed against, with no normality assumption.

Corrected by Holm within each of the six 75-cell (task x split) families. **435 of 450
cells reach p < 1.0e-4.** Cells that fail are marked `‡` in the tables — marking the
exceptions, since marking the rule would put a symbol on 435 cells.

| Task | Split | Fail | z range | median z |
|---|---|---|---|---|
| T1 | within | 0/75 | 10.9–81.7 | 40.5 |
| T1 | LOSO | 0/75 | 4.4–35.1 | 17.7 |
| T2 | within | 0/75 | 1.7–62.1 | 26.3 |
| T2 | LOSO | 1/75 | 1.4–25.9 | 11.7 |
| T3 | within | 0/75 | 2.6–45.1 | 18.9 |
| T3 | LOSO | **14/75** | −0.4–20.9 | 7.7 |

Because p saturates at the floor almost everywhere, **z is the discriminating
statistic** and is reported alongside. The 14 T3 LOSO failures are not scattered:
gaze fails at **all five** windows and head motion at three (the rest are the
Gaze+IMU pair at 20/30 s and four EEG-containing cells at 30 s), which is the
gaze-carries-hemisphere-but-not-eccentricity dissociation established by test rather
than by inspection.

**The objective ablation** (§V-F) answers the strongest reviewer objection — the
permutation hinges optimise the very quantity the paper reports. Removing them changes
the contribution by +0.24 points on average over the 16 EEG-containing cells, up in 9
and down in 7, with neither a paired t nor a signed-rank test rejecting zero. The four
video-only cells are an accidental but valid **negative control**: with no coupling
branch the hinges are never evaluated, so `w_null=0` cannot change the objective, yet
retraining moved them by up to 2.08 points. That is the pipeline's run-to-run noise,
and 13 of the 16 real differences are smaller than it.

---

## 8. Table and figure inventory

Labels were renamed on 2026-08-29 to match their contents; the old names
(`tab:aad_results` etc.) described a superseded layout.

| Label | Content |
|---|---|
| `tab:comparison` | dataset comparison (§II, original authors') |
| `tab:t1_within`, `tab:t1_loso` | T1, one table per split |
| `tab:t2_within`, `tab:t2_loso` | T2 hemisphere |
| `tab:t3_within`, `tab:t3_loso` | T3 eccentricity |
| `tab:probe` | audio-only acceptance probe |
| `tab:diagnostics` | zeros ablation, flip rate, collapse at 10 s |
| `fig:baseline_classification` | architecture (`media/baseline_network.png`) |
| `fig:aad_loso_bar` | per-participant contribution (`media/fig_folds.png`) |
| `fig:snr_str` | SNR bins with the null (`media/snr_fixed.png`) |

Grid cells read `accuracy / permuted / **contribution**` with the fold count as a
subscript and `‡` if the cell fails Holm correction.

---

## 9. Known soft spots

- ~~`subcaption` + `subfigure` clash~~ — resolved 2026-08-29 (see §1).
- **A commented-out "Ecological Validity" subsection** still sits in §VI. Keep or cut.
- **The linear reference decoder reconstructs at r ≈ 0.004**, one to two orders below
  published values for the method. No claim of the form "the deep model beats linear
  decoding" is supported until that is diagnosed; the text says so.
- **The attended/masker rendering asymmetry** (crest factor 11.1 vs 20.5 dB) is
  reported in §V-A as the mechanism the evidence favours, not as demonstrated — every
  recording appears in exactly one role, so it cannot be established causally from the
  rendered material. It arguably belongs in §II as a property of the released corpus.
- **Quantile matching is not lossless.** §IV-B now says so, citing `biesmans2016`
  (power-law compression, itself rank-preserving, changes AAD accuracy) against the
  claim that only temporal ordering matters. A contribution under the matched
  construction is a conservative estimate.
- **The trial-stratified control is vacuous at 30 s**, where one window per trial makes
  the within-trial permutation the identity. Stated in §V-F.

---

## 10. Prose-sync pass, 2026-08-29 (second session)

The 20 s retrain (§6) regenerated the tables but much of the Results/Discussion
prose still quoted pre-fix numbers. A full pass re-derived every quoted range from
`paper_tables/folds_t1.csv` and `folds_t2t3.csv`. The load-bearing corrections:

- §V-C (T2): EEG LOSO is 53.44–58.13 % (not 43.44–), contributions +2.92..+6.27
  (never negative — the old "negative at 20 s" claim is gone; the 30 s cell fails
  Holm instead), best-vs-EEG margin +14.94..+21.22 (not ..32.11).
- §VI-A: the 20 s dip in the EEG contribution no longer exists (it *was* the padding
  bug); the trajectory paragraph now says monotone rise on T1 and describes the real
  binary-task peaks (T2 15 s both splits; T3 15 s LOSO / 20 s within).
- §V-B: fold-level summary recomputed: **52 of 80**, mean gain 14.61, overall
  +7.33 ± 12.60, median +8.67, range −25.00..+35.00 (the TODO-20S comment is gone).
  The 20 s four-modality within accuracy is 62.80 (was printed 59.83). LOSO p-range
  0.0022–0.040. A footnote now pre-empts the "identical 5 s EEG cells" question
  (genuine coincidence, verified per fold).
- §V-D (T3): EEG 56.88–64.79 / +6.17..+13.98; gaze min 49.52, min folds 7/16; the
  20 s multimodal margin is +2.55, p = 0.60.
- §V-F: gaze fails all five T3-LOSO windows; failure breakdown corrected. **The
  stratified permutations are now actually reported** (they were promised and
  missing): position-stratified contribution ≈ unrestricted everywhere;
  trial-stratified is positive in all 64 EEG-containing 5–20 s cells but ~half the
  unrestricted value, stated with the window-overlap caveat.
- §V-A: new subsubsection reports the four raw-vs-matched trained cells (EEG+Gaze,
  10 s, within, both binary tasks — the `smoke_within_w10.0_*` runs, which also
  supply the raw probe rows): raw inflates the model null (0.601/0.568 vs
  0.506/0.492) and *depresses* the hemisphere contribution (+10.2 vs +16.7).
  §IV-B's "probe-only comparison" sentence and the §V cell-inventory cross-refs
  were aligned. The pairs are compared only with each other, not with the grid.
- `folds_t2t3.csv` covers both splits; the release description in §V now says so.
- EEG-free pooled null floor corrected to 0.2412 (was 0.2367).

Everything else in the prose was checked against the tables and found already
correct (modality-count means, window counts, probe arithmetic, T1-within t-tests,
diagnostics). Build after the pass: 22 pages, 0 errors, 0 overfull, no `!` lines.

### 10b. SNR subsection + figure — fixed 2026-08-31 (third session)

The §V SNR-stratified subsection escaped the 2026-08-29 pass because its *inputs*
were regenerated out of order: `media/snr_fixed.png` was copied into the paper on
Aug 26 19:23, and even `paper_figures/fig_snr.png` (Aug 27 14:48:41) was written
**46 s before** the final post-fix `snr/snr_eeg_loso_w20.json` (14:49:27). Only
`paper_tables/snr_bins.csv` (Aug 29) was current. Fixed:

- `fig_snr.png` regenerated from the current `snr/*.json` with
  `scripts/make_paper_figures.py`; copied to `media/snr_fixed.png` and to
  `fixbranch_results/paper_figures/`. Annotations are now +0.26/+0.20/+0.26/+0.25.
  (`fig_folds.png` regenerates byte-identical — it was never stale.)
- EEG bin accuracies 53.59/42.06/53.02/52.52 → **54.22/42.68/55.10/53.98 %**;
  contributions +25.5/+18.8/+23.1/+24.3 → **+26.3/+19.6/+25.6/+25.4**;
  nulls 28.1/23.2/29.9/28.3 → **27.9/23.0/29.5/28.6 %**.
- Two claims flipped, not just drifted: EEG's bin-index rank correlation is now
  **ρ = 0.00** (was −0.40), and **scene video alone is now strictly increasing**
  across bins (34.51→45.37 %), so "none of the 15 increases" was scoped to the
  eight EEG-containing rows with the video exception stated (orienting, not speech).
- "Removes about 40 % of the apparent structure" → **45 %** (computed 46 %);
  "six EEG-free rows hardest in the lowest bin" → **five**, Gaze+IMU is hardest in
  the highest bin. "Second bin hardest at 9 of 15" and the "≈7-point spread /
  flat-once-floored" conclusions survive unchanged.

Build after the fix: 22 pages, 0 errors, 0 overfull.

### 10c. Page-count reduction pass, 2026-09-01 (fourth session)

The paper was cut from 22 to 12 pages by rewriting Sections IV and V only, per the
author's instruction (target 10-11 pages; nothing outside IV/V was touched).

**Floats removed** (author-specified): Table II (diagnostics), Table III (probe),
Fig. 6 (per-fold dumbbell), Fig. 7 (SNR curve), and Tables VI-IX (all four T2/T3
tables). The two T1 tables were **merged into a single `table*` labeled `tab:t1`**
(within-subject panel on top, LOSO below) with one shortened caption; Fig. 5
(baseline network) kept at 0.80\linewidth with a shortened caption.

**Prose**: IV compressed to ~1.5 pages (tasks / splits / candidate construction with
eq. qmatch / permutation null with eq. contribution / baseline network / training;
the old IV-F Statistical Analysis was folded into IV-D as a paragraph, label
`ssec:stats` dropped, `eq:coupling` and label `4-classification`/`ssec:probe`
dropped — all were referenced only from removed text). V keeps short number
summaries where the tables were removed (T2, T3, SNR) and number-light prose for T1
(the merged table carries the numbers). All labels referenced from the Discussion
survive: `loso` (now on the V-B heading), `ssec:whatitmeans`, `ssec:configs`,
`ssec:certify`, `ssec:candidates`, `ssec:baseline`, `ssec:tasks`, `results`,
`benchmark`, `sec:res-t3`. Every V-F caveat the Discussion cites (rotation confound,
30-s stratified vacuity, residual floor 0.310→0.263, binary reliability 435/450,
hinge ablation) is retained in compressed form. bibtex was rerun.

**Amendment 2026-09-02 (author feedback):** the T2 and T3 results subsections were
then removed entirely (their tables are gone, so their discussion goes too), and every
CSV filename mention (`folds_t1.csv`, `folds_t2t3.csv`, `snr_bins.csv`) was purged from
the running text — release is now referred to only as "released with the benchmark".
Labels `sec:res-t3` and the V-F cross-ref to it are gone; V-A's forward pointer to the
eccentricity results was dropped. Section V now contains: Certifying the Candidate
Sets, T1 (merged table), SNR-Stratified Analysis, and What the Contribution
Establishes. NOTE: Section VI (untouched, per the only-IV/V constraint) still
interprets hemisphere/eccentricity findings and says "as Section V sets out" — the
author may want to reconcile VI themselves or authorize an edit.

Build: 12 pages, 0 errors, 0 overfull, no undefined refs (pdftotext shows no `??`).
The pre-cut 22-page source is saved at the session scratchpad as
`bare_jrnl_precut.tex.bak`; a durable copy is `bare_jrnl_PREVIOUS.tex.bak` (older).
Reaching 11 pages needs ~1 more page: candidate levers are dropping Fig. 5
(~0.35 pg), keeping only the LOSO panel of tab:t1 (~0.45 pg), or cuts outside IV/V
(Discussion is ~2.2 pages; Fig. 2 gaze montage is large). 10 pages is not reachable
from IV/V alone.


## 11. Prior context

`analysis/n_gh_checks/FIXED_MODEL.md` is the internal audit that led to all of this:
the ablation ladder showing candidate construction and objective dominate while the
encoder contributes nothing, the mechanism trace on the acoustic confound, and the
lag-band control. It is **not** the source of any number in the TASLP tables — it is a
single-window mechanism study with no hemisphere or eccentricity task at all. Do not
try to populate a table row from it.
