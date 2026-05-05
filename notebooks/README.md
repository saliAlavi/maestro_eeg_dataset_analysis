# maestro-eeg-dataset · analysis notebooks

Reproducible analyses of the [maestro-eeg-dataset](https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset) — a multimodal auditory-attention-decoding (AAD) corpus with EEG, gaze, IMU, audio, and Tobii Glasses 3 scene video. Every notebook in this directory is self-contained: it installs the [`maestro-loader`](https://pypi.org/project/maestro-loader/) PyPI package, pulls only the data it needs from the Hugging Face Hub, and reproduces a section of the accompanying paper plus extensions.

## Quickstart

```bash
git clone <this-repo>
cd notebooks/
jupyter lab nb00_setup_and_quickstart.ipynb
```

Run cell-by-cell. Notebook 00 installs all dependencies; notebooks 01–08 are independent and can be run in any order after that.

## The notebooks

| # | Notebook | What it covers |
|---|---|---|
| 00 | [`nb00_setup_and_quickstart`](nb00_setup_and_quickstart.ipynb) | Install, load one segment, sanity-check every modality. |
| 01 | [`nb01_dataset_overview`](nb01_dataset_overview.ipynb) | Cohort demographics, trial structure, modality coverage, behavioural performance, bad-channel inventory. |
| 02 | [`nb02_eeg_signal_quality`](nb02_eeg_signal_quality.ipynb) | Per-channel PSDs, drift, line-noise, saturation diagnostics, agreement with auto bad-channel detection. |
| 03 | [`nb03_gaze_dynamics`](nb03_gaze_dynamics.ipynb) | Gaze validity, fixation-density heatmaps, pupil dynamics, saccade rates. |
| 04 | [`nb04_audio_stimulus_features`](nb04_audio_stimulus_features.ipynb) | Hilbert envelope, mel spectrograms, per-speaker fingerprints, audio-mix calibration. |
| 05 | [`nb05_eeg_stimulus_decoding`](nb05_eeg_stimulus_decoding.ipynb) | Backward TRF reconstructing the attended speech envelope from EEG; attended-vs-competing comparison; TRF kernel inspection. |
| 06 | [`nb06_attention_decoding`](nb06_attention_decoding.ipynb) | Binary AAD classification — within-subject and LOSO. |
| 07 | [`nb07_multimodal_fusion`](nb07_multimodal_fusion.ipynb) | Late fusion of EEG + gaze + IMU + speaker-azimuth features for AAD. |
| 08 | [`nb08_benchmark_protocol`](nb08_benchmark_protocol.ipynb) | Official evaluation protocol, reference baselines, submission template. |

## Two backends — sample vs. full dataset

Each notebook has a single configuration block at the top:

```python
USE_SAMPLE = True              # 422 MB — 3 subjects × 5 trials, fast iteration
# USE_SAMPLE = False           # 41.7 GB — full 16 subjects × 105 trials

REPO_ID = "aspire-osu/maestro-eeg-dataset-sample" if USE_SAMPLE else "aspire-osu/maestro-eeg-dataset"
```

Develop and prototype against the sample, then flip the flag for the production run. Numbers reported in the paper are computed on the full release.

## How the notebooks are sourced

Notebooks are generated from per-notebook Python files under `_src/`. Source-of-truth lives there because it's:

- **Diffable in git** — Jupyter JSON is hostile to code review.
- **Importable** — you can run `python _src/nb05_eeg_stimulus_decoding.py` if you want to skip Jupyter entirely.
- **Cleanly version-controlled** — output cells aren't checked in (they're regenerated when you run the notebooks).

To rebuild the `.ipynb` files after editing a `_src/*.py`:

```bash
python _src/_build_notebooks.py
```

## Requirements

Each notebook installs its own dependencies in cell 1 via `%pip install`. The full dependency set is also captured in [`requirements.txt`](requirements.txt) for offline / containerised use:

```bash
pip install -r requirements.txt
```

## Reproducibility

- All notebooks set `np.random.seed(1337)` at the top.
- Default Ridge α = 1.0; lag window 0–250 ms; 64 Hz EEG/audio rate; 1–32 Hz EEG band-pass.
- The exact `maestro-loader` version pinned per notebook is `>= 0.1.2`.

## License & citation

Code: Apache-2.0. Dataset: CC-BY-4.0.

```bibtex
@misc{maestro_eeg_2026,
  title  = {{maestro-eeg-dataset}: A Multimodal Auditory-Attention Dataset
             with EEG, Gaze, IMU, and Egocentric Video},
  author = {Alavi, Ali and Hasan, N. and Williamson, D.},
  year   = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset}},
}
```
