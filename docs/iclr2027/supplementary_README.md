# MERIT — supplementary code

Code, trained weights and measured run outputs for *MERIT: Making Every
Modality Earn Its Contribution in Multimodal Attention Decoding*.
Anonymized for review: local paths are replaced with `/path/to/...`
placeholders; set them for your system via the config flags shown below.

## Contents

```
src/                    the framework: data / model / runner layers, CLI (python -m src.main)
  models/merit/         the MERIT decoder: modules, losses, exact Shapley credit (attribution.py)
  data/merit_data.py    windows, candidate constructions (unmatched/matched/same-talker), splits
  data/external_data.py KU Leuven and DTU through the same interface
  tools/                make_macros.py / make_figures.py: every number and figure in the paper
configs/                Hydra configs (data / model / runner groups)
weights/merit/          26 trained checkpoints: fused within (5 folds), fused LOSO (16), EEG-only (5)
docs/iclr2027/results/  the measured results.parquet of every run behind the paper's tables
notebooks/merit_reproduce.ipynb  end-to-end walkthrough (outputs cleared)
```

## Environment

Python ≥ 3.10 with PyTorch ≥ 2.0, plus `requirements.txt`
(hydra-core, omegaconf, pandas, pyarrow, scipy, scikit-learn, matplotlib,
tqdm, wandb — wandb can stay offline throughout).

## Reproduce the paper's tables and figures (no GPU, no data needed)

Every number in the manuscript is a `\R{key}` macro resolved from run outputs;
nothing is typed by hand. The measured outputs ship in
`docs/iclr2027/results/`, so from this directory:

```bash
python -m src.tools.make_macros  --out results_macros.tex   # 8000+ macros
python -m src.tools.make_figures --out figs                 # both paper figures
python -m src.tools.report --tag v2hinge2 --streams         # console tables
```

## Use the trained weights

Each checkpoint holds `state_dict` plus the full model config, and loads with
`strict=True`:

```python
import torch
from omegaconf import OmegaConf
from src.models.merit.model import MeritModel

ck = torch.load("weights/merit/fused_within/H4_nohinge_nodrop__within_f0.pt",
                map_location="cpu", weights_only=False)
model = MeritModel(OmegaConf.create(ck["cfg"]), feature_dims={"n_candidates": 4})
net = model.build_module()
net.load_state_dict(ck["state_dict"], strict=True)
```

`weights/merit/README.md` documents the folds and per-fold results
(`results.json`); the splits are fixed by seed 0, so `src/data/merit_data.py`
rebuilds exactly the test windows each file was scored on.

## Train from scratch

Needs the MAESTRO dataset (public; see the paper) and one GPU. Point the data
layer at your copies with `data.dataset_path=...` and `data.cache_root=...`.
The three released models:

```bash
CROP="data.window_sec=30 data.hop_sec=15 data.crop_len=640 data.n_train_crops=8 data.n_eval_crops=5"

# fused MERIT, within listeners (5 folds)
python -m src.main data=merit model=merit wandb.mode=offline $CROP runner=merit_release \
    data.euclidean_align=true data.audio_feats=env_onset

# fused MERIT, unseen listeners (16-fold LOSO)
python -m src.main data=merit model=merit wandb.mode=offline $CROP runner=merit_release \
    data.euclidean_align=true data.audio_feats=env_onset runner.protocols=[loso]

# EEG-only MERIT (5 folds)
python -m src.main data=merit model=merit wandb.mode=offline $CROP runner=merit_release_eeg \
    data.audio_feats=env_onset
```

The remaining experiments (ablation ladders, hinge 2×2, window sweep,
same-talker controls, KU Leuven / DTU transfer) are runner configs under
`configs/runner/merit*.yaml`; each writes a `results.parquet` that
`make_macros` picks up.

## Notes

- The candidate acceptance probe (Section 4(i) of the paper) runs before any
  training: `python -m src.main mode=certify data=merit model=merit`.
- The runner refuses to train on a declared stream the data layer did not
  build, and every result row carries its candidate construction, window and
  audio-only probe.
- `src/` also contains the other model packages developed on this framework;
  only `src/models/merit/` is used by the paper.
