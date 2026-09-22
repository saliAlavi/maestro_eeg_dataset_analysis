# MERIT weights

Trained weights for the two models the paper calls MERIT, one file per fold.
Each file holds the weights (`state_dict`) and the full model configuration
(`cfg`) needed to rebuild the network.

| folder | model | split | files | accuracy (mean) | paper reports |
|---|---|---|---|---|---|
| `fused_within/` | fused MERIT: EEG, gaze, head, flow, fovea; no hinge, no modality dropout | within listeners, 5 content-disjoint folds | 5 | 0.8204 | 0.8173 (Table 1) |
| `fused_loso/`   | same model | leave-one-listener-out, 16 folds | 16 | 0.7563 | 0.7438 (Appendix J, abstract) |
| `eeg_within/`   | EEG-only MERIT (per-listener correction, onset channel) | within listeners, 5 folds | 5 | 0.6732 | 0.6915 (Table 2) |

Four-way task, chance 0.25. `results.json` gives every fold's accuracy, shuffled
accuracy, contribution and (for the fused model) the EEG's Shapley credit.

**These are retrainings, not the runs behind the paper's tables.** The reported
runs did not save their weights, so we retrained with the same code, data
settings and folds. GPU training is not bit-for-bit repeatable, so the numbers
differ slightly from the paper's: by +0.003, +0.013 and -0.018, all well inside
the fold-to-fold spread (sd 0.022, 0.177 and 0.058).

## Loading

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

File names give the fold: `within_f0` ... `within_f4` are the five content-disjoint
folds, and `loso_sN` holds out listener N. The folds are fixed by seed 0, so
`src/data/merit_data.py` rebuilds exactly the test windows each file was scored on.

## Reproducing them

The weights were produced by `analysis/slurm_scripts/merit_release.sbatch`, using
`configs/runner/merit_release.yaml` (fused) and `configs/runner/merit_release_eeg.yaml`
(EEG only). The data settings for all three are 30 s trials cut into 10 s
crops (8 random crops per trial in training, 5 fixed crops at test), the envelope
plus an onset channel, and, for the fused model, per-listener whitening of the EEG.
