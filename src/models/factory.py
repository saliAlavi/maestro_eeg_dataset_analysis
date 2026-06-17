"""Model-layer factory.

Concrete models live in ``src/models/{model_name}/model.py`` and register
themselves on import via ``@MODEL_REGISTRY.register("name")``. The factory
imports the package named in the config and instantiates it, so the runner
never imports a concrete model and adding a model is a pure plug-in.
"""
from __future__ import annotations

import importlib

from ..common.registry import Registry

MODEL_REGISTRY = Registry("model")

# Map config name -> python subpackage under src.models
_PACKAGES = {
    "linear_backward": "linear_backward",
    "riemann_tangent": "riemann_tangent",
    "eegnet_mm": "eegnet_mm",
    "maestro": "maestro",
    "recon_mm": "recon_mm",
    "recon_mm_gaze": "recon_mm_gaze",
    "recon_mix": "recon_mix",
    "multipath_mm": "multipath_mm",
    "eeg_spatial": "eeg_spatial",
    "eeg_spatial_gaze": "eeg_spatial_gaze",
    "source_fusion": "source_fusion",
    "deep_match": "deep_match",
    "source_net": "source_net",
    "source_hier": "source_hier",
    "source_azi": "source_azi",
    "source_rel": "source_rel",
    "asad_eeg": "asad_eeg",
    "asad_mm": "asad_mm",
    "mm_recon": "mm_recon",
    "aad_contrastive": "aad_contrastive",
}


def build_model(cfg, feature_dims: dict):
    """``cfg`` is the resolved ``model`` config group (has ``cfg.name``)."""
    name = cfg.name
    pkg = _PACKAGES.get(name, name)
    importlib.import_module(f"src.models.{pkg}.model")  # triggers registration
    cls = MODEL_REGISTRY.get(name)
    if getattr(cls, "is_neural", False):
        return cls(cfg, feature_dims)
    return cls(cfg, feature_dims)


__all__ = ["MODEL_REGISTRY", "build_model"]
