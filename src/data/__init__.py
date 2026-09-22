"""Data layer: loading, preprocessing, caching, and the constant data view."""
from .base import AADView, AbstractDataModule, SplitDesc
from .factory import DATA_REGISTRY, build_datamodule
from .windows import WindowSpec

__all__ = [
    "AADView", "AbstractDataModule", "SplitDesc", "WindowSpec",
    "DATA_REGISTRY", "build_datamodule",
]
