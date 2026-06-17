"""Runner layer interface.

The runner is the only place that knows about both the data layer and the model
layer, and it talks to each only through their constant interfaces
(``AbstractDataModule`` / ``AbstractModel``). Swapping a model or changing how
data is cached therefore never touches the runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractRunner(ABC):
    @abstractmethod
    def run(self) -> "object":
        """Execute the full protocol sweep and return a results DataFrame."""
