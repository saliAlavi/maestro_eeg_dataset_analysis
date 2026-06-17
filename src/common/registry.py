"""A tiny name->class registry used to implement the factory pattern.

Each layer (data / model / runner) owns one ``Registry``. Concrete classes
register themselves with ``@REGISTRY.register("name")`` and the factory builds
them from the Hydra config by name, so adding a new model is a one-file change
that never touches the runner.
"""
from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._table: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            key = name.lower()
            if key in self._table:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._table[key] = cls
            return cls

        return deco

    def get(self, name: str) -> Type:
        key = name.lower()
        if key not in self._table:
            raise KeyError(
                f"unknown {self.kind} '{name}'. registered: {sorted(self._table)}"
            )
        return self._table[key]

    def available(self) -> list[str]:
        return sorted(self._table)
