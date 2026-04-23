"""Helper to generate Jupyter notebooks from compact (type, src) cell lists.

Usage (from another file):
    from _build_notebook import build
    build("analysis/01_data_audit.ipynb", [
        ("md", "# Title"),
        ("code", "print('hi')"),
    ])
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path


def _cell(kind: str, src: str) -> dict:
    src_lines = src.splitlines(keepends=True)
    cell_id = uuid.uuid4().hex[:12]
    if kind == "md":
        return {
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": src_lines,
        }
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src_lines,
    }


def build(path: str | Path, cells: list[tuple[str, str]]) -> Path:
    nb = {
        "cells": [_cell(k, s) for k, s in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    return path


if __name__ == "__main__":
    # Smoke test.
    p = build("/tmp/_nb_test.ipynb", [("md", "# hi"), ("code", "print('x')")])
    print("OK", p)
