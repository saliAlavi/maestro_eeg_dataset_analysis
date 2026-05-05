"""Build all .ipynb notebooks from the per-notebook Python sources in this dir.

Each notebook is defined in its own ``nbXX_*.py`` file as a list of cells in
``CELLS`` (alternating markdown / code blocks). This builder:
1. Reads each source.
2. Emits a valid Jupyter v4 ``.ipynb`` JSON file at ``../<name>.ipynb``.
3. Applies a consistent metadata + kernel spec.

Source-of-truth is the .py files (git-friendly, diffable). Re-run after edits:
    python notebooks/_src/_build_notebooks.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent

KERNEL_META = {
    "kernelspec": {
        "display_name": "Python 3 (maestro-loader)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11",
    },
}


def split_lines(s: str) -> list[str]:
    """Convert a string to the list-of-lines format Jupyter expects."""
    if not s:
        return []
    lines = s.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1]  # leave trailing line w/o newline
    return lines


_cell_counter = [0]


def _next_id() -> str:
    _cell_counter[0] += 1
    return f"cell-{_cell_counter[0]:04d}"


def make_cell(kind: str, source: str) -> dict:
    if kind == "md":
        return {
            "cell_type": "markdown",
            "id": _next_id(),
            "metadata": {},
            "source": split_lines(source),
        }
    if kind == "code":
        return {
            "cell_type": "code",
            "id": _next_id(),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": split_lines(source),
        }
    raise ValueError(kind)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_notebook(src_py: Path) -> Path:
    _cell_counter[0] = 0  # reset per notebook so ids are file-local
    mod = load_module(src_py)
    cells = [make_cell(kind, body) for kind, body in mod.CELLS]
    nb = {
        "cells": cells,
        "metadata": KERNEL_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = OUT / (src_py.stem + ".ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    return out


def main() -> int:
    sources = sorted(p for p in SRC.glob("nb*.py"))
    if not sources:
        print("no nb*.py sources found", file=sys.stderr)
        return 1
    for src in sources:
        out = build_notebook(src)
        print(f"  built {out.relative_to(OUT.parent)}  ({len(load_module(src).CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
