"""Build the anonymized supplementary code zip for conference submission.

Stages src/, configs/, the released weights, the measured-results snapshot and
the reproduction notebook (outputs cleared) into tmp/supp_stage/merit_code,
scrubs every identifying string, re-saves the checkpoints with plain-container
configs (the original ``dict(cfg)`` pickled nested DictConfigs whose parent
references carried the full config tree, cluster paths included), refuses to
zip if any identifying token survives, and writes tmp/merit_supplementary.zip.

    python -m src.tools.build_supplementary
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE_ROOT = os.path.join(REPO, "tmp", "supp_stage")
STAGE = os.path.join(STAGE_ROOT, "merit_code")
ZIP = os.path.join(REPO, "tmp", "merit_supplementary.zip")

# ordered: path prefixes first, then bare tokens
REPLACEMENTS = [
    ("/fs/scratch/PAS2301/alialavi", "/path/to/scratch"),
    ("/users/PAS2301/alialavi", "/path/to"),
    ("PAS2966", "ACCOUNT"), ("PAS2301", "ACCOUNT"),
    ("alialavi", "anon"),
    ("the OSU multimodal corpus", "the MAESTRO corpus"),
    ("SLURM submit scripts (ACCOUNT, nextgen partition)",
     "SLURM submit scripts (not included in this archive)"),
    ("Account `ACCOUNT`, partition `nextgen`. ", ""),
]
# tokens that must not survive anywhere, in text or binaries
FORBIDDEN = ["alialavi", "pas2301", "pas2966", "ohio", "williamson",
             "salialavi", "sa.alavi", "aspire", "github.com"]
BINARY_EXT = (".pt", ".parquet", ".png", ".pdf")


def build():
    if os.path.exists(STAGE_ROOT):
        shutil.rmtree(STAGE_ROOT)
    # this builder carries the tokens it scrubs, so it must not ship itself
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints",
                                 "build_supplementary.py")
    shutil.copytree(os.path.join(REPO, "src"), os.path.join(STAGE, "src"), ignore=ign)
    shutil.copytree(os.path.join(REPO, "configs"), os.path.join(STAGE, "configs"), ignore=ign)
    shutil.copytree(os.path.join(REPO, "weights", "merit"),
                    os.path.join(STAGE, "weights", "merit"))
    shutil.copytree(os.path.join(REPO, "docs", "iclr2027", "results"),
                    os.path.join(STAGE, "docs", "iclr2027", "results"))
    shutil.copy(os.path.join(REPO, "src", "requirements.txt"),
                os.path.join(STAGE, "requirements.txt"))
    shutil.copy(os.path.join(REPO, "docs", "iclr2027", "supplementary_README.md"),
                os.path.join(STAGE, "README.md"))

    # notebook: strip outputs, then the text pass below sanitizes its source
    import nbformat
    nb = nbformat.read(os.path.join(REPO, "notebooks", "merit_iclr2027_reproduce.ipynb"),
                       as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    os.makedirs(os.path.join(STAGE, "notebooks"), exist_ok=True)
    nbformat.write(nb, os.path.join(STAGE, "notebooks", "merit_reproduce.ipynb"))

    n_edit = 0
    for root, _, files in os.walk(STAGE):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(BINARY_EXT):
                continue
            with open(p, encoding="utf-8") as fh:
                t = fh.read()
            t2 = t
            for a, b in REPLACEMENTS:
                t2 = t2.replace(a, b)
            if t2 != t:
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(t2)
                n_edit += 1
    print(f"sanitized {n_edit} text files")

    import torch
    from omegaconf import OmegaConf
    n_ck = 0
    wdir = os.path.join(STAGE, "weights", "merit")
    for root, _, files in os.walk(wdir):
        for f in sorted(files):
            if not f.endswith(".pt"):
                continue
            p = os.path.join(root, f)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = OmegaConf.to_container(OmegaConf.create(ck["cfg"]), resolve=True)
            torch.save({"state_dict": ck["state_dict"], "cfg": cfg}, p)
            n_ck += 1
    print(f"re-saved {n_ck} checkpoints with plain cfg")

    # hard gate: nothing identifying may survive, in text or binaries
    leaks = []
    pat = re.compile("|".join(map(re.escape, FORBIDDEN)), re.IGNORECASE)
    for root, _, files in os.walk(STAGE):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                if pat.search(fh.read().decode("utf-8", errors="ignore")):
                    leaks.append(os.path.relpath(p, STAGE))
    if leaks:
        sys.exit(f"identifying strings survive in: {leaks}")
    print("leak scan clean")

    if os.path.exists(ZIP):
        os.remove(ZIP)
    subprocess.run(["zip", "-qr", ZIP, "merit_code"], cwd=STAGE_ROOT, check=True)
    print(f"{ZIP}  ({os.path.getsize(ZIP)/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
