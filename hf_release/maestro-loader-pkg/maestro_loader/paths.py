"""Resolve dataset files from a local directory or the HuggingFace hub."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "aspire-osu/maestro-eeg-dataset"


@dataclass
class DatasetPaths:
    """Resolve relative dataset paths to concrete files.

    With ``local_path`` set, files are read directly from disk. Otherwise each
    file is lazily downloaded from the HuggingFace dataset repo on first access
    (cached by ``huggingface_hub``).
    """
    local_path: Path | None = None
    repo_id: str = DEFAULT_REPO_ID
    revision: str | None = None
    cache_dir: Path | None = None
    _glob_cache: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.local_path is not None:
            self.local_path = Path(self.local_path)

    # -- core resolver -- #
    def resolve(self, rel: str) -> Path:
        if self.local_path is not None:
            return self.local_path / rel
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            repo_id=self.repo_id, repo_type="dataset", filename=rel,
            revision=self.revision,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        ))

    def exists(self, rel: str) -> bool:
        if self.local_path is not None:
            return (self.local_path / rel).exists()
        try:
            self.resolve(rel)
            return True
        except Exception:
            return False

    # -- typed accessors -- #
    def metadata(self, name: str) -> Path:
        return self.resolve(f"metadata/{name}")

    def split(self, setting: str, fname: str) -> Path:
        sub = "loso" if setting == "loso" else "within"
        return self.resolve(f"splits/{sub}/{fname}")

    def parquet(self, modality: str, sid: str, tid: str) -> Path:
        return self.resolve(f"data/{modality}/subject={sid}/trial={tid}.parquet")

    def timing(self, sid: str, tid: str) -> Path:
        return self.resolve(f"media/timing/subject={sid}/trial={tid}.json")

    def video(self, sid: str, tid: str) -> Path:
        return self.resolve(f"media/video/subject={sid}/{tid}.mp4")

    def audio_file(self, tid: str, fname: str) -> Path:
        return self.resolve(f"media/audio/{tid}/{fname}")

    def list_repo_files(self) -> list[str]:
        """All file paths in the dataset (used to discover subjects/trials on HF)."""
        if self.local_path is not None:
            return [str(p.relative_to(self.local_path))
                    for p in self.local_path.rglob("*") if p.is_file()]
        from huggingface_hub import HfApi
        return HfApi().list_repo_files(
            self.repo_id, repo_type="dataset", revision=self.revision)
