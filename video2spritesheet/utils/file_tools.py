"""Filesystem helpers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)
    return path


def default_output_path(video_path: Path, suffix: str = ".png") -> Path:
    """Return a default output path next to the video file."""

    if not suffix.startswith("."):
        suffix = "." + suffix
    return video_path.with_suffix(suffix)


def list_files_with_extensions(root: Path, extensions: set[str]) -> list[Path]:
    """Return sorted list of files in root with given extensions."""

    if not root.exists():
        return []
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


def format_output_filename(video_path: Path, pattern: str | None, timestamp: int | None = None) -> str:
    """Format an output filename using optional pattern with {stem}, {ext}, {ts}."""

    stem = video_path.stem
    ext = video_path.suffix.lstrip(".")
    fields = {"stem": stem, "ext": ext}
    if timestamp is not None:
        fields["ts"] = timestamp
    if pattern:
        return pattern.format(**fields)
    return f"{stem}.png"
