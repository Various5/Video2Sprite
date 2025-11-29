"""Core processing scaffolding for spritesheet generation."""

__all__ = [
    "VideoMetadata",
    "GenerationSettings",
    "ProcessingOutcome",
    "FrameInfo",
]

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoMetadata:
    """Basic metadata for a source video."""

    width: int
    height: int
    fps: float
    duration_seconds: float


@dataclass
class GenerationSettings:
    """User-configurable settings used for spritesheet generation."""

    video_path: Path
    output_path: Path
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    frame_count: Optional[int] = None
    frame_interval: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    columns: Optional[int] = None
    rows: Optional[int] = None
    generate_manifest: bool = False
    manifest_path: Optional[Path] = None
    max_frames: Optional[int] = None
    background_color: Optional[tuple[int, int, int, int]] = None
    remove_black_background: bool = False
    chroma_key_color: Optional[tuple[int, int, int, int]] = None
    chroma_key_tolerance: Optional[int] = None
    auto_edge_cutout: bool = False
    padding: int = 0
    output_pattern: Optional[str] = None


@dataclass
class ProcessingOutcome:
    """Result paths produced by a generation run."""

    spritesheet_path: Path
    manifest_path: Optional[Path]


@dataclass
class FrameInfo:
    """Metadata for an extracted frame."""

    index: int
    timestamp: float
    width: int
    height: int
    x: int = 0
    y: int = 0
