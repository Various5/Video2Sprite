"""Video loading and metadata discovery (placeholder implementation)."""

from __future__ import annotations

import logging
from pathlib import Path

from . import VideoMetadata
from .errors import InvalidVideoError, ProcessingError
from ..utils import validators

logger = logging.getLogger(__name__)


def load_metadata(video_path: Path) -> VideoMetadata:
    """Return basic metadata for the selected video.

    Placeholder: In production, load actual metadata via ffmpeg or moviepy.
    """

    validated_path = validators.validate_video_path(video_path)
    _ensure_ffmpeg_available()
    clip_class = _resolve_video_file_clip()

    try:
        with clip_class(str(validated_path)) as clip:
            width, height = clip.size
            fps = float(getattr(clip, "fps", 24.0) or 24.0)
            duration_seconds = float(getattr(clip, "duration", 1.0) or 1.0)
    except Exception as exc:  # pragma: no cover - backend dependent
        raise InvalidVideoError(validated_path, reason=f"Could not read metadata: {exc}") from exc

    logger.debug(
        "Loaded placeholder metadata for %s -> %sx%s @ %sfps, %ss",
        validated_path,
        width,
        height,
        fps,
        duration_seconds,
    )
    return VideoMetadata(width=width, height=height, fps=fps, duration_seconds=duration_seconds)


def ensure_video_is_supported(video_path: Path) -> None:
    """Validate the video path and surface a friendly error if unsupported."""

    try:
        validators.validate_video_path(video_path)
    except InvalidVideoError:
        raise
    except Exception as exc:  # pragma: no cover - future backend hook
        raise InvalidVideoError(video_path, reason=str(exc)) from exc


def _ensure_ffmpeg_available() -> None:
    """Raise a friendly error if ffmpeg is missing."""

    try:
        from moviepy.config import FFMPEG_BINARY  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ProcessingError("moviepy is not installed. Run pip install -r requirements.txt.") from exc

    if not FFMPEG_BINARY:
        raise ProcessingError("ffmpeg not found. Install ffmpeg and ensure it is on PATH.")


def _resolve_video_file_clip():
    """Import VideoFileClip from supported moviepy locations."""

    try:
        from moviepy.editor import VideoFileClip  # type: ignore
        return VideoFileClip
    except ModuleNotFoundError:
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
            return VideoFileClip
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ProcessingError("moviepy is not installed. Run pip install -r requirements.txt.") from exc
