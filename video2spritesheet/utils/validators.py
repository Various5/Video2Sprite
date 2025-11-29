"""Validation helpers for user inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.errors import InvalidVideoError, ValidationError


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def validate_video_path(path: Path) -> Path:
    """Ensure the video path exists and appears to be a supported format."""

    if not path:
        raise InvalidVideoError(Path("<unset>"), reason="No path provided")
    if not path.exists():
        raise InvalidVideoError(path, reason="File not found")
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise InvalidVideoError(path, reason="Unsupported format")
    return path


def parse_optional_int(value: str | None, field: str) -> Optional[int]:
    """Parse a positive integer from a string value, if provided."""

    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValidationError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise ValidationError(f"{field} must be greater than zero")
    return parsed


def parse_optional_non_negative_int(value: str | None, field: str) -> Optional[int]:
    """Parse a non-negative integer (0 allowed) from a string value."""

    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValidationError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ValidationError(f"{field} must be zero or greater")
    return parsed


def parse_optional_float(value: str | None, field: str) -> Optional[float]:
    """Parse a positive float from a string value, if provided."""

    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValidationError(f"{field} must be a number") from exc
    if parsed <= 0:
        raise ValidationError(f"{field} must be greater than zero")
    return parsed


def validate_dimensions(width: Optional[int], height: Optional[int]) -> None:
    """Ensure dimensions are both set or both empty to avoid distortion."""

    if (width is None) ^ (height is None):
        raise ValidationError("Provide both width and height, or leave both blank to keep original")


def validate_frame_selection(frame_count: Optional[int], frame_interval: Optional[float]) -> None:
    """Prevent conflicting frame selection strategies."""

    if frame_count is not None and frame_interval is not None:
        raise ValidationError("Set either frame count or frame interval, not both")


def validate_grid(columns: Optional[int], rows: Optional[int]) -> None:
    """Ensure grid dimensions are positive if provided."""

    if columns is not None and columns <= 0:
        raise ValidationError("Columns must be greater than zero")
    if rows is not None and rows <= 0:
        raise ValidationError("Rows must be greater than zero")


def parse_optional_non_negative_float(value: str | None, field: str) -> Optional[float]:
    """Parse a non-negative float from a string value, if provided."""

    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValidationError(f"{field} must be a number") from exc
    if parsed < 0:
        raise ValidationError(f"{field} must be zero or greater")
    return parsed


def validate_time_range(start: Optional[float], end: Optional[float]) -> None:
    """Ensure start/end make sense."""

    if start is not None and end is not None and end <= start:
        raise ValidationError("End time must be greater than start time")


def parse_color_tuple(value: str | None) -> Optional[tuple[int, int, int, int]]:
    """Parse an RGBA color string like '255,0,0,255'."""

    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) not in (3, 4):
        raise ValidationError("Background color must be R,G,B[,A]")
    try:
        numbers = [int(p) for p in parts]
    except ValueError as exc:
        raise ValidationError("Background color must be numeric R,G,B[,A]") from exc
    if len(numbers) == 3:
        numbers.append(255)
    if any(n < 0 or n > 255 for n in numbers):
        raise ValidationError("Background color values must be between 0 and 255")
    return tuple(numbers)  # type: ignore


def validate_tolerance(value: Optional[int], field: str = "Tolerance") -> None:
    """Ensure tolerance is within a safe range."""

    if value is None:
        return
    if value < 0 or value > 255:
        raise ValidationError(f"{field} must be between 0 and 255")
