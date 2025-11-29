"""Domain-specific exceptions for the spritesheet generator."""

from pathlib import Path


class InvalidVideoError(ValueError):
    """Raised when the selected video file is missing or unreadable."""

    def __init__(self, path: Path, reason: str | None = None):
        message = f"Invalid video file: {path}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)


class ValidationError(ValueError):
    """Raised when user-provided settings fail validation."""


class ProcessingError(RuntimeError):
    """Raised when the pipeline fails unexpectedly."""
