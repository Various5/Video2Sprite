"""FastAPI web surface for Video2SpriteSheet."""

from __future__ import annotations

__all__ = ["app"]

from .server import app  # noqa: E402  (app needs server definitions)
