"""Manifest writing logic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from . import FrameInfo, GenerationSettings
from ..utils import file_tools

logger = logging.getLogger(__name__)


def write_manifest(
    infos: Iterable[FrameInfo],
    settings: GenerationSettings,
    columns: int,
    rows: int,
) -> Path:
    """Create a JSON manifest describing frame coordinates."""

    if not settings.generate_manifest:
        raise ValueError("Manifest generation requested without flag set.")

    manifest_path = (settings.manifest_path or settings.output_path).with_suffix(".json")
    file_tools.ensure_directory(manifest_path.parent)

    frames_payload = {}
    for info in infos:
        frames_payload[f"frame_{info.index:04d}"] = {
            "x": info.x,
            "y": info.y,
            "width": info.width,
            "height": info.height,
            "timestamp": info.timestamp,
            "padding": settings.padding,
        }

    manifest = {
        "source": str(settings.video_path),
        "frames": frames_payload,
        "meta": {
            "columns": columns,
            "rows": rows,
            "spritesheet": str(settings.output_path.with_suffix(".png")),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest to %s", manifest_path)
    return manifest_path
