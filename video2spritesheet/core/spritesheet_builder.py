"""Spritesheet composition using Pillow."""

from __future__ import annotations

import math
import logging
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageChops, ImageFilter

from . import FrameInfo, GenerationSettings
from ..utils import file_tools

logger = logging.getLogger(__name__)


def _resolve_grid(frame_count: int, columns: int | None, rows: int | None) -> tuple[int, int]:
    """Compute grid layout; prefer provided values."""

    if columns and rows:
        return columns, rows
    if columns:
        rows = math.ceil(frame_count / columns)
        return columns, rows
    if rows:
        columns = math.ceil(frame_count / rows)
        return columns, rows

    # Square-ish fallback
    columns = math.ceil(math.sqrt(frame_count))
    rows = math.ceil(frame_count / columns)
    return columns, rows


def build_spritesheet(
    frames: List[Image.Image],
    infos: List[FrameInfo],
    settings: GenerationSettings,
) -> Tuple[Path, Image.Image, List[FrameInfo]]:
    """Backwards-compatible wrapper using streaming builder."""

    frame_items = [(frame, info) for frame, info in zip(frames, infos)]
    return build_spritesheet_streaming(frame_items, len(frames), settings)


def build_spritesheet_streaming(
    frame_items: Iterable[tuple[Image.Image, FrameInfo]],
    frame_count: int,
    settings: GenerationSettings,
) -> Tuple[Path, Image.Image, List[FrameInfo]]:
    """Pack frames into a spritesheet image and persist to disk."""

    output_path = settings.output_path.with_suffix(".png")
    file_tools.ensure_directory(output_path.parent)

    frame_iter = iter(frame_items)
    try:
        first_frame, first_info = next(frame_iter)
    except StopIteration:
        raise ValueError("No frames provided to pack.")

    columns, rows = _resolve_grid(frame_count, settings.columns, settings.rows)

    frame_width, frame_height = first_frame.size
    pad = max(0, settings.padding)
    cell_w = frame_width + pad
    cell_h = frame_height + pad
    sheet_width = columns * cell_w - pad
    sheet_height = rows * cell_h - pad
    bg = settings.background_color or (0, 0, 0, 0)
    sheet = Image.new("RGBA", (sheet_width, sheet_height), bg)

    updated_infos: list[FrameInfo] = []
    all_frames = chain([(first_frame, first_info)], frame_iter)
    for idx, (frame, info) in enumerate(all_frames):
        frame_to_paste = _apply_chroma_key(frame, settings) if settings.remove_black_background else frame
        col = idx % columns
        row = idx // columns
        x = col * cell_w
        y = row * cell_h
        sheet.paste(frame_to_paste, (x, y))
        updated_infos.append(
            FrameInfo(index=info.index, timestamp=info.timestamp, width=frame_width, height=frame_height, x=x, y=y)
        )

    sheet.save(output_path)
    logger.info("Wrote spritesheet to %s", output_path)
    return output_path, sheet, updated_infos


def _apply_chroma_key(image: Image.Image, settings: GenerationSettings) -> Image.Image:
    """Convert pixels near the key color to transparent with optional edge-aware mask."""

    img = image.convert("RGBA")
    key = settings.chroma_key_color or (0, 0, 0, 255)
    tolerance = settings.chroma_key_tolerance or 12

    solid = Image.new("RGBA", img.size, key)
    diff = ImageChops.difference(img, solid).convert("L")
    base_mask = diff.point(lambda v: 255 if v > tolerance else 0)

    if settings.auto_edge_cutout:
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edges_mask = edges.point(lambda v: 255 if v > 12 else 0)
        mask = ImageChops.lighter(base_mask, edges_mask)
    else:
        mask = base_mask

    img.putalpha(mask)
    return img
