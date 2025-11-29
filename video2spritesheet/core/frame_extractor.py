"""Frame extraction logic using moviepy (lightweight, eager)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from . import FrameInfo, GenerationSettings, VideoMetadata
from ..core.errors import ProcessingError
from .video_loader import _ensure_ffmpeg_available, _resolve_video_file_clip

logger = logging.getLogger(__name__)


def _compute_sample_times(
    metadata: VideoMetadata,
    frame_count: int | None,
    frame_interval: float | None,
    max_frames: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[float]:
    """Decide which timestamps to sample based on user input."""

    cap = 180 if max_frames is None else max_frames
    duration = metadata.duration_seconds
    start = max(0.0, start_time or 0.0)
    stop = min(end_time if end_time is not None else duration, duration)
    if stop <= start:
        stop = duration

    if frame_interval:
        times = list(np.arange(start, stop, frame_interval, dtype=float))
    elif frame_count:
        times = np.linspace(start, stop, num=frame_count, endpoint=False, dtype=float).tolist()
    else:
        # Default: roughly half the frames over the selected span, capped to max_frames.
        window = stop - start
        default_count = int(metadata.fps * window / 2) if window > 0 else int(metadata.fps * duration / 2)
        default_count = max(default_count, 1)
        if cap and cap > 0:
            default_count = min(default_count, cap)
        times = np.linspace(start, stop, num=default_count, endpoint=False, dtype=float).tolist()

    times = [min(t, max(duration - 0.001, 0.0)) for t in times]  # keep within clip
    unique_times = sorted(dict.fromkeys(times))  # preserve order, remove dupes
    if cap and cap > 0 and len(unique_times) > cap:
        logger.info("Capping frames to %s for memory safety (requested %s)", cap, len(unique_times))
        unique_times = unique_times[:cap]
    return unique_times


def extract_frames(
    settings: GenerationSettings, metadata: VideoMetadata
) -> Tuple[List[Image.Image], List[FrameInfo]]:
    """Extract frames as PIL Images along with FrameInfo metadata."""

    clip_class = _resolve_video_file_clip()
    _ensure_ffmpeg_available()

    times = _compute_sample_times(
        metadata,
        settings.frame_count,
        settings.frame_interval,
        settings.max_frames,
        settings.start_time,
        settings.end_time,
    )
    logger.info("Extracting %s frames from %s", len(times), settings.video_path)
    try:
        clip = clip_class(str(settings.video_path))
        frames: list[Image.Image] = []
        infos: list[FrameInfo] = []
        for idx, ts in enumerate(times):
            try:
                frame_array = clip.get_frame(ts)
                image = Image.fromarray(frame_array)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to decode frame at %.3fs: %s", ts, exc)
                continue

            target_size = _target_size(settings, metadata)
            if target_size:
                image = image.resize(target_size)

            frames.append(image.convert("RGBA"))
            infos.append(FrameInfo(index=idx, timestamp=ts, width=image.width, height=image.height))
    except Exception as exc:  # pragma: no cover - moviepy internals
        raise ProcessingError(f"Failed to open video: {exc}") from exc
    finally:
        try:
            clip.close()
        except Exception:
            pass

    if not frames:
        raise ProcessingError("No frames could be extracted from the video.")

    return frames, infos


def iter_frame_batches(frames: Iterable[Image.Image], batch_size: int = 50) -> Iterable[list[Image.Image]]:
    """Yield frames in batches."""

    batch: list[Image.Image] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _target_size(settings: GenerationSettings, metadata: VideoMetadata) -> tuple[int, int] | None:
    """Compute output size if provided; else keep original."""

    width = settings.output_width or metadata.width
    height = settings.output_height or metadata.height
    return (width, height)
