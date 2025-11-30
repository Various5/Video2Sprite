"""Frame extraction logic using moviepy (lightweight, eager)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
from PIL import Image

from . import FrameInfo, GenerationSettings, VideoMetadata
from ..core.errors import ProcessingError
from .video_loader import _ensure_ffmpeg_available, _resolve_video_file_clip

logger = logging.getLogger(__name__)
MAX_FRAME_CAP = 400


def _compute_sample_times(
    metadata: VideoMetadata,
    frame_count: int | None,
    frame_interval: float | None,
    max_frames: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[float]:
    """Decide which timestamps to sample based on user input."""

    user_cap = max_frames if max_frames is not None else MAX_FRAME_CAP
    cap = min(user_cap, MAX_FRAME_CAP)
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


def extract_frames(settings: GenerationSettings, metadata: VideoMetadata) -> Tuple[List[Image.Image], List[FrameInfo]]:
    """Extract frames eagerly (legacy API)."""

    times = _compute_sample_times(
        metadata,
        settings.frame_count,
        settings.frame_interval,
        settings.max_frames,
        settings.start_time,
        settings.end_time,
    )
    frames = []
    infos = []
    for frame, info in iter_frames(settings, metadata, times):
        frames.append(frame)
        infos.append(info)
    if not frames:
        raise ProcessingError("No frames could be extracted from the video.")
    return frames, infos


def iter_frames(
    settings: GenerationSettings, metadata: VideoMetadata, times: list[float] | None = None
) -> Iterator[tuple[Image.Image, FrameInfo]]:
    """Yield frames one-by-one to reduce memory pressure."""

    clip_class = _resolve_video_file_clip()
    _ensure_ffmpeg_available()

    times = times or _compute_sample_times(
        metadata,
        settings.frame_count,
        settings.frame_interval,
        settings.max_frames,
        settings.start_time,
        settings.end_time,
    )
    logger.info("Extracting %s frames from %s (streaming)", len(times), settings.video_path)
    clip = None
    try:
        clip = clip_class(str(settings.video_path))
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

            yield image.convert("RGBA"), FrameInfo(index=idx, timestamp=ts, width=image.width, height=image.height)
    except Exception as exc:  # pragma: no cover - moviepy internals
        raise ProcessingError(f"Failed to open video: {exc}") from exc
    finally:
        try:
            if clip:
                clip.close()
        except Exception:
            pass


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
