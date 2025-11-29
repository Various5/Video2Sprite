"""Image utility helpers for the web surface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageEnhance
from PIL import ImageOps

from ..utils import file_tools, validators

logger = logging.getLogger(__name__)


def load_image(path: Path) -> Image.Image:
    """Load an image safely."""

    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGBA")


def save_image(image: Image.Image, path: Path) -> Path:
    """Persist an image to disk."""

    file_tools.ensure_directory(path.parent)
    image.save(path)
    return path


def apply_chroma_key(
    image: Image.Image,
    chroma: Tuple[int, int, int, int],
    tolerance: int = 12,
    edge_cutout: bool = True,
) -> Image.Image:
    """Remove pixels near the chroma key color."""

    key = chroma
    tol = max(0, min(tolerance, 255))
    solid = Image.new("RGBA", image.size, key)
    diff = ImageChops.difference(image, solid).convert("L")
    base_mask = diff.point(lambda v: 255 if v > tol else 0)

    if edge_cutout:
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        edges_mask = edges.point(lambda v: 255 if v > 12 else 0)
        mask = ImageChops.lighter(base_mask, edges_mask)
    else:
        mask = base_mask

    result = image.copy()
    result.putalpha(mask)
    return result


def to_mask(image: Image.Image) -> Image.Image:
    """Return an RGBA red-tinted mask from alpha channel."""

    img = image.convert("RGBA")
    alpha = img.split()[-1]
    red = Image.new("L", img.size, 255)
    transparent = Image.new("L", img.size, 0)
    return Image.merge("RGBA", (red, transparent, transparent, alpha))


def apply_resize(image: Image.Image, width: Optional[int], height: Optional[int]) -> Image.Image:
    """Resize while preserving aspect if one dimension missing."""

    if width is None and height is None:
        return image
    w, h = image.size
    if width is None:
        width = int(w * (height / h))
    if height is None:
        height = int(h * (width / w))
    return image.resize((int(width), int(height)))


def apply_rotate(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate expanding canvas."""

    return image.rotate(degrees, expand=True)


def apply_flips(image: Image.Image, horizontal: bool, vertical: bool) -> Image.Image:
    """Flip image horizontally/vertically."""

    result = image
    if horizontal:
        result = ImageOps.mirror(result)
    if vertical:
        result = ImageOps.flip(result)
    return result


def apply_grayscale(image: Image.Image) -> Image.Image:
    """Convert to grayscale while keeping alpha."""

    gray = ImageOps.grayscale(image)
    if image.mode == "RGBA":
        return Image.merge("RGBA", (gray, gray, gray, image.split()[-1]))
    return gray.convert("RGBA")


def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur."""

    return image.filter(ImageFilter.GaussianBlur(radius))


def flatten_background(image: Image.Image, color: Tuple[int, int, int, int]) -> Image.Image:
    """Composite onto a solid background."""

    bg = Image.new("RGBA", image.size, color)
    bg.paste(image, mask=image.split()[-1])
    return bg


def apply_crop(image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """Crop a region, clamping to image bounds."""

    x = max(0, x)
    y = max(0, y)
    width = max(1, width)
    height = max(1, height)
    box = (
        x,
        y,
        min(x + width, image.width),
        min(y + height, image.height),
    )
    return image.crop(box)


def apply_invert(image: Image.Image) -> Image.Image:
    """Invert RGB channels, keep alpha."""

    rgb = ImageOps.invert(image.convert("RGB"))
    if image.mode == "RGBA":
        return Image.merge("RGBA", (*rgb.split(), image.split()[-1]))
    return rgb.convert("RGBA")


def apply_brightness_contrast(image: Image.Image, brightness: Optional[float], contrast: Optional[float]) -> Image.Image:
    """Apply brightness and contrast adjustments."""

    result = image
    if brightness is not None and brightness > 0:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if contrast is not None and contrast > 0:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    return result
