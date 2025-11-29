"""Command-line entry point for video-to-sprite workflows."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2sprite",
        description="Convert a source video into a packed sprite sheet and metadata.",
    )
    parser.add_argument("input", type=Path, help="Path to source video clip")
    parser.add_argument("output", type=Path, help="Destination sprite sheet path (PNG)")
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Override frames per second used for extraction (default: 24)",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize frames before packing (px)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON manifest output for frame timings and positions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and show plan without rendering outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        parser.print_help()
        return 0

    # Placeholder: actual extraction and packing logic should live in dedicated modules.
    raise NotImplementedError(
        "Sprite generation is not yet implemented. Implement frame extraction and packing."
    )


if __name__ == "__main__":
    sys.exit(main())
