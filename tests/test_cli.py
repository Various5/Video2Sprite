import pytest

from video2sprite import cli


def test_build_parser_creates_arguments():
    parser = cli.build_parser()
    args = parser.parse_args(["input.mp4", "out.png", "--fps", "12", "--dry-run"])
    assert args.input.name == "input.mp4"
    assert args.output.name == "out.png"
    assert args.fps == 12
    assert args.dry_run is True


def test_main_dry_run_returns_zero():
    assert cli.main(["input.mp4", "out.png", "--dry-run"]) == 0


def test_main_raises_not_implemented_when_running():
    with pytest.raises(NotImplementedError):
        cli.main(["input.mp4", "out.png"])
