# Repository Guidelines

## Project Structure & Module Organization
- Core app lives in `video2spritesheet/`: `main.py` starts the PySide6 GUI; `gui/` holds widgets (main window, settings panel, file picker); `core/` contains scaffolding for video loading, frame extraction, sheet building, and manifest writing; `utils/` has validation and filesystem helpers.
- Keep sample assets under `assets/` (tiny clips only) and generated outputs under `artifacts/` (ignored). Test scaffolding belongs in `tests/` mirroring package paths; empty `.gitkeep` files mark currently unused folders.

## Build, Test, and Development Commands
- `python -m venv .venv` then `\\.venv\\Scripts\\activate` to isolate deps (Python 3.12+).
- `python -m pip install -r requirements.txt` for runtime deps; add `requirements-dev.txt` for tooling.
- Launch the GUI: `python -m video2spritesheet.main` (PySide6 required).
- Run tests when added: `python -m pytest tests -q`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and full type hints. Keep GUI logic thin; push processing into `core/`.
- Naming: modules/packages `snake_case`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Use docstrings to describe placeholder logic and future implementation hooks (ffmpeg/moviepy, Pillow/OpenCV).
- If you add linters/formatters, prefer `black` + `ruff`; commit after formatting.

## Testing Guidelines
- Use pytest; name files `test_*.py` aligned with module paths. Avoid importing heavy GUI modules in unit tests; isolate validators and core helpers where possible.
- Prefer lightweight fixtures (short clips, stub images) to keep runs fast. Add regression tests for validation, manifest generation, and frame selection math as real logic lands.

## Commit & Pull Request Guidelines
- Write imperative commit subjects (Conventional Commit prefixes welcome: `feat:`, `fix:`, etc.); wrap bodies at ~72 chars.
- PRs should summarize intent, list manual/automated test commands, and link issues/tasks. Include screenshots/GIFs of the GUI when behavior changes.
- Note performance or output-quality impacts and update docs/examples when flags or outputs change.

## Security & Configuration Tips
- Never commit secrets or service tokens; keep them in env vars or an ignored `.env`.
- Avoid committing raw videos or large binaries; downsample for samples or use Git LFS if necessary.
