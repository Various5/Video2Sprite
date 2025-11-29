# Video2Sprite

FastAPI + PySide toolkit to turn videos into spritesheets with a modern web UI, media library, and image utilities.

## Features
- Web UI (FastAPI + static HTML/CSS/JS) with authentication
- Media Library: drag/drop uploads, image/video previews, renaming, shared selection across tools
- Spritesheet builder: upload or reuse videos from the library, generate sheets + JSON manifest
- Image studio tools: resize, crop, keying, masks, flips, tone/color tweaks, background removal, etc.
- Organized artifacts under `artifacts/images`, `artifacts/videos`, and generated manifests in `artifacts/`

## Getting started
1) Python 3.12+, create venv, install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
2) Run the web server:
   ```bash
   uvicorn video2spritesheet.web.server:app --host 0.0.0.0 --port 8000
   ```
   Or use the systemd unit (`video2sprite.service`) if installed.
3) Open `http://<host>:8000/login`, register first user, then log in and use the UI.

## Media handling
- Uploads in the web UI land in:
  - Images: `artifacts/images/`
  - Videos: `artifacts/videos/`
- Media can be selected in the Library and reused by all tools without re-uploading.

## GUI (optional)
- Launch the PySide GUI with:
  ```bash
  python -m video2spritesheet.main
  ```

## Tests
```bash
python -m pytest tests -q
```
