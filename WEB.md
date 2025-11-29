# Video2SpriteSheet Web Service

FastAPI-powered web surface for turning clips into spritesheets and JSON manifests without launching the PySide GUI.

## Quick start

1) Install deps (Python 3.12+):
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```
2) Run the server:
```
uvicorn video2spritesheet.web.server:app --host 0.0.0.0 --port 8000 --reload
```
3) Open the UI at http://localhost:8000/ and drop in a video.

Artifacts (spritesheets + manifests) land in `artifacts/` and are served at `/artifacts`.

## API

- `GET /health` – readiness probe.
- `GET /api/artifacts` – list generated spritesheets/manifests.
- `POST /api/generate` – multipart upload with:
  - `video`: file (mp4/mov/avi/mkv/webm)
  - `settings`: JSON string with any of:
    - `output_width`, `output_height`
    - `frame_count`, `frame_interval`, `start_time`, `end_time`, `max_frames`
    - `columns`, `rows`, `padding`
    - `generate_manifest` (bool)
    - `background_color`, `chroma_key_color` (`"R,G,B[,A]"`)
    - `chroma_key_tolerance` (0-255), `remove_black_background` (bool), `auto_edge_cutout` (bool)
    - `output_pattern` (e.g., `"{stem}_sheet_{ts}"`)

Example curl:
```
curl -X POST http://localhost:8000/api/generate ^
  -F "video=@assets/sample.mp4" ^
  -F "settings={\"frame_interval\":0.08,\"generate_manifest\":true}"
```
