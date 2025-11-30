"""FastAPI surface for Video2SpriteSheet processing."""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import time
from hashlib import pbkdf2_hmac
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageChops, ImageFilter, ImageEnhance
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

import os
import uuid

from ..core import GenerationSettings, ProcessingOutcome
from ..core import frame_extractor, manifest_writer, spritesheet_builder, video_loader
from ..core.errors import InvalidVideoError, ProcessingError, ValidationError
from ..utils import file_tools, validators
from . import image_tools

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MEDIA_IMAGES_DIR = ARTIFACTS_DIR / "images"
MEDIA_VIDEOS_DIR = ARTIFACTS_DIR / "videos"
PREVIEWS_DIR = ARTIFACTS_DIR / "previews"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"
LOGIN_PATH = STATIC_DIR / "login.html"
USERS_PATH = BASE_DIR / "config" / "users.json"
USAGE_PATH = BASE_DIR / "config" / "usage.json"
SESSION_TTL_SECONDS = 7 * 24 * 3600
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB guardrail
MAX_FRAME_CAP = 400
PREVIEW_TTL_SECONDS = 60 * 30  # 30 minutes
PER_USER_QUOTA = int(os.environ.get("V2S_USER_QUOTA_MB", "1024")) * 1024 * 1024  # default 1GB
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("V2S_ALLOWED_ORIGINS", "https://v2s.drevalis.com,http://localhost:8000").split(",")
    if origin.strip()
]
file_tools.ensure_directory(ARTIFACTS_DIR)
file_tools.ensure_directory(MEDIA_IMAGES_DIR)
file_tools.ensure_directory(MEDIA_VIDEOS_DIR)
file_tools.ensure_directory(PREVIEWS_DIR)
file_tools.ensure_directory(USERS_PATH.parent)

_SESSIONS: dict[str, dict[str, Any]] = {}


def _hash_password(password: str, salt: Optional[bytes] = None) -> dict[str, str]:
    salt_bytes = salt or secrets.token_bytes(16)
    hashed = pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return {"salt": salt_bytes.hex(), "hash": hashed.hex()}


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt_bytes = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(hash_hex)
    check = pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return secrets.compare_digest(check, expected)


def _load_users() -> list[dict[str, str]]:
    if not USERS_PATH.exists():
        return []
    try:
        with USERS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.error("Failed to read users.json, refusing to open registration: %s", exc)
        raise HTTPException(status_code=500, detail="User store unavailable") from exc


def _load_usage() -> dict[str, int]:
    if not USAGE_PATH.exists():
        return {}
    try:
        with USAGE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read usage.json, starting fresh: %s", exc)
        return {}


def _save_usage(usage: dict[str, int]) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(usage, handle, indent=2)


def _current_usage(user: str) -> int:
    usage = _load_usage()
    return int(usage.get(user, 0))


def _ensure_quota(user: str, incoming_bytes: int) -> None:
    """Guardrail for per-user quota before accepting a write."""

    if incoming_bytes <= 0:
        return
    if _current_usage(user) + incoming_bytes > PER_USER_QUOTA:
        raise HTTPException(status_code=507, detail="Quota exceeded for user")


def _record_usage(user: str, delta: int) -> None:
    usage = _load_usage()
    usage[user] = max(0, int(usage.get(user, 0)) + delta)
    _save_usage(usage)


def _save_users(users: list[dict[str, str]]) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)


def _issue_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    _SESSIONS[token] = {"user": username, "expires": time.time() + SESSION_TTL_SECONDS, "csrf": csrf}
    return token


def _get_session(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    data = _SESSIONS.get(token)
    if not data:
        return None
    if data["expires"] < time.time():
        _SESSIONS.pop(token, None)
        return None
    return data["user"]


def _get_session_record(token: Optional[str]) -> Optional[dict[str, Any]]:
    if not token:
        return None
    data = _SESSIONS.get(token)
    if not data:
        return None
    if data["expires"] < time.time():
        _SESSIONS.pop(token, None)
        return None
    return data


class GenerationRequest(BaseModel):
    """Incoming settings payload for spritesheet generation."""

    output_width: Optional[int] = Field(None, ge=1)
    output_height: Optional[int] = Field(None, ge=1)
    frame_count: Optional[int] = Field(None, ge=1)
    frame_interval: Optional[float] = Field(None, gt=0)
    start_time: Optional[float] = Field(None, ge=0)
    end_time: Optional[float] = Field(None, ge=0)
    columns: Optional[int] = Field(None, ge=1)
    rows: Optional[int] = Field(None, ge=1)
    max_frames: Optional[int] = Field(None, ge=0)
    padding: int = Field(0, ge=0, le=64)
    generate_manifest: bool = True
    background_color: Optional[tuple[int, int, int, int]] = None
    remove_black_background: bool = True
    chroma_key_color: Optional[tuple[int, int, int, int]] = None
    chroma_key_tolerance: int = Field(12, ge=0, le=255)
    auto_edge_cutout: bool = True
    output_pattern: Optional[str] = None
    preview_only: bool = False

    @field_validator("background_color", "chroma_key_color", mode="before")
    @classmethod
    def _parse_color(cls, value):
        if value in (None, "", "null"):
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 3:
                return (*value, 255)
            return tuple(value)
        if isinstance(value, str):
            return validators.parse_color_tuple(value)
        raise ValueError("Color must be R,G,B[,A]")

    @field_validator("max_frames")
    @classmethod
    def _normalize_max_frames(cls, value):
        if value == 0:
            return None
        return value


class GenerationResponse(BaseModel):
    """Payload returned after generation completes."""

    spritesheet_url: str
    manifest_url: Optional[str] = None
    frame_count: int
    columns: int
    rows: int
    width: int
    height: int
    frame_width: int
    frame_height: int
    video_metadata: dict[str, Any]


class ImageProcessRequest(BaseModel):
    """Incoming image processing options."""

    chroma_key_color: Optional[tuple[int, int, int, int]] = None
    chroma_key_tolerance: int = Field(12, ge=0, le=255)
    auto_edge_cutout: bool = True
    to_mask: bool = False
    resize_width: Optional[int] = Field(None, ge=1)
    resize_height: Optional[int] = Field(None, ge=1)
    rotate_degrees: Optional[float] = None
    flip_horizontal: bool = False
    flip_vertical: bool = False
    grayscale: bool = False
    blur_radius: Optional[float] = Field(None, ge=0)
    flatten_background: Optional[tuple[int, int, int, int]] = None
    crop_x: Optional[int] = Field(None, ge=0)
    crop_y: Optional[int] = Field(None, ge=0)
    crop_width: Optional[int] = Field(None, ge=1)
    crop_height: Optional[int] = Field(None, ge=1)
    brightness: Optional[float] = Field(None, gt=0)
    contrast: Optional[float] = Field(None, gt=0)
    invert_colors: bool = False

    @field_validator("chroma_key_color", "flatten_background", mode="before")
    @classmethod
    def _parse_color(cls, value):
        if value in (None, "", "null"):
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 3:
                return (*value, 255)
            return tuple(value)
        if isinstance(value, str):
            return validators.parse_color_tuple(value)
        raise ValueError("Color must be R,G,B[,A]")


class SeamlessRequest(BaseModel):
    """Incoming options for seamless texture generation."""

    tile_size: Optional[int] = Field(None, ge=16)
    feather: float = Field(8.0, ge=0.0)


class ParallaxRequest(BaseModel):
    """Incoming options for parallax layer preview."""

    segments: int = Field(3, ge=2, le=6)
    samples: int = Field(8, ge=1, le=50)


def require_auth(request: Request) -> str:
    """Dependency to enforce session-based auth."""

    token = request.cookies.get("v2s_session")
    user = _get_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def create_app() -> FastAPI:
    app = FastAPI(title="Video2SpriteSheet Web", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")

    @app.middleware("http")
    async def auth_gate(request: Request, call_next):
        allowed = (
            request.url.path.startswith("/health")
            or request.url.path.startswith("/auth/")
            or request.url.path.startswith("/login")
            or request.url.path.startswith("/static")
        )
        if not allowed:
            user = _get_session(request.cookies.get("v2s_session"))
            if not user:
                return RedirectResponse(url="/login", status_code=307)
        # CSRF: require header for state-changing requests when session exists
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not request.url.path.startswith("/auth/"):
            session = _get_session_record(request.cookies.get("v2s_session"))
            if session:
                header_token = request.headers.get("x-csrf-token")
                if not header_token or header_token != session.get("csrf"):
                    raise HTTPException(status_code=403, detail="CSRF token missing or invalid")
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login")
    async def login_page() -> FileResponse:
        return FileResponse(LOGIN_PATH)

    @app.get("/")
    async def index(user: str = Depends(require_auth)) -> FileResponse:
        return FileResponse(INDEX_PATH)

    @app.get("/api/artifacts")
    async def list_artifacts(user: str = Depends(require_auth)) -> dict[str, list[str]]:
        images = file_tools.list_files_with_extensions(MEDIA_IMAGES_DIR, {".png", ".jpg", ".jpeg"})
        videos = file_tools.list_files_with_extensions(MEDIA_VIDEOS_DIR, {".mp4", ".mov", ".avi", ".mkv", ".webm"})
        manifests = file_tools.list_files_with_extensions(ARTIFACTS_DIR, {".json"})
        return {
            "images": [f"/artifacts/images/{p.name}" for p in images],
            "videos": [f"/artifacts/videos/{p.name}" for p in videos],
            "manifests": [f"/artifacts/{p.name}" for p in manifests],
        }

    @app.post("/api/media/upload")
    async def upload_media(request: Request, file: UploadFile = File(...), user: str = Depends(require_auth)) -> dict[str, str]:
        suffix = Path(file.filename or "").suffix.lower()
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        if suffix in image_exts:
            target_dir = MEDIA_IMAGES_DIR
            media_type = "image"
        elif suffix in video_exts:
            target_dir = MEDIA_VIDEOS_DIR
            media_type = "video"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        _enforce_size_limit(request)
        content_len = request.headers.get("content-length")
        if content_len and content_len.isdigit():
            _ensure_quota(user, int(content_len))
        target_dir.mkdir(parents=True, exist_ok=True)
        target = _write_upload_file(file, target_dir, suffix or ".dat", owner=user)
        _basic_safety_check(target, media_type)
        _record_usage(user, target.stat().st_size)
        rel = target.relative_to(ARTIFACTS_DIR)
        return {"status": "ok", "type": media_type, "path": f"/artifacts/{rel.as_posix()}"}

    @app.delete("/api/media")
    async def delete_media(path: str, user: str = Depends(require_auth)) -> dict[str, str]:
        if not path.startswith("/artifacts/"):
            raise HTTPException(status_code=400, detail="Invalid path")
        target = ARTIFACTS_DIR / path.replace("/artifacts/", "", 1)
        try:
            target = target.resolve(strict=True)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Not found") from None
        if ARTIFACTS_DIR not in target.parents and target != ARTIFACTS_DIR:
            raise HTTPException(status_code=400, detail="Invalid location")
        try:
            size = target.stat().st_size if target.exists() else 0
            target.unlink()
            if size:
                _record_usage(user, -size)
            return {"status": "deleted"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete: {exc}") from exc

    @app.post("/api/media/rename")
    async def rename_media(
        old_path: str = Form(...),
        new_name: str = Form(...),
        user: str = Depends(require_auth),
    ) -> dict[str, str]:
        if not old_path.startswith("/artifacts/"):
            raise HTTPException(status_code=400, detail="Invalid path")
        src = ARTIFACTS_DIR / old_path.replace("/artifacts/", "", 1)
        try:
            src = src.resolve(strict=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Not found") from exc
        if ARTIFACTS_DIR not in src.parents and src != ARTIFACTS_DIR:
            raise HTTPException(status_code=400, detail="Invalid location")
        if not new_name:
            raise HTTPException(status_code=400, detail="New name required")
        suffix = src.suffix
        dst = src.with_name(f"{Path(new_name).stem}{suffix}")
        if dst.exists():
            raise HTTPException(status_code=400, detail="Target already exists")
        try:
            src.rename(dst)
            rel = dst.relative_to(ARTIFACTS_DIR)
            return {"status": "renamed", "path": f"/artifacts/{rel.as_posix()}"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Rename failed: {exc}") from exc

    @app.post("/api/process-image")
    async def process_image(
        request: Request,
        image: UploadFile | None = File(None),
        media_path: Optional[str] = Form(None),
        settings: str = Form("{}"),
        user: str = Depends(require_auth),
    ) -> dict[str, str]:
        try:
            payload = json.loads(settings) if settings else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {exc}") from exc

        try:
            request_settings = ImageProcessRequest.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            local_image: Optional[Path] = None
            if media_path:
                local_image = _resolve_media_path(media_path, MEDIA_IMAGES_DIR, {".png", ".jpg", ".jpeg", ".webp"})
            elif image:
                _enforce_size_limit(request)
                _ensure_quota(user, int(request.headers.get("content-length", "0") or 0))
                local_image = await _persist_upload(image, subdir="images", owner=user)
            else:
                raise HTTPException(status_code=400, detail="Provide an image or media_path")
            result_path = await run_in_threadpool(
                _run_image_processing,
                local_image,
                request_settings,
                Path(image.filename) if image and image.filename else Path(media_path or "image.png"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Image processing failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if result_path.exists():
            _record_usage(user, result_path.stat().st_size)
        return {"image_url": _artifact_url(result_path)}

    @app.post("/api/generate", response_model=GenerationResponse)
    async def generate_spritesheet(
        background_tasks: BackgroundTasks,
        request: Request,
        video: UploadFile | None = File(None),
        media_path: Optional[str] = Form(None),
        settings: str = Form("{}"),
        user: str = Depends(require_auth),
    ) -> GenerationResponse:
        local_video: Optional[Path] = None
        try:
            payload = json.loads(settings) if settings else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {exc}") from exc

        try:
            request_settings = GenerationRequest.model_validate(payload)
        except Exception as exc:  # pragma: no cover - validation error messaging
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            if media_path:
                local_video = _resolve_media_path(media_path, MEDIA_VIDEOS_DIR, {".mp4", ".mov", ".avi", ".mkv", ".webm"})
            elif video:
                _enforce_size_limit(request)
                _ensure_quota(user, int(request.headers.get("content-length", "0") or 0))
                local_video = await _persist_upload(video, owner=user)
            else:
                raise HTTPException(status_code=400, detail="Provide a video or media_path")
            if request_settings.max_frames is None or request_settings.max_frames > MAX_FRAME_CAP:
                request_settings.max_frames = MAX_FRAME_CAP
            result = await run_in_threadpool(_run_generation, local_video, request_settings)
        except (InvalidVideoError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProcessingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected failure during generation")
            raise HTTPException(status_code=500, detail="Unexpected error") from exc
        finally:
            if local_video:
                background_tasks.add_task(_cleanup_file, local_video)

        # Record usage or schedule cleanup
        total_size = 0
        if result.outcome.spritesheet_path and result.outcome.spritesheet_path.exists():
            total_size += result.outcome.spritesheet_path.stat().st_size
        if result.outcome.manifest_path and result.outcome.manifest_path.exists():
            total_size += result.outcome.manifest_path.stat().st_size
        if request_settings.preview_only:
            background_tasks.add_task(
                _cleanup_preview_after_delay,
                [result.outcome.spritesheet_path, result.outcome.manifest_path],
            )
        else:
            try:
                _ensure_quota(user, total_size)
            except HTTPException:
                _cleanup_preview([result.outcome.spritesheet_path, result.outcome.manifest_path])
                raise
            _record_usage(user, total_size)

        return GenerationResponse(
            spritesheet_url=_artifact_url(result.outcome.spritesheet_path),
            manifest_url=_artifact_url(result.outcome.manifest_path) if result.outcome.manifest_path else None,
            frame_count=result.frame_count,
            columns=result.columns,
            rows=result.rows,
            width=result.width,
            height=result.height,
            frame_width=result.frame_width,
            frame_height=result.frame_height,
            video_metadata={
                "width": result.metadata.width,
                "height": result.metadata.height,
                "fps": result.metadata.fps,
                "duration_seconds": result.metadata.duration_seconds,
            },
        )

    @app.post("/api/seamless-texture")
    async def seamless_texture(
        request: Request,
        image: UploadFile | None = File(None),
        media_path: Optional[str] = Form(None),
        tile_size: Optional[int] = Form(None),
        feather: float = Form(8.0),
        user: str = Depends(require_auth),
    ) -> dict[str, str]:
        try:
            local_image: Optional[Path] = None
            if media_path:
                local_image = _resolve_media_path(media_path, MEDIA_IMAGES_DIR, {".png", ".jpg", ".jpeg", ".webp"})
            elif image:
                _enforce_size_limit(request)
                _ensure_quota(user, int(request.headers.get("content-length", "0") or 0))
                local_image = await _persist_upload(image, subdir="images", owner=user)
            else:
                raise HTTPException(status_code=400, detail="Provide an image or media_path")
            request_settings = SeamlessRequest(tile_size=tile_size, feather=feather)
            out_path = await run_in_threadpool(_run_seamless_texture, local_image, request_settings)
            if out_path.exists():
                _record_usage(user, out_path.stat().st_size)
            return {"image_url": _artifact_url(out_path)}
        except Exception as exc:
            logger.exception("Seamless generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/parallax-preview")
    async def parallax_preview(
        request: Request,
        video: UploadFile | None = File(None),
        media_path: Optional[str] = Form(None),
        segments: int = Form(3),
        samples: int = Form(8),
        user: str = Depends(require_auth),
    ) -> dict[str, str]:
        try:
            local_video: Optional[Path] = None
            if media_path:
                local_video = _resolve_media_path(media_path, MEDIA_VIDEOS_DIR, {".mp4", ".mov", ".avi", ".mkv", ".webm"})
            elif video:
                _enforce_size_limit(request)
                _ensure_quota(user, int(request.headers.get("content-length", "0") or 0))
                local_video = await _persist_upload(video, owner=user)
            else:
                raise HTTPException(status_code=400, detail="Provide a video or media_path")
            request_settings = ParallaxRequest(segments=segments, samples=samples)
            paths = await run_in_threadpool(_run_parallax_layers, local_video, request_settings)
            for value in paths.values():
                if value.exists():
                    _record_usage(user, value.stat().st_size)
            return {
                "background_url": _artifact_url(paths["background"]),
                "mid_url": _artifact_url(paths["mid"]),
                "foreground_url": _artifact_url(paths["foreground"]),
            }
        except Exception as exc:
            logger.exception("Parallax preview failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


class GenerationResult(BaseModel):
    """Internal representation of a completed run."""

    outcome: ProcessingOutcome
    frame_count: int
    columns: int
    rows: int
    width: int
    height: int
    metadata: Any
    frame_width: int
    frame_height: int


def _artifact_url(path: Path | None) -> Optional[str]:
    if path is None:
        return None
    try:
        rel = path.relative_to(ARTIFACTS_DIR)
        return f"/artifacts/{rel.as_posix()}"
    except ValueError:
        return f"/artifacts/{path.name}"


async def _persist_upload(file: UploadFile, subdir: str = "uploads", owner: Optional[str] = None) -> Path:
    """Persist the uploaded file into artifacts/uploads for processing."""

    suffix = Path(file.filename or "upload").suffix or ".dat"
    target_dir = ARTIFACTS_DIR / subdir
    file_tools.ensure_directory(target_dir)
    return _write_upload_file(file, target_dir, suffix, owner=owner)


def _write_upload_file(file: UploadFile, target_dir: Path, suffix: str, owner: Optional[str] = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{owner}_" if owner else ""
    target = target_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
    written = 0
    with target.open("wb") as handle:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            handle.write(chunk)
    return target


def _resolve_media_path(path_str: str, base_dir: Path, allowed: set[str]) -> Path:
    """Resolve a media path within artifacts and validate extension."""

    path = Path(path_str)
    if path_str.startswith("/artifacts/"):
        path = ARTIFACTS_DIR / path_str.replace("/artifacts/", "", 1)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Media not found") from exc
    if base_dir not in resolved.parents and resolved.parent != base_dir:
        raise HTTPException(status_code=400, detail="Media outside allowed directory")
    if resolved.suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported media type")
    return resolved


def _basic_safety_check(path: Path, media_type: str) -> None:
    """Lightweight validation to ensure file is decodable."""

    try:
        if media_type == "image":
            with Image.open(path) as img:
                img.verify()
        elif media_type == "video":
            video_loader.load_metadata(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File failed validation: {exc}") from exc


def _enforce_size_limit(request: Request) -> None:
    """Simple guardrail on upload size based on Content-Length."""

    content_length = request.headers.get("content-length")
    if not content_length:
        return
    try:
        size = int(content_length)
    except ValueError:
        return
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds limit")


def _cleanup_file(path: Path) -> None:
    """Remove a temporary upload if it exists."""

    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.debug("Cleanup failed for %s", path)


def _cleanup_preview(paths: list[Path | None]) -> None:
    """Delete preview artifacts."""

    for path in paths:
        if path is None:
            continue
        try:
            path.unlink()
        except Exception:
            logger.debug("Preview cleanup failed for %s", path)


def _cleanup_preview_after_delay(paths: list[Path | None], delay: int = PREVIEW_TTL_SECONDS) -> None:
    """Sleep briefly then remove preview files."""

    try:
        time.sleep(delay)
    except Exception:
        pass
    _cleanup_preview(paths)


def _run_generation(video_path: Path, request: GenerationRequest) -> GenerationResult:
    """Perform the full pipeline for a single upload."""

    # Validate video and discover metadata
    validators.validate_video_path(video_path)
    metadata = video_loader.load_metadata(video_path)

    base_dir = PREVIEWS_DIR if request.preview_only else ARTIFACTS_DIR
    timestamp = int(time.time())
    filename = file_tools.format_output_filename(
        video_path, request.output_pattern or "{stem}_sheet_{ts}", timestamp=timestamp
    )
    output_path = base_dir / filename

    background_color = request.background_color
    chroma_color = request.chroma_key_color or background_color or (0, 0, 0, 255)

    capped_frames = (
        min(request.max_frames, MAX_FRAME_CAP) if request.max_frames is not None else MAX_FRAME_CAP
    )
    settings = GenerationSettings(
        video_path=video_path,
        output_path=output_path,
        output_width=request.output_width or metadata.width,
        output_height=request.output_height or metadata.height,
        frame_count=request.frame_count,
        frame_interval=request.frame_interval,
        start_time=request.start_time,
        end_time=request.end_time,
        columns=request.columns,
        rows=request.rows,
        generate_manifest=request.generate_manifest,
        manifest_path=output_path.with_suffix(".json") if request.generate_manifest else None,
        max_frames=capped_frames,
        background_color=background_color,
        remove_black_background=request.remove_black_background,
        chroma_key_color=chroma_color,
        chroma_key_tolerance=request.chroma_key_tolerance,
        auto_edge_cutout=request.auto_edge_cutout,
        padding=request.padding,
        output_pattern=request.output_pattern,
    )

    times = frame_extractor._compute_sample_times(  # type: ignore[attr-defined]
        metadata,
        request.frame_count,
        request.frame_interval,
        request.max_frames,
        request.start_time,
        request.end_time,
    )
    frame_iter = frame_extractor.iter_frames(settings, metadata, times)
    sheet_path, sheet_image, packed_infos = spritesheet_builder.build_spritesheet_streaming(
        frame_iter, len(times), settings
    )
    columns, rows = spritesheet_builder._resolve_grid(len(times), settings.columns, settings.rows)
    manifest_path = None
    if settings.generate_manifest:
        manifest_path = manifest_writer.write_manifest(packed_infos, settings, columns, rows)

    outcome = ProcessingOutcome(spritesheet_path=sheet_path, manifest_path=manifest_path)
    width, height = sheet_image.size
    frame_width, frame_height = packed_infos[0].width, packed_infos[0].height
    return GenerationResult(
        outcome=outcome,
        frame_count=len(frames),
        columns=columns,
        rows=rows,
        width=width,
        height=height,
        metadata=metadata,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _run_seamless_texture(path: Path, settings: SeamlessRequest) -> Path:
    """Build a simple seamless-like texture by offsetting and optional blur."""

    img = image_tools.load_image(path).convert("RGBA")
    if settings.tile_size:
        img = img.resize((settings.tile_size, settings.tile_size), Image.LANCZOS)
    shifted = ImageChops.offset(img, img.width // 2, img.height // 2)
    if settings.feather and settings.feather > 0:
        shifted = shifted.filter(ImageFilter.GaussianBlur(radius=settings.feather))
    out_path = ARTIFACTS_DIR / f"{path.stem}_seamless.png"
    return image_tools.save_image(shifted, out_path)


def _run_parallax_layers(path: Path, settings: ParallaxRequest) -> dict[str, Path]:
    """Create a placeholder parallax preview by splitting a representative frame."""

    try:
        from moviepy.editor import VideoFileClip  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing unavailable: {exc}") from exc

    with VideoFileClip(str(path), audio=False) as clip:
        duration = clip.duration or 0
        t = max(0.0, min(duration * 0.5, max(duration - 0.1, 0)))
        frame = clip.get_frame(t)
    img = Image.fromarray(frame).convert("RGBA")
    mid = img.copy()
    bg = img.filter(ImageFilter.GaussianBlur(radius=6))
    fg = img.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=3))

    bg_path = ARTIFACTS_DIR / f"{path.stem}_parallax_bg.png"
    mid_path = ARTIFACTS_DIR / f"{path.stem}_parallax_mid.png"
    fg_path = ARTIFACTS_DIR / f"{path.stem}_parallax_fg.png"

    image_tools.save_image(bg, bg_path)
    image_tools.save_image(mid, mid_path)
    image_tools.save_image(fg, fg_path)
    return {"background": bg_path, "mid": mid_path, "foreground": fg_path}


def _run_image_processing(path: Path, settings: ImageProcessRequest, original_name: Path) -> Path:
    """Apply image utilities and write a result file."""

    img = image_tools.load_image(path)
    result = img
    if settings.crop_x is not None and settings.crop_y is not None and settings.crop_width and settings.crop_height:
        result = image_tools.apply_crop(result, settings.crop_x, settings.crop_y, settings.crop_width, settings.crop_height)
    if settings.resize_width or settings.resize_height:
        result = image_tools.apply_resize(result, settings.resize_width, settings.resize_height)
    if settings.rotate_degrees:
        result = image_tools.apply_rotate(result, settings.rotate_degrees)
    if settings.flip_horizontal or settings.flip_vertical:
        result = image_tools.apply_flips(result, settings.flip_horizontal, settings.flip_vertical)
    if settings.grayscale:
        result = image_tools.apply_grayscale(result)
    if settings.invert_colors:
        result = image_tools.apply_invert(result)
    if settings.brightness or settings.contrast:
        result = image_tools.apply_brightness_contrast(result, settings.brightness, settings.contrast)
    if settings.blur_radius:
        result = image_tools.apply_blur(result, settings.blur_radius)
    if settings.chroma_key_color:
        result = image_tools.apply_chroma_key(
            result, settings.chroma_key_color, settings.chroma_key_tolerance, settings.auto_edge_cutout
        )
    if settings.to_mask:
        result = image_tools.to_mask(result)
    if settings.flatten_background:
        result = image_tools.flatten_background(result, settings.flatten_background)

    out_name = f"{original_name.stem}_processed{original_name.suffix or '.png'}"
    out_path = ARTIFACTS_DIR / out_name
    return image_tools.save_image(result, out_path)


app = create_app()

# Auth routes are registered after app creation to share helpers


@app.post("/auth/register")
async def register(payload: dict[str, str], request: Request) -> dict[str, str]:
    """Register a user. Open only when no users exist; otherwise requires auth."""

    users = _load_users()
    if users:
        if not _get_session(request.cookies.get("v2s_session")):
            raise HTTPException(status_code=401, detail="Registration locked; login required")
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if any(u["username"] == username for u in users):
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = _hash_password(password)
    users.append({"username": username, **hashed})
    _save_users(users)
    return {"status": "registered"}


@app.post("/auth/login")
async def login(payload: dict[str, str], response: Response) -> dict[str, str]:
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    users = _load_users()
    for user in users:
        if user["username"] == username and _verify_password(password, user["salt"], user["hash"]):
            token = _issue_session(username)
            response.set_cookie(
                "v2s_session",
                token,
                httponly=True,
                samesite="strict",
                secure=True,
                max_age=SESSION_TTL_SECONDS,
            )
            session = _get_session_record(token)
            response.set_cookie(
                "v2s_csrf",
                session["csrf"] if session else "",
                httponly=False,
                samesite="strict",
                secure=True,
                max_age=SESSION_TTL_SECONDS,
            )
            return {"status": "ok", "user": username, "csrf": session["csrf"] if session else ""}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/auth/logout")
async def logout(response: Response) -> dict[str, str]:
    response.delete_cookie("v2s_session")
    response.delete_cookie("v2s_csrf")
    return {"status": "logged_out"}


@app.get("/auth/status")
async def auth_status(request: Request) -> dict[str, Any]:
    record = _get_session_record(request.cookies.get("v2s_session"))
    return {"authenticated": bool(record), "user": record["user"] if record else None}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("video2spritesheet.web.server:app", host="0.0.0.0", port=8000, reload=True)
