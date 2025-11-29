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
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

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
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"
LOGIN_PATH = STATIC_DIR / "login.html"
USERS_PATH = BASE_DIR / "config" / "users.json"
SESSION_TTL_SECONDS = 7 * 24 * 3600
file_tools.ensure_directory(ARTIFACTS_DIR)
file_tools.ensure_directory(MEDIA_IMAGES_DIR)
file_tools.ensure_directory(MEDIA_VIDEOS_DIR)
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
    except Exception:
        return []


def _save_users(users: list[dict[str, str]]) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)


def _issue_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = {"user": username, "expires": time.time() + SESSION_TTL_SECONDS}
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
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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
    async def upload_media(file: UploadFile = File(...), user: str = Depends(require_auth)) -> dict[str, str]:
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
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        target = target_dir / f"upload_{timestamp}{suffix or '.dat'}"
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
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
            target.unlink()
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
                local_image = await _persist_upload(image, subdir="images")
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

        return {"image_url": _artifact_url(result_path)}

    @app.post("/api/generate", response_model=GenerationResponse)
    async def generate_spritesheet(
        background_tasks: BackgroundTasks,
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
                local_video = await _persist_upload(video)
            else:
                raise HTTPException(status_code=400, detail="Provide a video or media_path")
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


async def _persist_upload(file: UploadFile, subdir: str = "uploads") -> Path:
    """Persist the uploaded file into artifacts/uploads for processing."""

    suffix = Path(file.filename or "upload").suffix or ".dat"
    target_dir = ARTIFACTS_DIR / subdir
    file_tools.ensure_directory(target_dir)
    timestamp = int(time.time())
    target = target_dir / f"upload_{timestamp}{suffix}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
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


def _cleanup_file(path: Path) -> None:
    """Remove a temporary upload if it exists."""

    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.debug("Cleanup failed for %s", path)


def _run_generation(video_path: Path, request: GenerationRequest) -> GenerationResult:
    """Perform the full pipeline for a single upload."""

    # Validate video and discover metadata
    validators.validate_video_path(video_path)
    metadata = video_loader.load_metadata(video_path)

    timestamp = int(time.time())
    filename = file_tools.format_output_filename(
        video_path, request.output_pattern or "{stem}_sheet_{ts}", timestamp=timestamp
    )
    output_path = ARTIFACTS_DIR / filename

    background_color = request.background_color
    chroma_color = request.chroma_key_color or background_color or (0, 0, 0, 255)

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
        max_frames=request.max_frames,
        background_color=background_color,
        remove_black_background=request.remove_black_background,
        chroma_key_color=chroma_color,
        chroma_key_tolerance=request.chroma_key_tolerance,
        auto_edge_cutout=request.auto_edge_cutout,
        padding=request.padding,
        output_pattern=request.output_pattern,
    )

    frames, infos = frame_extractor.extract_frames(settings, metadata)
    sheet_path, sheet_image, packed_infos = spritesheet_builder.build_spritesheet(frames, infos, settings)
    columns, rows = spritesheet_builder._resolve_grid(len(frames), settings.columns, settings.rows)
    manifest_path = None
    if settings.generate_manifest:
        manifest_path = manifest_writer.write_manifest(packed_infos, settings, columns, rows)

    outcome = ProcessingOutcome(spritesheet_path=sheet_path, manifest_path=manifest_path)
    width, height = sheet_image.size
    frame_width, frame_height = frames[0].size
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
                samesite="lax",
                secure=False,
                max_age=SESSION_TTL_SECONDS,
            )
            return {"status": "ok", "user": username}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/auth/logout")
async def logout(response: Response) -> dict[str, str]:
    response.delete_cookie("v2s_session")
    return {"status": "logged_out"}


@app.get("/auth/status")
async def auth_status(request: Request) -> dict[str, Any]:
    user = _get_session(request.cookies.get("v2s_session"))
    return {"authenticated": bool(user), "user": user}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("video2spritesheet.web.server:app", host="0.0.0.0", port=8000, reload=True)
