import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.db import (check_and_record_rate_limit, get_db, get_detection,
                    list_detections, save_detection)
from app.services import llm
from app.services.images import (classify_upload, normalize_image,
                                 to_jpeg_bytes, to_llm_base64,
                                 to_jpeg_bytes_thumb,
                                 UnsupportedFileError, PdfRenderError)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["prefix"] = get_settings().root_path


def _uploads_dir() -> Path:
    return Path(get_settings().uploads_dir)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _render(request: Request, name: str, context: dict, **kwargs):
    """Render a template with the admin flag available to the chrome."""
    context.setdefault("is_admin", bool(request.session.get("is_admin")))
    return templates.TemplateResponse(request, name, context, **kwargs)


@router.get("/")
async def index(request: Request, db=Depends(get_db)):
    recent = await list_detections(db, limit=8)
    return _render(request, "index.html", {"recent": recent, "error": None, "description": ""})


@router.post("/detect")
async def create_detection(
    request: Request,
    file: UploadFile = File(...),
    description: str = Form(""),
    page: int = Form(1),
    db=Depends(get_db),
):
    """Upload a PDF/image, run vision detection, and jump to the result view."""
    settings = get_settings()
    recent = await list_detections(db, limit=8)
    description = description.strip()

    allowed = await check_and_record_rate_limit(
        db,
        ip=_client_ip(request),
        route="detect",
        limit=settings.rate_limit_per_minute,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed:
        return _render(
            request,
            "index.html",
            {"recent": recent, "error": "Too many requests — please wait a minute and try again.", "description": description},
            status_code=429,
        )

    if not description:
        return _render(
            request,
            "index.html",
            {"recent": recent, "error": "Please describe what you want detected.", "description": ""},
            status_code=400,
        )

    try:
        kind = classify_upload(file.filename or "")
    except UnsupportedFileError as exc:
        return _render(
            request, "index.html",
            {"recent": recent, "error": str(exc), "description": description},
            status_code=400,
        )

    raw = await file.read()
    if not raw:
        return _render(
            request, "index.html",
            {"recent": recent, "error": "That file is empty.", "description": description},
            status_code=400,
        )
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        return _render(
            request, "index.html",
            {"recent": recent,
             "error": f"File is over the {settings.max_upload_mb} MB limit.",
             "description": description},
            status_code=400,
        )

    try:
        norm = normalize_image(raw, kind, page=page)
    except (UnsupportedFileError, PdfRenderError) as exc:
        return _render(
            request, "index.html",
            {"recent": recent, "error": str(exc), "description": description},
            status_code=400,
        )

    # Persist the higher-res normalized image for display, plus a small preview
    # for history listings. The model gets a further-downscaled copy; because
    # boxes are normalized [0,1] the overlay maps 1:1 regardless.
    _uploads_dir().mkdir(parents=True, exist_ok=True)
    stem = f"{int(time.time() * 1000)}-{uuid4().hex[:8]}"
    media_name = f"{stem}.jpg"
    thumb_name = f"{stem}.thumb.jpg"
    ( _uploads_dir() / media_name).write_bytes(to_jpeg_bytes(norm.image))
    ( _uploads_dir() / thumb_name).write_bytes(to_jpeg_bytes_thumb(norm.image))

    data_url = to_llm_base64(norm.image)
    try:
        det = llm.detect(data_url, description)
        detection_id = await save_detection(
            db,
            original_name=file.filename or "upload",
            kind=kind,
            page=page if kind == "pdf" else 1,
            media_file=media_name,
            thumb_file=thumb_name,
            content_type="image/jpeg",
            width=norm.width,
            height=norm.height,
            description=description,
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
            label=det.label, confidence=det.confidence, model=det.model,
        )
    except (llm.LlmError, llm.BadBoxError) as exc:
        detection_id = await save_detection(
            db,
            original_name=file.filename or "upload",
            kind=kind,
            page=page if kind == "pdf" else 1,
            media_file=media_name,
            thumb_file=thumb_name,
            content_type="image/jpeg",
            width=norm.width,
            height=norm.height,
            description=description,
            status="error",
            error=str(exc),
        )
    return RedirectResponse(
        f"{settings.root_path}/r/{detection_id}", status_code=303
    )


@router.get("/r/{detection_id}")
async def result_view(request: Request, detection_id: int, db=Depends(get_db)):
    det = await get_detection(db, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found.")
    box = None
    if det["status"] == "ok" and all(v is not None for v in (det["x1"], det["y1"], det["x2"], det["y2"])):
        box = {
            "x1": round(det["x1"] * 100, 3),
            "y1": round(det["y1"] * 100, 3),
            "width": round((det["x2"] - det["x1"]) * 100, 3),
            "height": round((det["y2"] - det["y1"]) * 100, 3),
        }
    return _render(request, "result.html", {"det": det, "box": box})


@router.get("/media/{detection_id}")
async def media(request: Request, detection_id: int, db=Depends(get_db)):
    det = await get_detection(db, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Not found.")
    thumb = request.query_params.get("thumb") == "1"
    name = det["thumb_file"] if (thumb and det["thumb_file"]) else det["media_file"]
    path = _uploads_dir() / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image missing.")
    return FileResponse(path, media_type=det["content_type"])


@router.get("/history")
async def history(request: Request, db=Depends(get_db)):
    items = await list_detections(db, limit=get_settings().max_history_items)
    return _render(request, "history.html", {"items": items})


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}