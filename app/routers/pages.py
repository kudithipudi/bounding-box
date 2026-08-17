import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.db import (get_db, get_detection, list_detections, save_detection)
from app.services import llm
from app.services.images import (classify_upload, normalize_image,
                                 to_base64, to_jpeg_bytes,
                                 UnsupportedFileError, PdfRenderError)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["prefix"] = get_settings().root_path


def _uploads_dir() -> Path:
    return Path(get_settings().uploads_dir)


@router.get("/")
async def index(request: Request, db=Depends(get_db)):
    recent = await list_detections(db, limit=8)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"recent": recent, "error": None, "description": ""},
    )


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
    if not description:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"recent": recent, "error": "Please describe what you want detected.", "description": ""},
            status_code=400,
        )

    try:
        kind = classify_upload(file.filename or "")
    except UnsupportedFileError as exc:
        return templates.TemplateResponse(
            request, "index.html",
            {"recent": recent, "error": str(exc), "description": description},
            status_code=400,
        )

    raw = await file.read()
    if not raw:
        return templates.TemplateResponse(
            request, "index.html",
            {"recent": recent, "error": "That file is empty.", "description": description},
            status_code=400,
        )
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        return templates.TemplateResponse(
            request, "index.html",
            {"recent": recent,
             "error": f"File is over the {settings.max_upload_mb} MB limit.",
             "description": description},
            status_code=400,
        )

    try:
        norm = normalize_image(raw, kind, page=page)
    except (UnsupportedFileError, PdfRenderError) as exc:
        return templates.TemplateResponse(
            request, "index.html",
            {"recent": recent, "error": str(exc), "description": description},
            status_code=400,
        )

    # Persist the exact normalized image that will be sent to the model, so the
    # returned normalized box overlays pixel-perfect on what we display.
    _uploads_dir().mkdir(parents=True, exist_ok=True)
    media_name = f"{int(time.time() * 1000)}-{uuid4().hex[:8]}.jpg"
    ( _uploads_dir() / media_name).write_bytes(to_jpeg_bytes(norm.image))

    data_url = to_base64(norm.image, "image/jpeg")
    try:
        det = llm.detect(data_url, description)
        detection_id = await save_detection(
            db,
            original_name=file.filename or "upload",
            kind=kind,
            page=page if kind == "pdf" else 1,
            media_file=media_name,
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
    return templates.TemplateResponse(
        request, "result.html", {"det": det, "box": box}
    )


@router.get("/media/{detection_id}")
async def media(detection_id: int, db=Depends(get_db)):
    det = await get_detection(db, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Not found.")
    path = _uploads_dir() / det["media_file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image missing.")
    return FileResponse(path, media_type=det["content_type"])


@router.get("/history")
async def history(request: Request, db=Depends(get_db)):
    items = await list_detections(db, limit=get_settings().max_history_items)
    return templates.TemplateResponse(request, "history.html", {"items": items})


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}