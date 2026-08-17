import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.db import (check_and_record_rate_limit, count_detections, get_db,
                    get_detection, list_detections, prune_stale_pending,
                    save_detection)
from app.services import llm
from app.services.images import (classify_upload, normalize_image,
                                 pdf_page_count, to_jpeg_bytes, to_llm_base64,
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


async def _rate_limited(request: Request, db) -> bool:
    """True if this request should be rejected (and the hit recorded)."""
    settings = get_settings()
    return not await check_and_record_rate_limit(
        db,
        ip=_client_ip(request),
        route="detect",
        limit=settings.rate_limit_per_minute,
        window_seconds=settings.rate_limit_window_seconds,
    )


async def _prune_stale_pending(db) -> None:
    """Sweep abandoned 'pending' detections (uploaded but never confirmed)
    older than the configured TTL, removing their files too."""
    stale = await prune_stale_pending(db, get_settings().pending_ttl_hours)
    for row in stale:
        for name in (row["media_file"], row["thumb_file"]):
            if not name:
                continue
            path = _uploads_dir() / name
            if path.exists():
                path.unlink()


@router.get("/")
async def index(request: Request, db=Depends(get_db)):
    await _prune_stale_pending(db)
    recent = await list_detections(db, limit=8)
    return _render(request, "index.html", {"recent": recent, "error": None, "description": ""})


@router.post("/detect")
async def upload_step(
    request: Request,
    file: UploadFile = File(...),
    description: str = Form(""),
    page: int = Form(1),
    db=Depends(get_db),
):
    """Step 1: upload + interpret. Saves the pending detection and asks the LLM
    to restate what to look for, then sends the user to a confirmation page."""
    settings = get_settings()
    recent = await list_detections(db, limit=8)
    description = description.strip()

    if await _rate_limited(request, db):
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

    page = max(1, page)
    page_count = 1
    if kind == "pdf":
        try:
            page_count = pdf_page_count(raw)
        except PdfRenderError as exc:
            return _render(
                request, "index.html",
                {"recent": recent, "error": str(exc), "description": description},
                status_code=400,
            )
        if page > page_count:
            noun = "page" if page_count == 1 else "pages"
            return _render(
                request, "index.html",
                {"recent": recent,
                 "error": f"That PDF has {page_count} {noun} — choose a page from 1 to {page_count}.",
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

    _uploads_dir().mkdir(parents=True, exist_ok=True)
    stem = f"{int(time.time() * 1000)}-{uuid4().hex[:8]}"
    media_name = f"{stem}.jpg"
    thumb_name = f"{stem}.thumb.jpg"
    ( _uploads_dir() / media_name).write_bytes(to_jpeg_bytes(norm.image))
    ( _uploads_dir() / thumb_name).write_bytes(to_jpeg_bytes_thumb(norm.image))

    detection_id = await save_detection(
        db,
        original_name=file.filename or "upload",
        kind=kind,
        page=page if kind == "pdf" else 1,
        page_count=page_count if kind == "pdf" else 1,
        media_file=media_name,
        thumb_file=thumb_name,
        content_type="image/jpeg",
        width=norm.width,
        height=norm.height,
        description=description,
        status="pending",
    )

    # Interpret what the user wants to find (cheap text call). This is a
    # confirmation aid only — on failure we fall back to the raw description.
    try:
        target = llm.interpret(description)
    except llm.LlmError:
        target = description
    await _set_target(db, detection_id, target)

    return RedirectResponse(
        f"{settings.root_path}/confirm/{detection_id}", status_code=303
    )


@router.get("/confirm/{detection_id}")
async def confirm_view(request: Request, detection_id: int, db=Depends(get_db)):
    det = await get_detection(db, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found.")
    if det["status"] != "pending":
        return RedirectResponse(f"{get_settings().root_path}/r/{detection_id}", status_code=303)
    return _render(request, "confirm.html", {"det": det})


@router.post("/run/{detection_id}")
async def run_detection(
    request: Request,
    detection_id: int,
    target: str = Form(""),
    db=Depends(get_db),
):
    """Step 2: run vision detection with the (possibly edited) confirmed target."""
    settings = get_settings()
    det = await get_detection(db, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found.")
    if det["status"] != "pending":
        return RedirectResponse(f"{settings.root_path}/r/{detection_id}", status_code=303)

    if await _rate_limited(request, db):
        return _render(
            request,
            "confirm.html",
            {"det": det, "error": "Too many requests — please wait a minute and try again."},
            status_code=429,
        )

    target = target.strip() or det["target"] or det["description"]
    await _set_target(db, detection_id, target)

    path = _uploads_dir() / det["media_file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image missing.")

    from PIL import Image as PILImage

    norm = PILImage.open(path).convert("RGB")
    data_url = to_llm_base64(norm)

    try:
        result = llm.detect(data_url, target)
        primary = result.primary
        await save_result(
            db,
            detection_id,
            status="ok",
            x1=primary.x1 if primary else None,
            y1=primary.y1 if primary else None,
            x2=primary.x2 if primary else None,
            y2=primary.y2 if primary else None,
            label=primary.label if primary else "",
            confidence=primary.confidence if primary else None,
            boxes_json=json.dumps([
                {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                 "label": b.label, "confidence": b.confidence}
                for b in result.boxes
            ]),
            model=result.model,
        )
    except (llm.LlmError, llm.BadBoxError) as exc:
        await save_result(db, detection_id, status="error", error=str(exc))

    return RedirectResponse(f"{settings.root_path}/r/{detection_id}", status_code=303)


@router.get("/r/{detection_id}")
async def result_view(request: Request, detection_id: int, db=Depends(get_db)):
    det = await get_detection(db, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found.")
    if det["status"] == "pending":
        return RedirectResponse(f"{get_settings().root_path}/confirm/{detection_id}", status_code=303)

    boxes = []
    if det["status"] == "ok":
        try:
            raw_boxes = json.loads(det["boxes_json"] or "[]")
        except ValueError:
            raw_boxes = []
        if not raw_boxes and all(v is not None for v in (det["x1"], det["y1"], det["x2"], det["y2"])):
            raw_boxes = [{
                "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
                "label": det["label"], "confidence": det["confidence"],
            }]
        for b in raw_boxes:
            boxes.append({
                "x1": round(b["x1"] * 100, 3),
                "y1": round(b["y1"] * 100, 3),
                "width": round((b["x2"] - b["x1"]) * 100, 3),
                "height": round((b["y2"] - b["y1"]) * 100, 3),
                "label": b.get("label", ""),
                "confidence": b.get("confidence"),
            })
    return _render(request, "result.html", {"det": det, "boxes": boxes})


async def _set_target(db, detection_id: int, target: str) -> None:
    await db.execute("UPDATE detections SET target = ? WHERE id = ?", (target, detection_id))
    await db.commit()


async def save_result(db, detection_id: int, *, status: str, x1=None, y1=None, x2=None, y2=None,
                      label="", confidence=None, boxes_json="[]", model="", error=""):
    await db.execute(
        "UPDATE detections SET status = ?, x1 = ?, y1 = ?, x2 = ?, y2 = ?,"
        " label = ?, confidence = ?, boxes_json = ?, model = ?, error = ? WHERE id = ?",
        (status, x1, y1, x2, y2, label, confidence, boxes_json, model, error, detection_id),
    )
    await db.commit()


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
async def history(request: Request, page: int = 1, db=Depends(get_db)):
    page_size = get_settings().history_page_size
    total = await count_detections(db)
    total_pages = max(1, -(-total // page_size))  # ceil div
    page = min(max(1, page), total_pages)
    items = await list_detections(db, limit=page_size, offset=(page - 1) * page_size)
    return _render(request, "history.html", {
        "items": items, "page": page, "total_pages": total_pages, "total": total,
    })


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}