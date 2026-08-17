import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.db import delete_detection, get_db, get_detection, list_detections

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["prefix"] = get_settings().root_path


def _render(request: Request, name: str, context: dict, **kwargs):
    context.setdefault("is_admin", bool(request.session.get("is_admin")))
    return templates.TemplateResponse(request, name, context, **kwargs)


def _login_url(request: Request) -> str:
    return f"{get_settings().root_path}/admin/login"


def _require_admin(request: Request):
    """Raise/redirect helpers for protected routes. Called manually by each
    protected route so the page can 303 to the login form (human navigates it),
    while the delete POST just returns a redirect too."""
    if not request.session.get("is_admin"):
        return False
    return True


@router.get("/admin/login")
async def login_page(request: Request):
    if request.session.get("is_admin"):
        return RedirectResponse(f"{get_settings().root_path}/admin", status_code=303)
    return _render(request, "admin_login.html", {"error": None})


@router.post("/admin/login")
async def login_submit(request: Request, password: str = Form("")):
    settings = get_settings()
    expected = settings.admin_password
    if expected and secrets.compare_digest(password, expected):
        request.session["is_admin"] = True
        return RedirectResponse(f"{get_settings().root_path}/admin", status_code=303)
    return _render(
        request,
        "admin_login.html",
        {"error": "Wrong password."},
        status_code=401,
    )


@router.post("/admin/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(_login_url(request), status_code=303)


@router.get("/admin")
async def admin_page(request: Request, db=Depends(get_db)):
    if not _require_admin(request):
        return RedirectResponse(_login_url(request), status_code=303)
    items = await list_detections(db, limit=get_settings().max_history_items)
    return _render(request, "admin.html", {"items": items})


@router.post("/admin/delete/{detection_id}")
async def delete_item(request: Request, detection_id: int, db=Depends(get_db)):
    if not _require_admin(request):
        return RedirectResponse(_login_url(request), status_code=303)
    det = await get_detection(db, detection_id)
    if det:
        await delete_detection(db, detection_id)
        # Remove the stored image + thumbnail so uploads don't accumulate.
        for name in (det["media_file"], det["thumb_file"]):
            if not name:
                continue
            path = Path(get_settings().uploads_dir) / name
            if path.exists():
                path.unlink()
    return RedirectResponse(f"{get_settings().root_path}/admin", status_code=303)