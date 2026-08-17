import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import get_settings
from app.db import init_db
from app.routers import admin, pages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Startup complete")
    yield


settings = get_settings()
# Ensure the SQLite + uploads dirs exist before the lifespan runs.
Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)

# NOTE: don't pass root_path to FastAPI. nginx strips the /bounding-box prefix
# before forwarding, so the app sees bare paths like /static/css/app.css.
# Templates still get the public prefix via the ROOT_PATH env -> {{ prefix }}
# global (see §5 / §1 of the lab standards).
app = FastAPI(lifespan=lifespan)
_here = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_here / "static")), name="static")

# Signs the admin login cookie. Falls back to admin_password so a fresh
# checkout still runs, but a real deploy should set SESSION_SECRET in .env —
# without a stable secret, every gunicorn restart logs everyone out.
# Deliberately left at the default cookie path "/" rather than root_path:
# nginx strips /bounding-box before proxying, so the app only ever sees bare
# paths like /admin — a cookie scoped to the subpath would never match those.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret or settings.admin_password or "dev-only-insecure-secret",
    session_cookie="bounding_box_session",
    max_age=12 * 60 * 60,
)

templates = Jinja2Templates(directory="app/templates")
templates.env.globals["prefix"] = settings.root_path

app.include_router(pages.router)
app.include_router(admin.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404 and request.url.path.startswith("/static"):
        return JSONResponse(status_code=404, content={"detail": exc.detail})
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})