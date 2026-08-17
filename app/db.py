import logging
from pathlib import Path

import aiosqlite

from app.config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


async def connect(db_path: str | None = None) -> aiosqlite.Connection:
    db_path = db_path or get_settings().db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Idempotent migrations for DBs created before a column was added. The base
# schema.sql only creates tables; ALTERs are listed here so a running deploy
# upgrades in place on the next restart.
_MIGRATIONS = [
    # bounding-box v2: thumbnails + rate limiting
    "ALTER TABLE detections ADD COLUMN thumb_file TEXT NOT NULL DEFAULT ''",
]


async def _migrate(conn: aiosqlite.Connection) -> None:
    for statement in _MIGRATIONS:
        try:
            await conn.execute(statement)
        except aiosqlite.OperationalError as exc:
            if "duplicate column name" in str(exc):
                continue  # already applied
            raise
    await conn.commit()


async def init_db(db_path: str | None = None) -> None:
    conn = await connect(db_path)
    try:
        schema = _SCHEMA_PATH.read_text()
        await conn.executescript(schema)
        await _migrate(conn)
        await conn.commit()
        logger.info("Database schema applied")
    finally:
        await conn.close()


async def get_db():
    conn = await connect()
    try:
        yield conn
    finally:
        await conn.close()


async def save_detection(
    conn: aiosqlite.Connection,
    *,
    original_name: str,
    kind: str,
    page: int,
    media_file: str,
    thumb_file: str,
    content_type: str,
    width: int,
    height: int,
    description: str,
    x1: float | None = None,
    y1: float | None = None,
    x2: float | None = None,
    y2: float | None = None,
    label: str = "",
    confidence: float | None = None,
    model: str = "",
    status: str = "ok",
    error: str = "",
) -> int:
    cur = await conn.execute(
        "INSERT INTO detections (original_name, kind, page, media_file, thumb_file,"
        " content_type, width, height, description, x1, y1, x2, y2, label, confidence,"
        " model, status, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            original_name, kind, page, media_file, thumb_file, content_type,
            width, height, description, x1, y1, x2, y2, label, confidence, model,
            status, error,
        ),
    )
    await conn.commit()
    return cur.lastrowid


async def get_detection(conn: aiosqlite.Connection, detection_id: int) -> dict | None:
    rows = await conn.execute_fetchall(
        "SELECT * FROM detections WHERE id = ?", (detection_id,)
    )
    return dict(rows[0]) if rows else None


async def delete_detection(conn: aiosqlite.Connection, detection_id: int) -> bool:
    """Delete a detection row. Returns True if a row was removed."""
    cur = await conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    await conn.commit()
    return cur.rowcount > 0


async def list_detections(conn: aiosqlite.Connection, limit: int = 50) -> list[dict]:
    rows = await conn.execute_fetchall(
        "SELECT id, original_name, kind, description, label, confidence, status,"
        " thumb_file, created_at FROM detections ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


async def check_and_record_rate_limit(
    conn: aiosqlite.Connection,
    *,
    ip: str,
    route: str,
    limit: int,
    window_seconds: int,
) -> bool:
    """Record a hit for (ip, route) and return whether it's within `limit`
    hits in the trailing `window_seconds`. Also prunes hits for this route
    older than the window, so the table doesn't grow unbounded."""
    offset = f"-{window_seconds} seconds"
    await conn.execute(
        "DELETE FROM rate_limit_hits WHERE route = ?"
        " AND created_at < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)",
        (route, offset),
    )
    cur = await conn.execute(
        "SELECT COUNT(*) FROM rate_limit_hits WHERE route = ? AND ip = ?"
        " AND created_at >= strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)",
        (route, ip, offset),
    )
    row = await cur.fetchone()
    if row[0] >= limit:
        await conn.commit()
        return False
    await conn.execute(
        "INSERT INTO rate_limit_hits (ip, route) VALUES (?, ?)", (ip, route)
    )
    await conn.commit()
    return True