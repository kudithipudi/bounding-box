-- Bounding box detections. One row per uploaded file + user description.
-- Bounding box coordinates are normalized to [0, 1] relative to the displayed
-- image (width/height columns), so they overlay 1:1 regardless of render size.
CREATE TABLE IF NOT EXISTS detections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name  TEXT NOT NULL,
    kind           TEXT NOT NULL,               -- 'image' | 'pdf'
    page           INTEGER NOT NULL DEFAULT 1,  -- rendered page for PDFs
    media_file     TEXT NOT NULL,               -- filename under data/uploads
    content_type   TEXT NOT NULL,               -- served media content type
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    description    TEXT NOT NULL,
    x1             REAL,                        -- normalized 0..1
    y1             REAL,
    x2             REAL,
    y2             REAL,
    label          TEXT,
    confidence     REAL,
    model          TEXT,
    status         TEXT NOT NULL DEFAULT 'ok',  -- 'ok' | 'error'
    error          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections (created_at DESC);