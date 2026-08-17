# bounding-box

Point a vision LLM at a bounding box. Upload a PDF or image, describe an object
you want located, and the model returns the object's bounding box, drawn as an
overlay on your upload.

## What it is

- Accepts a PDF or a common image format (`.png`, `.jpg`, `.jpeg`, `.webp`,
  `.bmp`, `.gif`, `.tiff`).
- PDFs are rasterized page-first with PyMuPDF; images are normalized (RGB, long
  edge capped at `MAX_IMAGE_DIMENSION`, default 1280px).
- The normalized image is sent to a vision LLM over an **OpenAI-compatible**
  endpoint (orcarouter) as a base64 `image_url` content part, along with your
  description.
- The model returns a normalized `[0,1]` bounding box, which is overlaid on the
  exact same normalized image that was sent to the model — so the box maps
  1:1 to what you see, at any render size.
- Every run is recorded in SQLite (`data/bounding-box.db`) and listed on the
  History page.

## Stack

Python 3.12 · FastAPI · gunicorn (uvicorn worker) · Jinja2 · Tailwind CSS ·
SQLite (aiosqlite) · PyMuPDF · Pillow · httpx

## Run locally

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
# copy the template, then set ORCAROUTER_API_KEY (+ verify the model id)
cp .env.example .env
./venv/bin/uvicorn app.main:app --reload
```

Open http://localhost:8000/ .

Tests (from the venv):

```bash
./venv/bin/python -m pytest
```

## Deploy

This app follows the lab deployment pattern: systemd unit + gunicorn on a unix
socket behind nginx at `https://lab.kudithipudi.org/bounding-box/`.

```bash
sudo cp bounding-box.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bounding-box
sudo systemctl restart bounding-box
```

nginx already needs a `location /bounding-box/` block that strips the prefix
(`rewrite ^/bounding-box(/.*)$ $1 break;`) and proxies to
`unix:/var/www/bounding-box/bounding-box.sock`.

## Env vars

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROOT_PATH` | `/bounding-box` | Public subpath the app is mounted at (template prefix only). |
| `ORCAROUTER_API_KEY` | *(empty)* | API key for the orcarouter endpoint. Required for detection. |
| `ORCAROUTER_BASE_URL` | `https://api.orcarouter.ai/v1` | OpenAI-compatible base URL (the `/chat/completions` suffix is appended). |
| `ORCAROUTER_MODEL` | `qwen/qwen3.8-27b-free` | Exact model id to call. |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature. |
| `LLM_TIMEOUT_SECONDS` | `90` | Per-attempt HTTP timeout. |
| `LLM_MAX_RETRIES` | `2` | Retries on transient errors. |
| `MAX_IMAGE_DIMENSION` | `1280` | Long edge cap for the image sent to the model. |
| `JPEG_QUALITY` | `85` | JPEG quality of the stored/sent image. |
| `PDF_RENDER_DPI` | `150` | Rasterization DPI for PDF pages. |
| `MAX_UPLOAD_MB` | `25` | Upload size cap. |
| `DB_PATH` | `data/bounding-box.db` | SQLite file. |
| `UPLOADS_DIR` | `data/uploads` | Where normalized images live. |

## Rebuilding Tailwind CSS

`app/static/css/app.css` is built from `app/static/css/input.css` with the lab
standalone CLI (vendored at `./tailwindcss`) and committed:

```bash
./tailwindcss -i ./app/static/css/input.css -o ./app/static/css/app.css --minify
```

## Notes / limitations

- The default model is `qwen/qwen3.8-27b-free` on orcarouter — a free
  multimodal (vision) model. Override `ORCAROUTER_MODEL` if you want a different
  one.
- Only the rendered page of a multi-page PDF is sent to the model (page 1 by
  default — say which page you mean in your description). The model sees one
  image, not the whole document.
- If the model returns a degenerate box (zero width/height) or invalid JSON,
  the run is recorded as `error` and shown on the result page.