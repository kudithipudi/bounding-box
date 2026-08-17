# bounding-box

Point a vision LLM at a bounding box. Upload a PDF or image, describe an object
you want located, and the model returns the object's bounding box, drawn as an
overlay on your upload.

## What it is

- Accepts a PDF or a common image format (`.png`, `.jpg`, `.jpeg`, `.webp`,
  `.bmp`, `.gif`, `.tiff`).
- PDFs are rasterized page-first with PyMuPDF; images are normalized (RGB, long
  edge capped at `MAX_IMAGE_DIMENSION`, default 1600px).
- The normalized image is sent to a vision LLM over an **OpenAI-compatible**
  endpoint as a base64 `image_url` content part, along with your
  description. Any OpenAI-compatible provider works — OpenRouter, orcarouter,
  a local vLLM/Ollama server, etc.
- **Token-conscious**: the model actually receives a further-downscaled copy
  (long edge capped at `LLM_MAX_IMAGE_DIMENSION`, default 1024px). Vision
  tokens scale with resolution, and since boxes are normalized `[0,1]`, the
  returned box overlays the higher-res display image 1:1.
- The model returns a normalized `[0,1]` bounding box, drawn as an overlay on
  the uploaded image.
- Every run is recorded in SQLite (`data/bounding-box.db`) and listed on the
  History page with preview thumbnails.
- POST /detect is rate-limited per IP (`RATE_LIMIT_PER_MINUTE` over a 60s
  window), same pattern as pretty-print.
- A password-gated `/admin` area (session-cookie login) lets the operator
  delete entries from history.

## Stack

Python 3.12 · FastAPI · gunicorn (uvicorn worker) · Jinja2 · Tailwind CSS ·
SQLite (aiosqlite) · PyMuPDF · Pillow · httpx

## Run locally

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
# copy the template, then set LLM_API_KEY (+ verify the model id)
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
| `LLM_API_KEY` | *(empty)* | API key for your OpenAI-compatible LLM endpoint. Required for detection. |
| `LLM_BASE_URL` | `https://api.orcarouter.ai/v1` | OpenAI-compatible base URL (the `/chat/completions` suffix is appended). Works with any provider. |
| `LLM_MODEL` | `qwen/qwen3.8-27b-free` | Exact model id to call. |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature. |
| `LLM_TIMEOUT_SECONDS` | `90` | Per-attempt HTTP timeout. |
| `LLM_MAX_RETRIES` | `2` | Retries on transient errors. |
| `MAX_IMAGE_DIMENSION` | `1600` | Long edge cap for the display/stored image. |
| `LLM_MAX_IMAGE_DIMENSION` | `1024` | Long edge cap for the copy sent to the model (keeps vision tokens low). |
| `JPEG_QUALITY` | `85` | JPEG quality of the stored image. |
| `LLM_JPEG_QUALITY` | `80` | JPEG quality of the model-bound copy. |
| `PDF_RENDER_DPI` | `150` | Rasterization DPI for PDF pages. |
| `MAX_UPLOAD_MB` | `25` | Upload size cap (nginx `client_max_body_size` must allow it). |
| `RATE_LIMIT_PER_MINUTE` | `10` | Max POST /detect calls per IP per window. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window length. |
| `ADMIN_PASSWORD` | *(empty)* | Password for the /admin area. |
| `SESSION_SECRET` | *(empty)* | Signs the admin session cookie (stable value for prod). |
| `DB_PATH` | `data/bounding-box.db` | SQLite file. |
| `UPLOADS_DIR` | `data/uploads` | Where normalized images + thumbnails live. |

## Rebuilding Tailwind CSS

`app/static/css/app.css` is built from `app/static/css/input.css` with the lab
standalone CLI (vendored at `./tailwindcss`) and committed:

```bash
./tailwindcss -i ./app/static/css/input.css -o ./app/static/css/app.css --minify
```

## Notes / limitations

- The default model is `qwen/qwen3.8-27b-free` (free multimodal) — swap
  `LLM_MODEL`/`LLM_BASE_URL`/`LLM_API_KEY` for any OpenAI-compatible provider.
- Natural-language requests are interpreted by the LLM before the vision call:
  a bare noun ("banana") and plural/quantified phrases ("all circles",
  "circles") both mean *every* matching instance, one box per instance; a
  count or ordinal ("first three signatures", "the top two rows") means
  *exactly that many* boxes, in top-to-bottom order, stopping at the count.
  The interpreted target is shown on the confirm page and can be edited.
- Only the rendered page of a multi-page PDF is sent to the model (page 1 by
  default — say which page you mean in your description). The model sees one
  image, not the whole document.
- If the model returns a degenerate box (zero width/height) or invalid JSON,
  the run is recorded as `error` and shown on the result page.
- Deployments behind nginx need `client_max_body_size` large enough for uploads
  (the bounding-box location uses 30M; the app caps at `MAX_UPLOAD_MB`).