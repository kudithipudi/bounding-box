"""Upload processing: classify, rasterize PDFs, and normalize images.

Every upload — image or PDF page — is turned into a single normalized RGB image
that is (a) what gets sent to the vision model and (b) what gets displayed on the
result page. Because both use the exact same pixels, the model's normalized
bounding box coordinates overlay perfectly on what the user sees.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from app.config import get_settings

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}


class UnsupportedFileError(Exception):
    pass


class PdfRenderError(Exception):
    pass


@dataclass
class NormalizedImage:
    image: Image.Image
    width: int
    height: int


def classify_upload(filename: str) -> str:
    """Return 'image', 'pdf', or raise UnsupportedFileError."""
    ext = _extension(filename)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    raise UnsupportedFileError(
        f"Unsupported file type {ext or '(none)'}. Upload a PDF or a common image format."
    )


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def normalize_image(raw: bytes, kind: str, page: int = 1) -> NormalizedImage:
    """Turn raw upload bytes into a normalized RGB image.

    PDFs are rasterized page-first with PyMuPDF; images are opened directly.
    The result is downscaled so its long edge is at most max_image_dimension.
    """
    settings = get_settings()
    if kind == "pdf":
        pil = _render_pdf_page(raw, page)
    else:
        try:
            pil = Image.open(io.BytesIO(raw))
        except (UnidentifiedImageError, OSError) as exc:
            raise UnsupportedFileError("Could not read that image file.") from exc
        pil.load()

    pil = pil.convert("RGB")
    max_dim = settings.max_image_dimension
    if max(pil.size) > max_dim:
        scale = max_dim / max(pil.size)
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.LANCZOS)
    return NormalizedImage(image=pil, width=pil.width, height=pil.height)


def _render_pdf_page(raw: bytes, page: int) -> Image.Image:
    try:
        import pymupdf  # PyMuPDF >= 1.24 exposes the modern `pymupdf` name.
    except ImportError:
        try:
            import fitz as pymupdf  # older alias
        except ImportError as exc:  # pragma: no cover
            raise PdfRenderError("PDF support is not installed.") from exc

    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise PdfRenderError("Could not open that PDF.") from exc

    if doc.page_count == 0:
        doc.close()
        raise PdfRenderError("That PDF has no pages.")

    idx = max(0, min(page - 1, doc.page_count - 1))
    settings = get_settings()
    try:
        page_obj = doc.load_page(idx)
        # matrix zooms so that dpi maps to ~dpi/72 scale of PDF points.
        scale = settings.pdf_render_dpi / 72.0
        pix = page_obj.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception as exc:
        raise PdfRenderError("Could not render that PDF page.") from exc
    finally:
        doc.close()
    return pil


def to_jpeg_bytes(image: Image.Image) -> bytes:
    """Encode a normalized image as JPEG bytes (used for display + LLM input)."""
    buf = io.BytesIO()
    settings = get_settings()
    image.save(buf, format="JPEG", quality=settings.jpeg_quality, optimize=True)
    return buf.getvalue()


def to_base64(image: Image.Image, mime: str = "image/jpeg") -> str:
    import base64

    buf = io.BytesIO()
    if mime == "image/png":
        image.save(buf, format="PNG")
    else:
        image.save(buf, format="JPEG", quality=get_settings().jpeg_quality, optimize=True)
    return "data:" + mime + ";base64," + base64.b64encode(buf.getvalue()).decode("ascii")