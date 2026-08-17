import pytest

from app.services import llm
from app.services.images import classify_upload, normalize_image, to_base64, UnsupportedFileError


# --- upload classification -----------------------------------------------


def test_classify_accepts_images():
    assert classify_upload("photo.JPG") == "image"
    assert classify_upload("scan.png") == "image"
    assert classify_upload("a.webp") == "image"
    assert classify_upload("a.bmp") == "image"


def test_classify_accepts_pdf():
    assert classify_upload("report.pdf") == "pdf"


@pytest.mark.parametrize(
    "name", ["notes.txt", "archive.zip", "noextension", ""]
)
def test_classify_rejects_others(name):
    with pytest.raises(UnsupportedFileError):
        classify_upload(name)


# --- image normalization -------------------------------------------------


def test_normalize_image_keeps_dimensions(tmp_path, monkeypatch):
    from PIL import Image

    import io

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (200, 30, 30)).save(buf, format="PNG")
    norm = normalize_image(buf.getvalue(), "image")
    assert (norm.width, norm.height) == (640, 480)
    assert norm.image.mode == "RGB"


def test_normalize_image_downscales_long_edge(monkeypatch):
    from PIL import Image

    import io

    monkeypatch.setenv("MAX_IMAGE_DIMENSION", "300")
    buf = io.BytesIO()
    Image.new("RGB", (1200, 600), (0, 0, 200)).save(buf, format="JPEG")
    norm = normalize_image(buf.getvalue(), "image")
    assert norm.width == 300
    assert norm.height == 150


def test_normalize_rejects_garbage():
    with pytest.raises(UnsupportedFileError):
        normalize_image(b"this is definitely not an image", "image")


def test_to_base64_jpeg_data_url(monkeypatch):
    from PIL import Image

    norm = normalize_image(
        _png_bytes(), "image"
    )
    url = to_base64(norm.image, "image/jpeg")
    assert url.startswith("data:image/jpeg;base64,")


def _png_bytes() -> bytes:
    from PIL import Image

    import io

    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# --- LLM response parsing ------------------------------------------------


def test_extract_json_plain():
    assert llm._extract_json('{"x1": 0.1, "y1": 0.2}') == {"x1": 0.1, "y1": 0.2}


def test_extract_json_fenced():
    data = llm._extract_json('```json\n{"x1": 0.1}\n```')
    assert data == {"x1": 0.1}


def test_extract_json_prose():
    data = llm._extract_json('Sure! Here it is: {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.9} hope that helps')
    assert data["x1"] == 0.1


def test_extract_json_garbage_raises():
    with pytest.raises(llm.BadBoxError):
        llm._extract_json("no json here at all")


def test_parse_box_clamps_and_sorts():
    x1, y1, x2, y2 = llm._parse_box({"x1": -1.0, "y1": 2.0, "x2": 5.0, "y2": -2.0})
    assert (x1, y1, x2, y2) == (0.0, 1.0, 1.0, 0.0) or (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0)


def test_parse_box_degenerate_raises():
    with pytest.raises(llm.BadBoxError):
        llm._parse_box({"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5})


def test_parse_box_missing_raises():
    with pytest.raises(llm.BadBoxError):
        llm._parse_box({"label": "nothing"})


def test_detect_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(llm.LlmError, match="LLM_API_KEY"):
        llm.detect("data:image/jpeg;base64,AAAA", "the thing")