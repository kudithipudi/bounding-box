import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import Detection


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a throwaway SQLite db, temp uploads dir, and no
    orcarouter key, so tests never touch the network."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app-test.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ROOT_PATH", "/bounding-box")
    monkeypatch.setenv("ORCAROUTER_API_KEY", "")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    """Stub llm.detect so POST /detect never performs a real LLM call.

    Returns (calls, result) and lets tests override the result/error. Coordinates
    are normalized, same as the real service returns."""
    calls: list[tuple[str, str]] = []

    def _install(result: Detection | None = None, exc: Exception | None = None):
        _result = result or Detection(
            x1=0.1, y1=0.2, x2=0.6, y2=0.8, label="the object",
            confidence=0.9, model="test-model", raw="{}",
        )

        def _fake(image_data_url: str, description: str) -> Detection:
            calls.append((image_data_url, description))
            if exc is not None:
                raise exc
            return _result

        monkeypatch.setattr("app.routers.pages.llm.detect", _fake)
        return calls

    return _install


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid 4x3 PNG (red square on white)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()