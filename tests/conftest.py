import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import Detection, DetectionBox


TEST_ADMIN_PASSWORD = "test-admin-password"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a throwaway SQLite db, temp uploads dir, and no
    LLM API key, so tests never touch the network."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app-test.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ROOT_PATH", "/bounding-box")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("ADMIN_PASSWORD", TEST_ADMIN_PASSWORD)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def anon_client(client):
    """A client with no admin session — for exercising what a visitor who
    hasn't logged in can and can't reach."""
    return client


@pytest.fixture
def admin_client(client):
    """A client already logged in to /admin (session cookie carries over to
    every subsequent request, same as a real browser)."""
    resp = client.post(
        "/admin/login", data={"password": TEST_ADMIN_PASSWORD}, follow_redirects=False
    )
    assert resp.status_code == 303
    return client


@pytest.fixture
def fake_llm(monkeypatch):
    """Stub llm.interpret and llm.detect so the flow never hits the network.

    Returns (calls, result) and lets tests override the result/error.
    interpret returns `target`; detect returns a Detection (defaults to a single
    box). Coordinates are normalized, same as the real service returns."""
    calls: list[tuple[str, str]] = []

    def _install(
        result: Detection | None = None,
        exc: Exception | None = None,
        target: str = "the object",
        interpret_exc: Exception | None = None,
    ):
        _result = result or Detection(
            boxes=[DetectionBox(x1=0.1, y1=0.2, x2=0.6, y2=0.8, label="the object", confidence=0.9)],
            model="test-model",
            raw="{}",
        )

        def _fake_interpret(description: str) -> str:
            if interpret_exc is not None:
                raise interpret_exc
            return target

        def _fake_detect(image_data_url: str, description: str) -> Detection:
            calls.append((image_data_url, description))
            if exc is not None:
                raise exc
            return _result

        monkeypatch.setattr("app.routers.pages.llm.interpret", _fake_interpret)
        monkeypatch.setattr("app.routers.pages.llm.detect", _fake_detect)
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