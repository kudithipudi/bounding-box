from app.services import llm
from app.services.llm import Detection, DetectionBox


# --- public pages (smoke tests) ------------------------------------------


def test_index_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "bounding-box" in resp.text
    assert 'name="description"' in resp.text
    assert 'name="file"' in resp.text


def test_history_ok_empty(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "History" in resp.text


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unknown_detection_404(client):
    resp = client.get("/r/99999")
    assert resp.status_code == 404


# --- two-step detect flow -------------------------------------------------


def _upload(client, filename="thing.png", content=None, description="the red circle", **extra):
    data = {"description": description, **extra}
    files = {"file": (filename, content, "image/png")}
    return client.post("/detect", data=data, files=files, follow_redirects=False)


def _confirm(client, detection_id=1, target=None):
    data = {"target": target} if target is not None else {}
    return client.post(f"/run/{detection_id}", data=data, follow_redirects=False)


def test_upload_saves_pending_and_redirects_to_confirm(client, fake_llm, png_bytes):
    calls = fake_llm()
    resp = _upload(client, content=png_bytes)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/confirm/1"
    # The interpreter got the raw description; the vision call hasn't run yet.
    assert len(calls) == 0
    # Confirm page shows the interpreted target.
    resp = client.get("/confirm/1")
    assert resp.status_code == 200
    assert "the object" in resp.text
    assert "the red circle" in resp.text


def test_full_flow_redirects_to_result(client, fake_llm, png_bytes):
    calls = fake_llm()
    _upload(client, content=png_bytes)
    resp = _confirm(client)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/r/1"
    # The vision call received the confirmed target.
    assert len(calls) == 1
    img_data, target = calls[0]
    assert target == "the object"
    assert img_data.startswith("data:image/jpeg;base64,")


def test_confirm_edit_target(client, fake_llm, png_bytes):
    calls = fake_llm()
    _upload(client, content=png_bytes)
    _confirm(client, target="all circles")
    assert calls[0][1] == "all circles"


def test_result_view_renders_box(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    _confirm(client)
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert "the red circle" in resp.text
    assert "the object" in resp.text
    # Overlay uses percentage box coordinates derived from the normalized box.
    assert "left: 10.0%" in resp.text
    assert "top: 20.0%" in resp.text
    assert "width: 50.0%" in resp.text
    assert "height: 60.0%" in resp.text


def test_result_serves_media(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    _confirm(client)
    resp = client.get("/media/1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_missing_description_rejected(client, fake_llm, png_bytes):
    fake_llm()
    resp = _upload(client, content=png_bytes, description="  ")
    assert resp.status_code == 400
    assert "describe" in resp.text


def test_unsupported_file_rejected(client, fake_llm, png_bytes):
    fake_llm()
    resp = _upload(client, filename="notes.txt", content=b"hello", description="the text")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.text


def test_bad_image_rejected(client, fake_llm):
    fake_llm()
    resp = _upload(client, content=b"not really a png")
    assert resp.status_code == 400


def test_llm_error_saved_as_failed(client, fake_llm, png_bytes):
    fake_llm(exc=llm.LlmError("upstream refused"))
    _upload(client, content=png_bytes)
    _confirm(client)
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert "couldn't run" in resp.text
    assert "upstream refused" in resp.text


def test_bad_box_saved_as_failed(client, fake_llm, png_bytes):
    fake_llm(exc=llm.BadBoxError("Degenerate box"))
    _upload(client, content=png_bytes)
    _confirm(client)
    resp = client.get("/r/1")
    assert "Degenerate box" in resp.text


def test_interpret_failure_falls_back_to_description(client, fake_llm, png_bytes):
    fake_llm(interpret_exc=llm.LlmError("boom"))
    _upload(client, content=png_bytes)
    resp = client.get("/confirm/1")
    assert resp.status_code == 200
    assert 'value="the red circle"' in resp.text


def test_pending_redirects_to_confirm(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    resp = client.get("/r/1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/confirm/1"


def test_history_lists_saved_detections(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    _confirm(client)
    _upload(client, content=png_bytes, description="the blue square")
    _confirm(client)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.text.count("the red circle") >= 1
    assert resp.text.count("the blue square") >= 1


def test_pdf_render_and_detect(client, fake_llm, tmp_path):
    """PDFs are rasterized to a page image before hitting the LLM; the result
    view should show the pdf source."""
    import pymupdf

    pdf_path = tmp_path / "t.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(40, 40, 160, 160), color=(0, 0, 1), fill=(0, 0, 1))
    doc.save(str(pdf_path))
    doc.close()

    fake_llm()
    _upload(client, filename="t.pdf", content=pdf_path.read_bytes(), description="the square")
    _confirm(client)
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert "pdf" in resp.text
    assert "p1" in resp.text


# --- multiple boxes ------------------------------------------------------


def test_multiple_boxes_rendered(client, fake_llm, png_bytes):
    result = Detection(
        boxes=[
            DetectionBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3, label="circle", confidence=0.9),
            DetectionBox(x1=0.5, y1=0.5, x2=0.7, y2=0.7, label="circle", confidence=0.8),
        ],
        model="test-model",
        raw="{}",
    )
    fake_llm(result=result)
    _upload(client, content=png_bytes)
    _confirm(client)
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert resp.text.count("left: 10.0%") >= 1
    assert resp.text.count("left: 50.0%") >= 1
    assert resp.text.count("circle") >= 2


def test_no_boxes_found(client, fake_llm, png_bytes):
    fake_llm(result=Detection(boxes=[], model="test-model", raw="{}"))
    _upload(client, content=png_bytes)
    _confirm(client)
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert "found no objects" in resp.text


# --- thumbnails ----------------------------------------------------------


def test_thumbnail_generated_and_served(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    resp = client.get("/media/1?thumb=1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_history_shows_thumbnail(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "media/1?thumb=1" in resp.text


# --- rate limiting -------------------------------------------------------


def test_detect_rate_limited_per_ip(client, fake_llm, png_bytes, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    fake_llm()

    assert _upload(client, content=png_bytes).status_code == 303
    assert _upload(client, content=png_bytes).status_code == 303
    resp = _upload(client, content=png_bytes)
    assert resp.status_code == 429
    assert "Too many requests" in resp.text


# --- admin ---------------------------------------------------------------


def test_admin_login_page(client):
    resp = client.get("/admin/login")
    assert resp.status_code == 200
    assert "Admin login" in resp.text


def test_admin_redirects_when_logged_out(client):
    resp = client.get("/admin", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/admin/login"


def test_admin_wrong_password(client):
    resp = client.post("/admin/login", data={"password": "nope"}, follow_redirects=False)
    assert resp.status_code == 401
    assert "Wrong password" in resp.text


def test_admin_login_and_logout(client):
    resp = client.post(
        "/admin/login", data={"password": "test-admin-password"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert client.get("/admin", follow_redirects=False).status_code == 200
    resp = client.post("/admin/logout", follow_redirects=False)
    assert resp.status_code == 303
    assert client.get("/admin", follow_redirects=False).status_code == 303


def test_admin_delete_requires_login(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    resp = client.post("/admin/delete/1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/admin/login"


def test_admin_delete_removes_detection(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    client.post("/admin/login", data={"password": "test-admin-password"})
    resp = client.post("/admin/delete/1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/admin"
    # Detection no longer viewable.
    assert client.get("/r/1").status_code == 404
    assert client.get("/media/1").status_code == 404


def test_admin_delete_removes_media_files(client, fake_llm, png_bytes, tmp_path, monkeypatch):
    """Deleting a detection also unlinks its stored image and thumbnail."""
    import glob
    import os

    uploads = str(tmp_path / "uploads")
    monkeypatch.setenv("UPLOADS_DIR", uploads)
    fake_llm()
    _upload(client, content=png_bytes)
    files = glob.glob(os.path.join(uploads, "*"))
    assert len(files) == 2  # full image + thumbnail

    client.post("/admin/login", data={"password": "test-admin-password"})
    client.post("/admin/delete/1", follow_redirects=False)
    assert glob.glob(os.path.join(uploads, "*")) == []


def test_admin_page_lists_detections_with_thumbnail(client, fake_llm, png_bytes):
    fake_llm()
    _upload(client, content=png_bytes)
    client.post("/admin/login", data={"password": "test-admin-password"})
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "the red circle" in resp.text
    assert "media/1?thumb=1" in resp.text