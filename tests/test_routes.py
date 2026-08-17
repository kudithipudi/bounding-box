from app.services import llm


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


# --- detect flow ---------------------------------------------------------


def _post_detect(client, filename="thing.png", content=None, description="the red circle", **extra):
    data = {"description": description, **extra}
    files = {"file": (filename, content, "image/png")}
    return client.post("/detect", data=data, files=files, follow_redirects=False)


def test_detect_saves_and_redirects(client, fake_llm, png_bytes):
    calls = fake_llm()
    resp = _post_detect(client, content=png_bytes)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/r/1"
    # The description must reach the (stubbed) LLM along with a base64 data URL.
    assert len(calls) == 1
    img_data, desc = calls[0]
    assert desc == "the red circle"
    assert img_data.startswith("data:image/jpeg;base64,")


def test_result_view_renders_box(client, fake_llm, png_bytes):
    fake_llm()
    _post_detect(client, content=png_bytes)
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
    _post_detect(client, content=png_bytes)
    resp = client.get("/media/1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_missing_description_rejected(client, fake_llm, png_bytes):
    fake_llm()
    resp = _post_detect(client, content=png_bytes, description="  ")
    assert resp.status_code == 400
    assert "describe" in resp.text


def test_unsupported_file_rejected(client, fake_llm, png_bytes):
    fake_llm()
    resp = _post_detect(client, filename="notes.txt", content=b"hello", description="the text")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.text


def test_bad_image_rejected(client, fake_llm):
    fake_llm()
    resp = _post_detect(client, content=b"not really a png")
    assert resp.status_code == 400


def test_llm_error_saved_as_failed(client, fake_llm, png_bytes):
    fake_llm(exc=llm.LlmError("upstream refused"))
    resp = _post_detect(client, content=png_bytes)
    assert resp.status_code == 303
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert "couldn't find" in resp.text
    assert "upstream refused" in resp.text


def test_bad_box_saved_as_failed(client, fake_llm, png_bytes):
    fake_llm(exc=llm.BadBoxError("Degenerate box"))
    resp = _post_detect(client, content=png_bytes)
    assert resp.status_code == 303
    resp = client.get("/r/1")
    assert "Degenerate box" in resp.text


def test_history_lists_saved_detections(client, fake_llm, png_bytes):
    fake_llm()
    _post_detect(client, content=png_bytes)
    _post_detect(client, content=png_bytes, description="the blue square")
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
    resp = _post_detect(client, filename="t.pdf", content=pdf_path.read_bytes(), description="the square")
    assert resp.status_code == 303
    resp = client.get("/r/1")
    assert resp.status_code == 200
    assert "pdf" in resp.text
    assert "p1" in resp.text


# --- thumbnails ----------------------------------------------------------


def test_thumbnail_generated_and_served(client, fake_llm, png_bytes):
    fake_llm()
    _post_detect(client, content=png_bytes)
    resp = client.get("/media/1?thumb=1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_history_shows_thumbnail(client, fake_llm, png_bytes):
    fake_llm()
    _post_detect(client, content=png_bytes)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "media/1?thumb=1" in resp.text


# --- rate limiting -------------------------------------------------------


def test_detect_rate_limited_per_ip(client, fake_llm, png_bytes, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    fake_llm()

    assert _post_detect(client, content=png_bytes).status_code == 303
    assert _post_detect(client, content=png_bytes).status_code == 303
    resp = _post_detect(client, content=png_bytes)
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
    _post_detect(client, content=png_bytes)
    resp = client.post("/admin/delete/1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/bounding-box/admin/login"


def test_admin_delete_removes_detection(client, fake_llm, png_bytes):
    fake_llm()
    _post_detect(client, content=png_bytes)
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
    _post_detect(client, content=png_bytes)
    files = glob.glob(os.path.join(uploads, "*"))
    assert len(files) == 2  # full image + thumbnail

    client.post("/admin/login", data={"password": "test-admin-password"})
    client.post("/admin/delete/1", follow_redirects=False)
    assert glob.glob(os.path.join(uploads, "*")) == []


def test_admin_page_lists_detections_with_thumbnail(client, fake_llm, png_bytes):
    fake_llm()
    _post_detect(client, content=png_bytes)
    client.post("/admin/login", data={"password": "test-admin-password"})
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "the red circle" in resp.text
    assert "media/1?thumb=1" in resp.text