from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    root_path: str = "/bounding-box"
    db_path: str = "data/bounding-box.db"
    uploads_dir: str = "data/uploads"

    # Vision LLM via any OpenAI-compatible endpoint (OpenRouter, orcarouter, a
    # local vLLM/Ollama server, ...). Point LLM_BASE_URL at the /v1 root of the
    # provider and set LLM_MODEL to the exact model id.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.orcarouter.ai/v1"
    llm_model: str = "qwen/qwen3.8-27b-free"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 90
    llm_max_retries: int = 2

    # Vision input handling. The image shown to the user is stored at
    # max_image_dimension; the copy sent to the model is downscaled further to
    # llm_max_image_dimension. Bounding boxes are normalized [0,1] relative to
    # the model's image, so they overlay the higher-res display image 1:1 while
    # keeping vision-token usage (which scales with resolution) low.
    max_image_dimension: int = 1600
    llm_max_image_dimension: int = 1024
    jpeg_quality: int = 85
    llm_jpeg_quality: int = 80
    pdf_render_dpi: int = 150

    # Uploads
    max_upload_mb: int = 25

    # Per-IP cap on POST /detect (the route that calls the LLM), enforced over
    # a trailing window — same pattern as pretty-print.
    rate_limit_per_minute: int = 10
    rate_limit_window_seconds: int = 60

    # Admin section. ADMIN_PASSWORD gates /admin; SESSION_SECRET signs the
    # login cookie (falls back to ADMIN_PASSWORD if unset so a fresh checkout
    # runs, but set a stable SESSION_SECRET in .env).
    admin_password: str = ""
    session_secret: str = ""

    # History cap for the /history page.
    max_history_items: int = 100


def get_settings() -> Settings:
    # Not cached: this app runs a single gunicorn worker and Settings() is cheap
    # to build, so we always read the current environment/.env rather than risk
    # a stale cached instance (e.g. across tests that monkeypatch env vars).
    return Settings()