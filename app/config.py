from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    root_path: str = "/bounding-box"
    db_path: str = "data/bounding-box.db"
    uploads_dir: str = "data/uploads"

    # orcarouter — an OpenAI-compatible endpoint. Point ORCAROUTER_BASE_URL at
    # the /v1 root of the router and set ORCAROUTER_MODEL to the exact model id
    # (e.g. "qwen/qwen3.8-27b-free"). Both come from .env.
    orcarouter_api_key: str = ""
    orcarouter_base_url: str = "https://api.orcarouter.ai/v1"
    orcarouter_model: str = "qwen/qwen3.8-27b-free"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 90
    llm_max_retries: int = 2

    # Vision input handling: images are normalized (capped at this max
    # dimension on the long edge, JPEG-encoded) before being sent to the model,
    # and that same normalized file is what gets displayed — so the bounding
    # box coordinates map 1:1 onto what the user sees.
    max_image_dimension: int = 1280
    jpeg_quality: int = 85
    pdf_render_dpi: int = 150

    # Uploads
    max_upload_mb: int = 25

    # History cap for the /history page.
    max_history_items: int = 100


def get_settings() -> Settings:
    # Not cached: this app runs a single gunicorn worker and Settings() is cheap
    # to build, so we always read the current environment/.env rather than risk
    # a stale cached instance (e.g. across tests that monkeypatch env vars).
    return Settings()