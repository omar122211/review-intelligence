"""Settings, read from environment variables or a local .env file (never committed)."""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini API (free tier) — summaries, explanations with quotes, owner replies
    gemini_api_key: SecretStr | None = None
    chat_model: str = "gemini-3.5-flash-lite"
    fallback_chat_model: str | None = "gemini-3.5-flash"

    # Local models
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    demo_file: Path = ROOT / "demo" / "campgrounds.json"

    # API limits
    max_review_chars: int = 3000
    rate_limit_per_minute: int = 30

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
