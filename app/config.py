from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, read from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"

    confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    max_upload_mb: float = Field(default=10.0, gt=0)

    # Comma-separated website origins allowed to call the API from a browser,
    # e.g. "https://mysite.com,http://localhost:5173". Empty = no cross-origin access.
    cors_origins: str = ""

    model_path: Path = Path("models/mobilenetv2_waste_classifier.keras")
    kb_path: Path = Path("knowledge/waste_knowledge_base.json")
    class_mapping_path: Path = Path("config/class_mapping.json")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()]

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings()
