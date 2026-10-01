from pathlib import Path

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

    confidence_threshold: float = 0.90

    model_path: Path = Path("models/mobilenetv2_waste_classifier.keras")
    kb_path: Path = Path("knowledge/waste_knowledge_base.json")
    class_mapping_path: Path = Path("config/class_mapping.json")

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings()
