"""
Central configuration for Measured RAG.

Uses `pydantic-settings` so every tunable value can live in `.env`
with safe defaults and automatic type validation.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory (d:\EvalRag)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"

    # --- Milestone 1: Chunking parameters ---
    chunk_max_words: int = 350
    chunk_overlap_words: int = 50


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
