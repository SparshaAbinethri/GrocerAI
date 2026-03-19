"""
core/config.py
Centralised configuration — all env vars loaded once here.
Import `settings` anywhere in the codebase.

Keys are read with os.getenv() so imports never crash.
Missing required keys surface as clear errors only when the
relevant agent/client actually tries to use them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── OpenAI ───────────────────────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    vision_max_tokens: int = field(default_factory=lambda: int(os.getenv("VISION_MAX_TOKENS", "1500")))

    # ── Kroger API ───────────────────────────────────────────────────────────────
    kroger_client_id: str = field(default_factory=lambda: os.getenv("KROGER_CLIENT_ID", ""))
    kroger_client_secret: str = field(default_factory=lambda: os.getenv("KROGER_CLIENT_SECRET", ""))
    kroger_base_url: str = field(
        default_factory=lambda: os.getenv("KROGER_BASE_URL", "https://api.kroger.com/v1")
    )
    kroger_location_id: str = field(
        default_factory=lambda: os.getenv("KROGER_LOCATION_ID", "")
    )
    kroger_redirect_uri: str = field(
        default_factory=lambda: os.getenv("KROGER_REDIRECT_URI", "http://localhost:8501")
    )

    # ── RAG / FAISS ──────────────────────────────────────────────────────────────
    faiss_index_path: Path = field(
        default_factory=lambda: Path(os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))

    # ── App ──────────────────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_search_results_per_item: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    )
    session_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TTL", "3600"))
    )

    # ── Production / observability ───────────────────────────────────────────
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    sentry_dsn: str = field(default_factory=lambda: os.getenv("SENTRY_DSN", ""))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    health_port: int = field(default_factory=lambda: int(os.getenv("HEALTH_PORT", "8502")))
    rate_limit_pipeline: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PIPELINE", "10")))
    rate_limit_api: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_API", "60")))

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    def __post_init__(self) -> None:
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        if self.sentry_dsn:
            try:
                import sentry_sdk
                sentry_sdk.init(
                    dsn=self.sentry_dsn,
                    environment=self.app_env,
                    traces_sample_rate=0.2,
                )
            except ImportError:
                pass

    def validate(self) -> list[str]:
        """
        Returns a list of missing required config keys.
        Call this at startup to surface clear errors rather than
        cryptic KeyErrors deep in the call stack.
        """
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.kroger_client_id:
            missing.append("KROGER_CLIENT_ID")
        if not self.kroger_client_secret:
            missing.append("KROGER_CLIENT_SECRET")
        return missing


# Singleton — import this everywhere
settings = Settings()
