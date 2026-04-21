"""Production config loaded strictly from environment variables."""
import logging
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Server
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # App
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "HR AI Agent"))
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))

    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    enable_nemo_guardrails: bool = field(
        default_factory=lambda: os.getenv("ENABLE_NEMO_GUARDRAILS", "true").lower() == "true"
    )

    # Security
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "dev-jwt-secret"))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expire_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_EXPIRE_MINUTES", "60")))
    jwt_refresh_expire_minutes: int = field(
        default_factory=lambda: int(os.getenv("JWT_REFRESH_EXPIRE_MINUTES", "10080"))
    )
    admin_email: str = field(default_factory=lambda: os.getenv("ADMIN_EMAIL", "admin@example.com").lower())
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "Admin@123456"))
    allowed_origins: list = field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*").split(",")
    )

    # Rate limiting
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")))

    # Budget
    daily_budget_usd: float = field(
        default_factory=lambda: float(os.getenv("DAILY_BUDGET_USD", "5.0"))
    )

    # Storage
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", ".chromadb"))
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "mysql+pymysql://agent_user:agent_pass@localhost:3306/agent_db",
        )
    )

    # Observability — Langfuse
    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    langfuse_host: str = field(default_factory=lambda: os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))

    def validate(self):
        logger = logging.getLogger(__name__)
        if self.environment == "production":
            if self.jwt_secret == "dev-jwt-secret":
                raise ValueError("JWT_SECRET must be set in production!")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — using mock LLM")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be > 0")
        if self.daily_budget_usd <= 0:
            raise ValueError("DAILY_BUDGET_USD must be > 0")
        if self.jwt_expire_minutes <= 0:
            raise ValueError("JWT_EXPIRE_MINUTES must be > 0")
        if self.jwt_refresh_expire_minutes <= 0:
            raise ValueError("JWT_REFRESH_EXPIRE_MINUTES must be > 0")
        return self


settings = Settings().validate()
