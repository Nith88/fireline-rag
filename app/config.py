"""Settings, read from environment variables / .env."""
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anthropic does not offer an embeddings model, so embeddings come from another provider.
DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "voyage": "voyage-3",
    "local": "sentence-transformers/all-MiniLM-L6-v2",
    "fake": "fake-deterministic",  # offline plumbing tests only; similarity is meaningless
}
DEFAULT_DIMS = {"openai": 1536, "voyage": 1024, "local": 384, "fake": 1536}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://fireline:fireline@localhost:5432/fireline"

    embedding_provider: str = "openai"
    embedding_model: str | None = None
    embedding_dim: int | None = None

    llm_model: str = "claude-sonnet-5-5"
    # The design doc budgets 3 s for a dashboard summary. A Q&A answer is a longer call, so this
    # defaults higher; measure p95 and tune.
    llm_timeout_s: float = 8.0

    k_incidents: int = 5
    k_runbooks: int = 3
    min_similarity: float = 0.25

    @model_validator(mode="after")
    def _fill_defaults(self) -> "Settings":
        if self.embedding_provider not in DEFAULT_MODELS:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {sorted(DEFAULT_MODELS)}")
        self.embedding_model = self.embedding_model or DEFAULT_MODELS[self.embedding_provider]
        self.embedding_dim = self.embedding_dim or DEFAULT_DIMS[self.embedding_provider]
        return self

    @property
    def embedding_model_id(self) -> str:
        """Stored on every chunk; retrieval only compares vectors with the same id."""
        return f"{self.embedding_provider}:{self.embedding_model}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
