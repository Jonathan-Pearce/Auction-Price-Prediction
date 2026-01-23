# =============================================================================
# Auction Price Prediction - Configuration
# =============================================================================
"""
Central configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file.
All settings are validated and type-checked at startup.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Path Configuration
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Standard directories following cookiecutter-data-science structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
REFERENCES_DIR = PROJECT_ROOT / "references"


# =============================================================================
# Settings Classes
# =============================================================================


class MaxSoldSettings(BaseSettings):
    """MaxSold API configuration."""

    model_config = SettingsConfigDict(env_prefix="MAXSOLD_")

    # API endpoints (not configurable - hardcoded for MaxSold)
    base_url: str = "https://maxsold.maxsold.com/msapi"
    enriched_base_url: str = "https://api.maxsold.com"

    # Rate limiting
    rate_limit: int = Field(default=10, description="Requests per second")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries (seconds)")

    # Scraping settings
    batch_size: int = Field(default=100, description="Auctions per batch")
    items_limit: int = Field(default=2500, description="Max items per auction request")


class DatabaseSettings(BaseSettings):
    """DuckDB database configuration."""

    model_config = SettingsConfigDict(env_prefix="DUCKDB_")

    path: Path = Field(default=DATA_DIR / "auction.duckdb")
    read_only: bool = Field(default=False)


class HuggingFaceSettings(BaseSettings):
    """Hugging Face Hub configuration."""

    model_config = SettingsConfigDict(env_prefix="HF_")

    token: str | None = Field(default=None, description="HF API token")
    dataset_repo: str = Field(default="maxsold-auctions")
    model_repo: str = Field(default="auction-price-predictor")
    space_repo: str = Field(default="auction-price-predictor")
    organization: str | None = Field(default=None)

    @property
    def dataset_id(self) -> str:
        """Full dataset repository ID."""
        if self.organization:
            return f"{self.organization}/{self.dataset_repo}"
        return self.dataset_repo

    @property
    def model_id(self) -> str:
        """Full model repository ID."""
        if self.organization:
            return f"{self.organization}/{self.model_repo}"
        return self.model_repo

    @property
    def space_id(self) -> str:
        """Full space repository ID."""
        if self.organization:
            return f"{self.organization}/{self.space_repo}"
        return self.space_repo


class TrainingSettings(BaseSettings):
    """Model training configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    # Device settings
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto")
    torch_dtype: Literal["float32", "float16", "bfloat16"] = Field(default="float32")

    # Training hyperparameters (defaults - override via CLI or config files)
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)
    num_epochs: int = Field(default=100)
    early_stopping_patience: int = Field(default=10)

    # Experiment tracking
    wandb_project: str = Field(default="auction-price-prediction")
    wandb_api_key: str | None = Field(default=None)
    wandb_disabled: bool = Field(default=True)

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Resolve 'auto' to the best available device."""
        if v == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )


class GradioSettings(BaseSettings):
    """Gradio interface configuration."""

    model_config = SettingsConfigDict(env_prefix="GRADIO_")

    server_port: int = Field(default=7860)
    share: bool = Field(default=False)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # Sub-configurations
    maxsold: MaxSoldSettings = Field(default_factory=MaxSoldSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    api: APISettings = Field(default_factory=APISettings)
    gradio: GradioSettings = Field(default_factory=GradioSettings)

    # Paths (convenience accessors)
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def models_dir(self) -> Path:
        return MODELS_DIR

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


# =============================================================================
# Settings Instance
# =============================================================================


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


# =============================================================================
# Database Initialization
# =============================================================================


def init_database() -> None:
    """Initialize DuckDB database with schema."""
    import duckdb

    # Ensure data directory exists
    settings.database.path.parent.mkdir(parents=True, exist_ok=True)

    # Read and execute schema
    schema_path = REFERENCES_DIR / "schema.sql"
    if schema_path.exists():
        with duckdb.connect(str(settings.database.path)) as conn:
            schema_sql = schema_path.read_text()
            conn.execute(schema_sql)
            print(f"Database initialized at {settings.database.path}")
    else:
        print(f"Schema file not found at {schema_path}")


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging() -> None:
    """Configure logging with loguru."""
    from loguru import logger
    import sys

    # Remove default handler
    logger.remove()

    # Add custom handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Add file handler in production
    if settings.is_production:
        logger.add(
            "logs/app.log",
            rotation="10 MB",
            retention="1 week",
            level="INFO",
        )

    return logger
