"""Configuration settings for spotq-eta-service loaded dynamically via Infisical."""

from enum import Enum
from typing import Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

DEFAULT_MODEL_PATH = (
    "app/infrastructure/ml_models/artifacts/eta_model_v1.json"
)

class EnvironmentType(str, Enum):
    """Application environment modes."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings mapped to Infisical runtime environment variables."""

    # Core Application Configuration
    ENVIRONMENT: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Deployment environment mode",
    )
    PORT: int = Field(default=8000, description="Service HTTP port")
    LOG_LEVEL: str = Field(default="INFO", description="Logging output level")

    # Redis Cloud Configuration
    REDIS_HOST: str = Field(
        default="localhost", description="Redis host address"
    )
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_USERNAME: str = Field(
        default="default", description="Redis username"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None, description="Redis authentication password"
    )

    # MongoDB Ingestion Buffer Configuration
    MONGO_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection string",
    )
    MONGO_DB_NAME: str = Field(
        default="spotq_eta_db", description="MongoDB database name"
    )

    # XGBoost Machine Learning Artifact Path
    MODEL_PATH: str = Field(
        default="app/infrastructure/ml_models/artifacts/eta_model_v1.json",
        description="Path to serialized XGBoost model artifact",
    )
    ALLOW_SYNTHETIC_MODEL: bool = Field(
        default=False,
        description="Allow a synthetic-data model for development only",
    )


    DEFAULT_RESTAURANT_TIMEZONE: str = Field(
        default="UTC",
        description="Default timezone for restaurants without a configured timezone",
    )
    COLD_START_AVERAGE_WAIT_MINUTES: float = Field(
        default=20.0, ge=0, le=120,
        description="Fallback historical average wait in minutes",
    )
    REDIS_LOOKUP_TIMEOUT_SECONDS: float = Field(
        default=0.015, gt=0, le=1,
        description="Total Redis history lookup timeout in seconds",
    )
    ETA_MIN_WAIT_MINUTES: float = Field(default=1.0, gt=0, le=120)
    ETA_MAX_WAIT_MINUTES: float = Field(default=120.0, gt=0, le=120)

    # BullMQ event consumer
    BULLMQ_QUEUE_NAME: str = Field(
        default="ml-eta-queue",
        description="BullMQ queue used by the MERN producer",
    )
    BULLMQ_PREFIX: str = Field(
        default="bull",
        description="BullMQ Redis key prefix; must match the producer",
    )
    BULLMQ_CONCURRENCY: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent queue-event jobs",
    )

    @model_validator(mode="after")
    def validate_production_model(self) -> "Settings":
        """Enforce production model configuration safeguards."""
        if self.ENVIRONMENT == EnvironmentType.PRODUCTION:
            if self.ALLOW_SYNTHETIC_MODEL:
                raise ValueError(
                    "ALLOW_SYNTHETIC_MODEL must be false in production."
                )

            configured_path = Path(self.MODEL_PATH).resolve()
            default_path = Path(DEFAULT_MODEL_PATH).resolve()

            if configured_path == default_path:
                raise ValueError(
                    "Production must use a validated model artifact, "
                    "not the default eta_model_v1.json."
                )

        return self

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()