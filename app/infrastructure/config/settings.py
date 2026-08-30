"""Configuration settings for spotq-eta-service loaded dynamically via Infisical."""

from enum import Enum
from typing import Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @model_validator(mode="after")
    def validate_production_model(self) -> "Settings":
        """Ensure production environments do not use synthetic dummy defaults."""
        if self.ENVIRONMENT == EnvironmentType.PRODUCTION:
            if "eta_model_v1.json" in self.MODEL_PATH:
                raise ValueError(
                    "Production environment requires an explicitly provisioned model path "
                    "instead of the synthetic development default."
                )
        return self

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()