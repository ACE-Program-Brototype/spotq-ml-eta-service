"""Configuration settings for spotq-eta-service loaded dynamically via Infisical."""

from enum import Enum
from typing import Optional
from pydantic import AliasChoices, Field, model_validator
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
    MONGO_COLLECTION_NAME: str = Field(
        default="queue_events",
        description="Collection storing completed queue events for training",
    )

    # MLflow Experiment Tracking Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="spotq-eta-training",
        description="MLflow experiment name for SpotQ ETA model training",
    )

    # XGBoost Machine Learning Artifact Path
    MODEL_PATH: str = Field(
        default="app/infrastructure/ml_models/artifacts/eta_model_v1.json",
        description="Path to serialized XGBoost model artifact",
    )
    ALLOW_SYNTHETIC_MODEL: bool = Field(
        default=True,
        description="Allow a synthetic-data model for development only",
    )

    ETA_MODEL_ADMIN_KEY: Optional[str] = Field(
        default=None,
        min_length=32,
        description="Secret required to authorize model reload operations",
    )

    # S3 Artifact Storage & Model Registry (SCRUM-1228)
    S3_BUCKET_NAME: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("S3_BUCKET_NAME", "AWS_S3_BUCKET"),
        description="S3 bucket storing versioned model artifacts",
    )
    S3_REGION: str = Field(
        default="ap-south-1",
        validation_alias=AliasChoices("S3_REGION", "AWS_REGION"),
        description="AWS region for S3 artifact storage",
    )
    S3_ENDPOINT_URL: Optional[str] = Field(
        default=None,
        description="Custom S3 endpoint URL (e.g. MinIO, LocalStack)",
    )
    S3_ARTIFACT_PREFIX: str = Field(
        default="ml-models/eta",
        description="S3 key prefix for versioned ETA model artifacts",
    )
    AWS_ACCESS_KEY_ID: Optional[str] = Field(
        default=None,
        description="AWS access key ID (optional when using IAM roles)",
    )
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(
        default=None,
        description="AWS secret access key (optional when using IAM roles)",
    )
    S3_LOCAL_STORAGE_DIR: str = Field(
        default="app/infrastructure/ml_models/artifacts/s3_storage",
        description="Local directory for S3 artifact staging and offline fallback",
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

    # RabbitMQ Event Integration (SCRUM-1257)
    RABBITMQ_URL: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="RabbitMQ connection URL",
    )
    RABBITMQ_HOST: str = Field(
        default="localhost",
        description="RabbitMQ broker host address",
    )
    RABBITMQ_PORT: int = Field(default=5672, description="RabbitMQ broker AMQP port")
    RABBITMQ_USER: str = Field(default="guest", description="RabbitMQ username")
    RABBITMQ_PASSWORD: str = Field(default="guest", description="RabbitMQ password")
    RABBITMQ_VHOST: str = Field(default="/", description="RabbitMQ virtual host")
    RABBITMQ_QUEUE_NAME: str = Field(
        default="ml-eta-queue",
        description="RabbitMQ queue name for ETA queue events",
    )
    RABBITMQ_EXCHANGE_NAME: str = Field(
        default="spotq.events",
        description="RabbitMQ exchange name for queue events",
    )
    RABBITMQ_ROUTING_KEY: str = Field(
        default="queue.event.completed",
        description="Routing key pattern for completed queue events",
    )
    RABBITMQ_PREFETCH_COUNT: int = Field(
        default=10,
        ge=1,
        le=100,
        description="RabbitMQ prefetch count (unacknowledged messages per consumer)",
    )
    RABBITMQ_RECONNECT_INTERVAL_SECONDS: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Seconds to wait before reconnecting to RabbitMQ broker",
    )

    # BullMQ event consumer (legacy / migration)
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

    # Model Monitoring & Retraining Triggers (SCRUM-1229)
    ENABLE_AUTOMATIC_RETRAINING_TRIGGERS: bool = Field(
        default=True,
        description="Master toggle allowing automatic retraining requests on drift or degradation",
    )
    RETRAINING_COOLDOWN_HOURS: float = Field(
        default=4.0,
        ge=0.1,
        description="Minimum cooldown in hours between automated retraining runs",
    )
    MONITORING_MIN_SAMPLE_SIZE: int = Field(
        default=30,
        ge=5,
        description="Minimum valid samples required for reliable performance and drift evaluation",
    )
    MONITORING_MAX_ACCEPTABLE_MAE_MINUTES: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum acceptable rolling MAE in minutes before triggering retraining",
    )
    MONITORING_MAX_ACCEPTABLE_RMSE_MINUTES: float = Field(
        default=8.0,
        gt=0.0,
        description="Maximum acceptable rolling RMSE in minutes before triggering retraining",
    )
    MONITORING_PSI_DRIFT_THRESHOLD: float = Field(
        default=0.25,
        gt=0.0,
        description="Population Stability Index (PSI) threshold for significant feature drift",
    )
    MONITORING_EVALUATION_WINDOW_HOURS: float = Field(
        default=24.0,
        ge=1.0,
        description="Default rolling lookback window in hours for monitoring evaluation",
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