"""Load, validate, and report the state of the active XGBoost ETA model."""

from pathlib import Path
from typing import Literal

import xgboost as xgb

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.features import FEATURE_NAMES


ModelStatus = Literal[
    "loaded", "loaded_synthetic", "missing", "incompatible", "not_loaded"
]


class ModelLoader:
    """Manage one schema-validated XGBoost model shared by prediction requests."""

    def __init__(self) -> None:
        """Initialize the loader without assuming a model is available."""
        self.booster: xgb.Booster | None = None
        self.status: ModelStatus = "not_loaded"
        self.failure_reason: str | None = None

    def load_model(self) -> bool:
        """Load the configured artifact and validate its exact feature schema.

        Returns:
            True if the model is compatible and loaded; otherwise False.
        """
        self.booster = None
        self.status = "not_loaded"
        self.failure_reason = None
        model_path = Path(settings.MODEL_PATH)

        if not model_path.is_file():
            self.status = "missing"
            self.failure_reason = "Configured model artifact does not exist"
            logger.warning(
                "XGBoost ETA model artifact not found",
                path=str(model_path),
            )
            return False

        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            actual_names = booster.feature_names

            if actual_names is None or tuple(actual_names) != FEATURE_NAMES:
                self.status = "incompatible"
                self.failure_reason = (
                    "Model feature schema mismatch: "
                    f"expected {list(FEATURE_NAMES)}, got {actual_names}"
                )
                logger.error(
                    "XGBoost ETA model has an incompatible feature schema",
                    path=str(model_path),
                    expected_features=list(FEATURE_NAMES),
                    actual_features=actual_names,
                )
                return False

            dataset_mode = booster.attr("dataset_mode")

            if (
                settings.ENVIRONMENT == EnvironmentType.PRODUCTION
                and dataset_mode != "real"
            ):
                self.status = "incompatible"
                self.failure_reason = (
                    "Production requires verified real-data model metadata"
                )
                logger.error(
                    "Production rejected model without verified real-data provenance",
                    path=str(model_path),
                    dataset_mode=dataset_mode,
                )
                return False

            if dataset_mode == "synthetic" and not (
                settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
                and settings.ALLOW_SYNTHETIC_MODEL
            ):
                self.status = "incompatible"
                self.failure_reason = (
                    "Synthetic model rejected; explicitly enable "
                    "ALLOW_SYNTHETIC_MODEL in development only"
                )
                logger.error(
                    "Synthetic ETA model rejected by environment policy",
                    path=str(model_path),
                    environment=settings.ENVIRONMENT.value,
                )
                return False

            self.booster = booster
            self.status = (
                "loaded_synthetic" if dataset_mode == "synthetic" else "loaded"
            )
            logger.info(
                "XGBoost ETA model loaded and schema validated",
                path=str(model_path),
                feature_count=len(FEATURE_NAMES),
                dataset_mode=dataset_mode or "unspecified",
                model_status=self.status,
            )
            return True

        except Exception as exc:
            self.status = "incompatible"
            self.failure_reason = (
                f"Model artifact could not be loaded or validated: "
                f"{type(exc).__name__}"
            )
            logger.error(
                "XGBoost ETA model failed to load or validate",
                path=str(model_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False

    def is_healthy(self) -> bool:
        """Return whether a compatible model is loaded and permitted to serve."""
        if self.booster is None:
            return False
        if self.status == "loaded":
            return True
        return (
            self.status == "loaded_synthetic"
            and settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
            and settings.ALLOW_SYNTHETIC_MODEL
        )


model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Provide the process-wide model loader to application dependencies."""
    return model_loader
