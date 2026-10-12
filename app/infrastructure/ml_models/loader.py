"""Load and validate the active XGBoost ETA model."""
from pathlib import Path

import xgboost as xgb

from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.features import FEATURE_NAMES


class ModelLoader:
    """Keep one schema-validated model in memory for prediction requests."""

    def __init__(self) -> None:
        self.booster: xgb.Booster | None = None

    def load_model(self) -> bool:
        """Load the configured artifact and reject incompatible feature schemas."""
        self.booster = None
        model_path = Path(settings.MODEL_PATH)

        if not model_path.is_file():
            logger.warning("XGBoost model artifact not found", path=str(model_path))
            return False

        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            actual_names = booster.feature_names

            if actual_names is None or tuple(actual_names) != FEATURE_NAMES:
                raise ValueError(
                    "Model feature schema mismatch: "
                    f"expected {list(FEATURE_NAMES)}, got {actual_names}"
                )

            self.booster = booster
            logger.info(
                "XGBoost ETA model loaded and schema validated",
                path=str(model_path),
                feature_count=len(FEATURE_NAMES),
            )
            return True
        except Exception as exc:
            logger.error(
                "XGBoost ETA model failed to load or validate",
                path=str(model_path),
                error=str(exc),
            )
            self.booster = None
            return False

    def is_healthy(self) -> bool:
        """Return whether a compatible model is loaded."""
        return self.booster is not None


model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """FastAPI dependency provider for the shared model loader."""
    return model_loader
