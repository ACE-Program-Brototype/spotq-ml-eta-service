from pathlib import Path
import xgboost as xgb
from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger


class ModelLoader:
    """Loads and verifies XGBoost model artifacts for low-latency wait-time prediction."""

    def __init__(self):
        self.booster: xgb.Booster | None = None

    def load_model(self) -> bool:
        """Load model artifact from configured path."""
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            logger.warning("XGBoost model artifact not found on disk", path=str(model_path))
            return False

        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            self.booster = booster
            logger.info("Successfully loaded XGBoost model artifact", path=str(model_path))
            return True
        except Exception as exc:
            logger.error("Failed to load XGBoost model artifact", error=str(exc))
            return False

    def is_healthy(self) -> bool:
        """Check if model booster is actively loaded in memory."""
        return self.booster is not None


model_loader = ModelLoader()