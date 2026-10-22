"""Load, validate, and manage the active XGBoost ETA model."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal
from dataclasses import dataclass
from threading import RLock

import numpy as np
import xgboost as xgb

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.observability.metrics import (
    MODEL_LOAD_DURATION,
    MODEL_LOAD_FAILURE_TOTAL,
    MODEL_LOAD_SUCCESS_TOTAL,
    MODEL_RELOAD_DURATION,
    MODEL_RELOAD_FAILURE_TOTAL,
    MODEL_RELOAD_SUCCESS_TOTAL,
)

ModelStatus = Literal[
    "loaded", "loaded_synthetic", "missing", "incompatible", "not_loaded"
]

@dataclass(frozen=True)
class ModelSnapshot:
    """Consistent view of the active model and its metadata."""

    booster: xgb.Booster | None
    status: ModelStatus
    model_version: str | None
    artifact_id: str | None
    loaded_at: str | None
    failure_reason: str | None

class ModelLoader:
    """Manage a validated model and retain the last known-good version."""

    def __init__(self) -> None:
        """Initialize model state without assuming an artifact is available."""
        self._lock = RLock()
        self.booster: xgb.Booster | None = None
        self.status: ModelStatus = "not_loaded"
        self.failure_reason: str | None = None
        self.last_reload_error: str | None = None
        self.model_version: str | None = None
        self.artifact_id: str | None = None
        self.loaded_at: str | None = None
        self.previous_booster: xgb.Booster | None = None
        self.previous_model_version: str | None = None
        self.previous_artifact_id: str | None = None
        self.previous_loaded_at: str | None = None
        self.previous_status: ModelStatus | None = None

    def get_snapshot(self) -> ModelSnapshot:
        """Return active model metadata from one consistent state."""
        with self._lock:
            return ModelSnapshot(
                booster=self.booster,
                status=self.status,
                model_version=self.model_version,
                artifact_id=self.artifact_id,
                loaded_at=self.loaded_at,
                failure_reason=self.failure_reason,
            )

    def _load_candidate(
        self, model_path: Path
    ) -> tuple[xgb.Booster, ModelStatus, str, str]:
        """Load and validate a candidate without modifying the active model."""
        if not model_path.is_file():
            raise FileNotFoundError("Configured model artifact does not exist")

        candidate = xgb.Booster()
        candidate.load_model(str(model_path))

        actual_names = candidate.feature_names
        if (
            candidate.num_features() != len(FEATURE_NAMES)
            or actual_names is None
            or tuple(actual_names) != FEATURE_NAMES
        ):
            raise ValueError(
                "Model feature schema mismatch: "
                f"expected {list(FEATURE_NAMES)}, got {actual_names}"
            )

        dataset_mode = candidate.attr("dataset_mode")
        if dataset_mode not in {"real", "synthetic"}:
            raise ValueError(
                "Model dataset_mode metadata must be 'real' or 'synthetic'"
            )
        if (
            settings.ENVIRONMENT == EnvironmentType.PRODUCTION
            and dataset_mode != "real"
        ):
            raise ValueError(
                "Production requires verified real-data model metadata"
            )

        if dataset_mode == "synthetic" and not (
            settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
            and settings.ALLOW_SYNTHETIC_MODEL
        ):
            raise ValueError(
                "Synthetic model rejected; explicitly enable "
                "ALLOW_SYNTHETIC_MODEL in development only"
            )

        smoke_matrix = xgb.DMatrix(
            np.asarray([[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32),
            feature_names=list(FEATURE_NAMES),
        )
        prediction = np.asarray(candidate.predict(smoke_matrix)).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction[0]):
            raise ValueError("Model smoke prediction was not a single finite value")

        model_status: ModelStatus = (
            "loaded_synthetic" if dataset_mode == "synthetic" else "loaded"
        )
        version = candidate.attr("model_version") or model_path.stem
        return candidate, model_status, version, dataset_mode or "unspecified"


    def load_model(self, *, is_reload: bool = False) -> bool:
        """Validate a candidate before atomically activating it.

        A failed reload never replaces the active model.
        """
        started = perf_counter()
        model_path = Path(settings.MODEL_PATH)

        try:
            candidate, candidate_status, version, dataset_mode = (
                self._load_candidate(model_path)
            )
        except Exception as exc:
            if is_reload:
                MODEL_RELOAD_FAILURE_TOTAL.inc()
            else:
                MODEL_LOAD_FAILURE_TOTAL.inc()

            with self._lock:
                self.last_reload_error = str(exc)
                if self.booster is None:
                    self.status = (
                        "missing" if isinstance(exc, FileNotFoundError)
                        else "incompatible"
                    )
                    self.failure_reason = str(exc)

            logger.error(
                "ETA model candidate rejected; active model preserved",
                error_type=type(exc).__name__,
                error=str(exc),
                is_reload=is_reload,
            )
            return False
        finally:
            duration = perf_counter() - started
            if is_reload:
                MODEL_RELOAD_DURATION.observe(duration)
            else:
                MODEL_LOAD_DURATION.observe(duration)

        loaded_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if self.booster is not None:
                self.previous_booster = self.booster
                self.previous_model_version = self.model_version
                self.previous_artifact_id = self.artifact_id
                self.previous_loaded_at = self.loaded_at
                self.previous_status = self.status

            self.booster = candidate
            self.status = candidate_status
            self.model_version = version
            self.artifact_id = model_path.name
            self.loaded_at = loaded_at
            self.failure_reason = None
            self.last_reload_error = None

        if is_reload:
            MODEL_RELOAD_SUCCESS_TOTAL.inc()
        else:
            MODEL_LOAD_SUCCESS_TOTAL.inc()

        logger.info(
            "ETA model loaded and validated",
            model_version=version,
            artifact_id=model_path.name,
            feature_count=len(FEATURE_NAMES),
            dataset_mode=dataset_mode,
            is_reload=is_reload,
        )
        return True



    def rollback_model(self) -> bool:
        """Reactivate the previous known-good model atomically."""
        with self._lock:
            if self.previous_booster is None:
                return False

            current = (
                self.booster,
                self.model_version,
                self.artifact_id,
                self.loaded_at,
                self.status,
            )

            (
                self.booster,
                self.model_version,
                self.artifact_id,
                self.loaded_at,
                self.status,
            ) = (
                self.previous_booster,
                self.previous_model_version,
                self.previous_artifact_id,
                self.previous_loaded_at,
                self.previous_status or "loaded",
            )

            (
                self.previous_booster,
                self.previous_model_version,
                self.previous_artifact_id,
                self.previous_loaded_at,
                self.previous_status,
            ) = current

            self.failure_reason = None

            logger.warning(
                "ETA model rolled back",
                model_version=self.model_version,
                artifact_id=self.artifact_id,
            )
            return True



    def is_healthy(self) -> bool:
        """Return whether a compatible model is loaded and permitted to serve."""
        snapshot = self.get_snapshot()

        if snapshot.booster is None:
            return False

        if snapshot.status == "loaded":
            return True

        return (
            snapshot.status == "loaded_synthetic"
            and settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
            and settings.ALLOW_SYNTHETIC_MODEL
        )



model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Provide the process-wide model loader to application dependencies."""
    return model_loader
