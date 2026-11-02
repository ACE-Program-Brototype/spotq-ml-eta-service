"""Load, validate, and manage the active XGBoost ETA model."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Literal

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
    checksum_sha256: str | None = None
    dataset_mode: str | None = None

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
        self.dataset_mode: str | None = None
        self.manifest: Any | None = None
        self.previous_booster: xgb.Booster | None = None
        self.previous_model_version: str | None = None
        self.previous_artifact_id: str | None = None
        self.previous_loaded_at: str | None = None
        self.previous_status: ModelStatus | None = None
        self.previous_dataset_mode: str | None = None
        self.previous_manifest: Any | None = None

    def get_snapshot(self) -> ModelSnapshot:
        """Return active model metadata from one consistent state."""
        with self._lock:
            checksum = (
                self.manifest.checksum_sha256 if hasattr(self.manifest, "checksum_sha256")
                else None
            )
            return ModelSnapshot(
                booster=self.booster,
                status=self.status,
                model_version=self.model_version,
                artifact_id=self.artifact_id,
                loaded_at=self.loaded_at,
                failure_reason=self.failure_reason,
                checksum_sha256=checksum,
                dataset_mode=self.dataset_mode,
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

        if (
            not is_reload
            and self.booster is None
            and not model_path.is_file()
            and settings.ENVIRONMENT != EnvironmentType.TESTING
        ):
            # Attempt to bootstrap from S3 storage if remote is configured
            try:
                from app.infrastructure.storage.s3_storage import s3_storage

                if s3_storage.is_remote:
                    candidates = s3_storage.list_candidates()
                    if candidates:
                        # Find matching version or pick the latest candidate
                        chosen = None
                        for c in candidates:
                            if c.model_version == model_path.stem:
                                chosen = c
                                break
                        target = chosen or candidates[0]
                        cache_dir = Path("app/infrastructure/ml_models/artifacts/s3_storage")
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        downloaded_file, manifest = s3_storage.download_candidate(
                            target.model_version, cache_dir
                        )
                        model_path = downloaded_file
                        logger.info(
                            "Bootstrapped model artifact from S3 registry",
                            model_version=target.model_version,
                            s3_uri=manifest.s3_uri,
                            checksum_sha256=manifest.checksum_sha256,
                        )
            except Exception as bootstrap_exc:
                logger.warning(
                    "S3 model bootstrap failed; falling back to local file load",
                    error=str(bootstrap_exc),
                )

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
            self.dataset_mode = dataset_mode
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

    def activate_candidate(
        self,
        candidate_path: Path,
        manifest: Any | None = None,
    ) -> bool:
        """Atomically activate a validated S3 candidate artifact.

        Preserves the previous model for instant rollback on failure or operator request.
        """
        started = perf_counter()
        try:
            candidate, candidate_status, version, dataset_mode = (
                self._load_candidate(candidate_path)
            )
        except Exception as exc:
            MODEL_RELOAD_FAILURE_TOTAL.inc()
            with self._lock:
                self.last_reload_error = str(exc)
            logger.error(
                "Candidate model activation failed; active model preserved",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False
        finally:
            duration = perf_counter() - started
            MODEL_RELOAD_DURATION.observe(duration)

        loaded_at = datetime.now(timezone.utc).isoformat()
        target_version = manifest.model_version if manifest else version

        with self._lock:
            if self.booster is not None:
                self.previous_booster = self.booster
                self.previous_model_version = self.model_version
                self.previous_artifact_id = self.artifact_id
                self.previous_loaded_at = self.loaded_at
                self.previous_status = self.status
                self.previous_dataset_mode = self.dataset_mode
                self.previous_manifest = self.manifest

            self.booster = candidate
            self.status = candidate_status
            self.model_version = target_version
            self.artifact_id = candidate_path.name
            self.loaded_at = loaded_at
            self.dataset_mode = dataset_mode
            self.manifest = manifest
            self.failure_reason = None
            self.last_reload_error = None

        MODEL_RELOAD_SUCCESS_TOTAL.inc()
        logger.info(
            "Candidate model activated as live production model",
            model_version=target_version,
            artifact_id=candidate_path.name,
            dataset_mode=dataset_mode,
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
                self.dataset_mode,
                self.manifest,
            )

            (
                self.booster,
                self.model_version,
                self.artifact_id,
                self.loaded_at,
                self.status,
                self.dataset_mode,
                self.manifest,
            ) = (
                self.previous_booster,
                self.previous_model_version,
                self.previous_artifact_id,
                self.previous_loaded_at,
                self.previous_status or "loaded",
                self.previous_dataset_mode,
                self.previous_manifest,
            )

            (
                self.previous_booster,
                self.previous_model_version,
                self.previous_artifact_id,
                self.previous_loaded_at,
                self.previous_status,
                self.previous_dataset_mode,
                self.previous_manifest,
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

        booster = getattr(snapshot, "booster", None)
        status_val = getattr(snapshot, "status", None)
        if booster is None:
            return status_val in ("loaded", "loaded_synthetic")

        if status_val == "loaded":
            return True

        return (
            status_val == "loaded_synthetic"
            and settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
            and settings.ALLOW_SYNTHETIC_MODEL
        )



model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Provide the process-wide model loader to application dependencies."""
    return model_loader
