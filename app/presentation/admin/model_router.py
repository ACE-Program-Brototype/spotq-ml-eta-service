"""Authenticated model promotion, reload, and rollback endpoints for SCRUM-1228."""

from datetime import datetime, timezone
import hmac
from pathlib import Path
import tempfile
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.loader import model_loader
from app.infrastructure.ml_models.validator import validate_candidate_for_promotion
from app.infrastructure.observability.metrics import (
    MODEL_PROMOTION_FAILURE_TOTAL,
    MODEL_PROMOTION_SUCCESS_TOTAL,
    MODEL_ROLLBACK_FAILURE_TOTAL,
    MODEL_ROLLBACK_SUCCESS_TOTAL,
)
from app.infrastructure.storage.s3_storage import (
    ArtifactNotFoundError,
    ChecksumMismatchError,
    StorageError,
    s3_storage,
)

router = APIRouter(prefix="/api/v1/eta/admin", tags=["ETA Model Administration"])


class PromotionRequest(BaseModel):
    """Payload to authorize promotion of a candidate model version."""

    model_version: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Version tag of the registered candidate model",
    )
    approver_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identity of the operator authorizing promotion",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional operational notes or reason for promotion",
    )


def authorize(x_admin_key: str | None) -> None:
    """Validate the configured admin key using constant-time comparison."""
    expected = settings.ETA_MODEL_ADMIN_KEY
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model administration is not configured",
        )
    if x_admin_key is None or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing admin key",
        )


def model_state() -> dict[str, Any]:
    """Return consistent model metadata without exposing raw host filesystem paths."""
    snapshot = model_loader.get_snapshot()
    return {
        "status": getattr(snapshot, "status", None),
        "model_version": getattr(snapshot, "model_version", None),
        "artifact_id": getattr(snapshot, "artifact_id", None),
        "loaded_at": getattr(snapshot, "loaded_at", None),
        "checksum_sha256": getattr(snapshot, "checksum_sha256", None),
        "dataset_mode": getattr(snapshot, "dataset_mode", None),
        "is_healthy": model_loader.is_healthy(),
        "has_previous_model": model_loader.previous_booster is not None,
        "previous_model_version": model_loader.previous_model_version,
    }


@router.get("/model-status")
def get_active_model_status(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, Any]:
    """Inspect active model state, checksum, loaded version, and health."""
    authorize(x_admin_key)
    return {
        "active_model": model_state(),
        "storage_mode": "remote_s3" if s3_storage.is_remote else "local_fallback",
        "s3_bucket": s3_storage.bucket_name,
    }


@router.get("/candidates")
def list_candidates(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, Any]:
    """List registered candidate model artifacts in storage with their validation metadata."""
    authorize(x_admin_key)
    candidates = s3_storage.list_candidates()
    return {
        "candidate_count": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }


@router.post("/promote-candidate")
def promote_candidate(
    payload: PromotionRequest,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, Any]:
    """Validate, promote, and atomically activate an eligible candidate model from storage."""
    authorize(x_admin_key)

    logger.info(
        "Candidate promotion requested",
        model_version=payload.model_version,
        approver_id=payload.approver_id,
        notes=payload.notes,
    )

    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = Path(staging_dir)

        # 1. Download and verify SHA-256 integrity
        try:
            downloaded_file, manifest = s3_storage.download_candidate(
                payload.model_version, staging_path
            )
        except ArtifactNotFoundError as exc:
            MODEL_PROMOTION_FAILURE_TOTAL.inc()
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Candidate artifact not found in storage: {exc}",
            ) from exc
        except ChecksumMismatchError as exc:
            MODEL_PROMOTION_FAILURE_TOTAL.inc()
            logger.critical("Integrity violation during candidate download", error=str(exc))
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Artifact integrity error: {exc}",
            ) from exc
        except StorageError as exc:
            MODEL_PROMOTION_FAILURE_TOTAL.inc()
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Storage retrieval failed: {exc}",
            ) from exc

        # 2. Isolated Pre-Promotion Validation Gate
        validation_result = validate_candidate_for_promotion(manifest, downloaded_file)
        if not validation_result.is_valid:
            MODEL_PROMOTION_FAILURE_TOTAL.inc()
            logger.warning(
                "Candidate rejected during pre-promotion validation",
                model_version=payload.model_version,
                reason=validation_result.rejection_reason,
            )
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Candidate promotion rejected: {validation_result.rejection_reason}",
            )

        # 3. Update Audit Provenance in Manifest
        manifest.promoted_at_utc = datetime.now(timezone.utc).isoformat()
        manifest.promoted_by = payload.approver_id

        # 4. Atomic In-Memory Hot-Swap
        activation_succeeded = model_loader.activate_candidate(downloaded_file, manifest)
        if not activation_succeeded:
            MODEL_PROMOTION_FAILURE_TOTAL.inc()
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Model activation failed; the active model was safely preserved",
            )

        MODEL_PROMOTION_SUCCESS_TOTAL.inc()
        logger.info(
            "Candidate model promoted and activated successfully",
            model_version=payload.model_version,
            approver_id=payload.approver_id,
            checksum_sha256=manifest.checksum_sha256,
        )

        return {
            "message": "Candidate model successfully promoted and activated",
            "promoted_version": payload.model_version,
            "approver_id": payload.approver_id,
            "promoted_at_utc": manifest.promoted_at_utc,
            **model_state(),
        }


@router.post("/reload-model")
def reload_model(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, Any]:
    """Validate and reload the model from the configured default artifact path."""
    authorize(x_admin_key)
    if not model_loader.load_model(is_reload=True):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model reload failed; the existing model was preserved if available",
        )
    return {"message": "Model reloaded successfully", **model_state()}


@router.post("/rollback-model")
def rollback_model(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, Any]:
    """Atomically restore the previous known-good model version in this process."""
    authorize(x_admin_key)
    if not model_loader.rollback_model():
        MODEL_ROLLBACK_FAILURE_TOTAL.inc()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No previous model is available for rollback",
        )

    MODEL_ROLLBACK_SUCCESS_TOTAL.inc()
    return {"message": "Model rolled back successfully", **model_state()}
