"""Authenticated model reload and rollback endpoints."""

import hmac

from fastapi import APIRouter, Header, HTTPException

from app.infrastructure.config.settings import settings
from app.infrastructure.ml_models.loader import model_loader

router = APIRouter(prefix="/api/v1/eta/admin", tags=["ETA Model Administration"])


def authorize(x_admin_key: str | None) -> None:
    """Validate the configured admin key using constant-time comparison."""
    expected = settings.ETA_MODEL_ADMIN_KEY
    if not expected:
        raise HTTPException(503, "Model administration is not configured")
    if x_admin_key is None or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(401, "Invalid or missing admin key")


def model_state() -> dict[str, str | None]:
    """Return consistent model metadata without exposing filesystem paths."""
    snapshot = model_loader.get_snapshot()
    return {
        "status": snapshot.status,
        "model_version": snapshot.model_version,
        "artifact_id": snapshot.artifact_id,
        "loaded_at": snapshot.loaded_at,
    }


@router.post("/reload-model")
def reload_model(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, str | None]:
    """Validate and activate the model at the configured artifact location."""
    authorize(x_admin_key)
    if not model_loader.load_model(is_reload=True):
        raise HTTPException(
            503, "Model reload failed; the existing model was preserved if available"
        )
    return {"message": "Model reloaded successfully", **model_state()}


@router.post("/rollback-model")
def rollback_model(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, str | None]:
    """Reactivate the previous known-good model in this process."""
    authorize(x_admin_key)
    if not model_loader.rollback_model():
        raise HTTPException(409, "No previous model is available for rollback")
    return {"message": "Model rolled back successfully", **model_state()}
