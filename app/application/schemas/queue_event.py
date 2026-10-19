"""Shared schema for completed queue events."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ETAIngestRequest(BaseModel):
    """Validated completed queue event."""

    queue_entry_id: str = Field(..., min_length=1, max_length=128)
    restaurant_id: str = Field(..., min_length=1, max_length=128)
    party_size: int = Field(..., ge=1, le=100)
    joined_at: str = Field(..., min_length=1)
    completed_at: str = Field(..., min_length=1)
    actual_wait_time_minutes: float = Field(
        ..., gt=0, le=120, allow_inf_nan=False
    )
    total_people_ahead_at_join: int = Field(..., ge=0, le=1000)

    @model_validator(mode="after")
    def validate_queue_event(self):
        """Require timezone-aware timestamps in chronological order."""
        for field_name in ("queue_entry_id", "restaurant_id"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")

        try:
            joined = datetime.fromisoformat(
                self.joined_at.replace("Z", "+00:00")
            )
            completed = datetime.fromisoformat(
                self.completed_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "Timestamps must be valid ISO 8601 values"
            ) from exc

        if (
            joined.tzinfo is None
            or joined.utcoffset() is None
            or completed.tzinfo is None
            or completed.utcoffset() is None
        ):
            raise ValueError("Timestamps must include timezone information")

        if completed <= joined:
            raise ValueError("completed_at must be after joined_at")

        return self
