"""Shared schema for completed queue events."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ETAIngestRequest(BaseModel):
    """Validated completed queue event."""

    queue_entry_id: str = Field(..., min_length=1, max_length=128)
    restaurant_id: str = Field(..., min_length=1, max_length=128)
    party_size: int = Field(..., ge=1, le=100)
    joined_at: str = Field(..., min_length=1)
    completed_at: str | None = Field(default=None, min_length=1)
    seated_at: str | None = Field(default=None, min_length=1)
    actual_wait_time_minutes: float | None = Field(
        default=None, gt=0, le=120, allow_inf_nan=False
    )
    total_people_ahead_at_join: int = Field(..., ge=0, le=1000)

    @model_validator(mode="before")
    @classmethod
    def populate_completed_at(cls, data: object) -> object:
        """Support both seated_at and completed_at from queue events."""
        if isinstance(data, dict):
            if not data.get("completed_at") and data.get("seated_at"):
                data["completed_at"] = data["seated_at"]
            elif not data.get("seated_at") and data.get("completed_at"):
                data["seated_at"] = data["completed_at"]
        return data

    @model_validator(mode="after")
    def validate_queue_event(self):
        """Require timezone-aware timestamps in chronological order and compute wait time if omitted."""
        for field_name in ("queue_entry_id", "restaurant_id"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")

        if not self.completed_at:
            raise ValueError("completed_at or seated_at timestamp is required")

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

        calculated_wait = (completed - joined).total_seconds() / 60.0
        if calculated_wait <= 0 or calculated_wait > 120.0:
            raise ValueError(
                f"Wait duration {calculated_wait:.2f} minutes is outside valid bounds (0, 120]"
            )

        if self.actual_wait_time_minutes is None:
            self.actual_wait_time_minutes = round(calculated_wait, 2)
        elif abs(self.actual_wait_time_minutes - calculated_wait) > 0.51:
            raise ValueError(
                f"Reported wait time {self.actual_wait_time_minutes}m does not match timestamp difference {calculated_wait:.2f}m"
            )

        return self
