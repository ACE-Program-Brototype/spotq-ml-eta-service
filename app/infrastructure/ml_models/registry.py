"""Model version metadata manifest and registry schemas for SCRUM-1228."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional

from app.infrastructure.ml_models.features import FEATURE_NAMES


@dataclass
class ModelManifest:
    """Versioned model metadata associating candidate artifacts with their training provenance."""

    model_version: str
    s3_uri: str
    checksum_sha256: str
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    dataset_mode: str = "synthetic"
    metrics: dict[str, Any] = field(default_factory=dict)
    mlflow_run_id: Optional[str] = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_eligible: bool = False
    artifact_format: str = "xgboost_json"
    feature_contract_version: str = "v1"
    promoted_at_utc: Optional[str] = None
    promoted_by: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelManifest":
        """Reconstruct ModelManifest from dictionary."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: Path) -> "ModelManifest":
        """Load manifest from JSON file."""
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: Path) -> None:
        """Write manifest atomically to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
