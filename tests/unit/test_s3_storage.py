"""Unit tests for S3 artifact storage, SHA-256 checksum calculation, and integrity checks."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.infrastructure.storage.s3_storage import (
    ArtifactNotFoundError,
    ChecksumMismatchError,
    S3ModelStorage,
    compute_sha256,
    verify_checksum,
)


@pytest.fixture
def sample_model_file(tmp_path: Path) -> Path:
    """Create a temporary dummy model file."""
    model_path = tmp_path / "model.json"
    model_path.write_text('{"mock_model": true, "tree": [1, 2, 3]}', encoding="utf-8")
    return model_path


def test_compute_sha256_is_deterministic(sample_model_file: Path):
    digest1 = compute_sha256(sample_model_file)
    digest2 = compute_sha256(sample_model_file)
    assert digest1 == digest2
    assert len(digest1) == 64
    assert isinstance(digest1, str)


def test_verify_checksum_identifies_matching_and_tampered_files(sample_model_file: Path):
    valid_sha = compute_sha256(sample_model_file)
    assert verify_checksum(sample_model_file, valid_sha) is True
    assert verify_checksum(sample_model_file, "0" * 64) is False

    # Tamper with file
    sample_model_file.write_text('{"tampered": true}', encoding="utf-8")
    assert verify_checksum(sample_model_file, valid_sha) is False


def test_storage_upload_and_download_roundtrip(tmp_path: Path, sample_model_file: Path):
    storage_dir = tmp_path / "storage"
    storage = S3ModelStorage(local_dir=storage_dir)

    manifest = storage.upload_candidate(
        model_version="v1.0.0",
        model_file_path=sample_model_file,
        dataset_mode="synthetic",
        metrics={"mae": 2.5, "rmse": 3.8},
        mlflow_run_id="run-1234",
        is_eligible=True,
    )

    assert manifest.model_version == "v1.0.0"
    assert manifest.checksum_sha256 == compute_sha256(sample_model_file)
    assert manifest.metrics["mae"] == 2.5

    # Download to staging directory
    staging_dir = tmp_path / "staging"
    downloaded_file, downloaded_manifest = storage.download_candidate(
        "v1.0.0", staging_dir
    )

    assert downloaded_file.is_file()
    assert downloaded_manifest.model_version == "v1.0.0"
    assert compute_sha256(downloaded_file) == manifest.checksum_sha256


def test_download_rejects_corrupted_artifact(tmp_path: Path, sample_model_file: Path):
    storage_dir = tmp_path / "storage"
    storage = S3ModelStorage(local_dir=storage_dir)

    storage.upload_candidate(
        model_version="v2.0.0",
        model_file_path=sample_model_file,
    )

    # Intentionally corrupt the stored file
    stored_file = storage_dir / "models" / "v2.0.0" / "eta_model.json"
    stored_file.write_text("CORRUPTED_PAYLOAD_BITS", encoding="utf-8")

    staging_dir = tmp_path / "staging"
    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch for version v2.0.0"):
        storage.download_candidate("v2.0.0", staging_dir)

    # Corrupt target file must be cleaned up
    expected_target = staging_dir / "eta_model_v2.0.0.json"
    assert not expected_target.exists()


def test_storage_raises_artifact_not_found(tmp_path: Path):
    storage = S3ModelStorage(local_dir=tmp_path / "storage")
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        storage.get_candidate_manifest("non-existent-version")

    with pytest.raises(ArtifactNotFoundError, match="not found"):
        storage.download_candidate("non-existent-version", tmp_path / "staging")


def test_storage_list_candidates(tmp_path: Path, sample_model_file: Path):
    storage = S3ModelStorage(local_dir=tmp_path / "storage")

    storage.upload_candidate("cand-1", sample_model_file)
    storage.upload_candidate("cand-2", sample_model_file)

    candidates = storage.list_candidates()
    versions = [c.model_version for c in candidates]
    assert "cand-1" in versions
    assert "cand-2" in versions


def test_remote_s3_upload_and_download(tmp_path: Path, sample_model_file: Path):
    """Test remote S3 operations with mocked boto3 S3 client."""
    storage = S3ModelStorage(
        bucket_name="spotq-test-bucket",
        region="us-east-1",
        local_dir=tmp_path / "storage",
    )
    mock_s3 = MagicMock()
    storage._s3_client = mock_s3

    assert storage.is_remote is True

    # Upload
    manifest = storage.upload_candidate(
        model_version="v3.0.0",
        model_file_path=sample_model_file,
    )
    assert manifest.s3_uri == "s3://spotq-test-bucket/models/v3.0.0/eta_model.json"
    assert mock_s3.upload_file.called
    assert mock_s3.put_object.called
