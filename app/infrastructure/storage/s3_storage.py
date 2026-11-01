"""S3 Artifact Storage and SHA-256 integrity engine for versioned ETA model artifacts."""

import hashlib
import hmac
import json
from pathlib import Path
import shutil
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from app.infrastructure.config.settings import EnvironmentType, settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.ml_models.registry import ModelManifest


class StorageError(Exception):
    """Base exception for artifact storage operations."""


class ArtifactNotFoundError(StorageError):
    """Raised when the requested model artifact is not found in storage."""


class ChecksumMismatchError(StorageError):
    """Raised when downloaded artifact checksum does not match registered manifest."""


def compute_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hexadecimal digest of a file using chunked streaming.

    Args:
        file_path: Local path to the file.

    Returns:
        Hexadecimal SHA-256 checksum string.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify that a local file matches an expected SHA-256 digest in constant time.

    Args:
        file_path: Path to the downloaded file.
        expected_sha256: Expected 64-character hex digest.

    Returns:
        True if the file matches the expected digest, False otherwise.
    """
    if not file_path.is_file():
        return False
    actual_sha256 = compute_sha256(file_path)
    return hmac.compare_digest(actual_sha256.lower(), expected_sha256.lower())


class S3ModelStorage:
    """Durable versioned model storage supporting AWS S3 and offline local staging."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        prefix: Optional[str] = None,
        local_dir: Optional[Path] = None,
    ) -> None:
        if bucket_name is not None:
            self.bucket_name = bucket_name if bucket_name else None
        elif settings.ENVIRONMENT == EnvironmentType.TESTING:
            self.bucket_name = None
        else:
            self.bucket_name = settings.S3_BUCKET_NAME
        self.region = region or settings.S3_REGION
        self.endpoint_url = endpoint_url or settings.S3_ENDPOINT_URL
        self.prefix = (prefix or settings.S3_ARTIFACT_PREFIX).strip("/")
        self.local_dir = local_dir or Path(settings.S3_LOCAL_STORAGE_DIR)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._s3_client = None
        if self.bucket_name:
            client_kwargs: dict[str, Any] = {"region_name": self.region}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                client_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
                client_kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

            try:
                self._s3_client = boto3.client("s3", **client_kwargs)
            except Exception as exc:
                logger.warning(
                    "S3 client initialization failed; using local storage adapter",
                    error=str(exc),
                )
                self._s3_client = None

    @property
    def is_remote(self) -> bool:
        """Return True if connected to a real S3 backend, False if using local adapter."""
        return self._s3_client is not None and bool(self.bucket_name)

    def _get_s3_key_prefix(self, model_version: str) -> str:
        return f"{self.prefix}/{model_version}"

    def upload_candidate(
        self,
        model_version: str,
        model_file_path: Path,
        dataset_mode: str = "synthetic",
        metrics: Optional[dict[str, Any]] = None,
        mlflow_run_id: Optional[str] = None,
        is_eligible: bool = False,
    ) -> ModelManifest:
        """Store candidate artifact and its metadata manifest with SHA-256 verification.

        Args:
            model_version: Unique candidate version tag (e.g. 'v1.2.0' or 'candidate-20261031').
            model_file_path: Path to serialized booster JSON file.
            dataset_mode: Dataset type ('real' or 'synthetic').
            metrics: Training/evaluation metrics (MAE, RMSE, etc.).
            mlflow_run_id: Associated MLflow run identifier.
            is_eligible: Whether candidate satisfies qualification thresholds.

        Returns:
            Saved ModelManifest with checksum and storage URI.
        """
        if not model_file_path.is_file():
            raise FileNotFoundError(f"Model file not found at {model_file_path}")

        checksum = compute_sha256(model_file_path)
        metrics_dict = metrics or {}

        if self.is_remote:
            s3_model_key = f"{self._get_s3_key_prefix(model_version)}/eta_model.json"
            s3_manifest_key = f"{self._get_s3_key_prefix(model_version)}/manifest.json"
            s3_uri = f"s3://{self.bucket_name}/{s3_model_key}"

            manifest = ModelManifest(
                model_version=model_version,
                s3_uri=s3_uri,
                checksum_sha256=checksum,
                dataset_mode=dataset_mode,
                metrics=metrics_dict,
                mlflow_run_id=mlflow_run_id,
                is_eligible=is_eligible,
            )

            try:
                # Upload model file
                self._s3_client.upload_file(
                    str(model_file_path),
                    self.bucket_name,
                    s3_model_key,
                    ExtraArgs={"ContentType": "application/json"},
                )
                # Upload manifest
                manifest_json = manifest.to_json()
                self._s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_manifest_key,
                    Body=manifest_json.encode("utf-8"),
                    ContentType="application/json",
                )
            except ClientError as exc:
                raise StorageError(f"Failed to upload artifact to S3: {exc}") from exc
        else:
            # Local directory fallback storage
            version_dir = self.local_dir / self.prefix / model_version
            version_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = version_dir / "eta_model.json"
            local_manifest_path = version_dir / "manifest.json"

            shutil.copy2(model_file_path, local_model_path)
            s3_uri = f"file://{local_model_path.resolve()}"

            manifest = ModelManifest(
                model_version=model_version,
                s3_uri=s3_uri,
                checksum_sha256=checksum,
                dataset_mode=dataset_mode,
                metrics=metrics_dict,
                mlflow_run_id=mlflow_run_id,
                is_eligible=is_eligible,
            )
            manifest.save(local_manifest_path)

        logger.info(
            "Model artifact registered in storage",
            model_version=model_version,
            s3_uri=manifest.s3_uri,
            checksum_sha256=checksum,
            is_remote=self.is_remote,
        )
        return manifest

    def get_candidate_manifest(self, model_version: str) -> ModelManifest:
        """Retrieve the metadata manifest for a specific model version."""
        if self.is_remote:
            s3_manifest_key = f"{self._get_s3_key_prefix(model_version)}/manifest.json"
            try:
                response = self._s3_client.get_object(
                    Bucket=self.bucket_name, Key=s3_manifest_key
                )
                raw_data = json.loads(response["Body"].read().decode("utf-8"))
                return ModelManifest.from_dict(raw_data)
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    raise ArtifactNotFoundError(
                        f"Manifest for version {model_version} not found in S3"
                    ) from exc
                raise StorageError(f"S3 error retrieving manifest: {exc}") from exc
        else:
            local_manifest = self.local_dir / self.prefix / model_version / "manifest.json"
            if not local_manifest.is_file():
                raise ArtifactNotFoundError(
                    f"Local manifest for version {model_version} not found at {local_manifest}"
                )
            return ModelManifest.from_file(local_manifest)

    def download_candidate(
        self, model_version: str, target_dir: Path
    ) -> tuple[Path, ModelManifest]:
        """Download candidate model artifact and verify its SHA-256 integrity.

        Args:
            model_version: Model version identifier.
            target_dir: Local staging directory where the artifact should be saved.

        Returns:
            Tuple of (downloaded_file_path, manifest).

        Raises:
            ArtifactNotFoundError: If artifact or manifest does not exist.
            ChecksumMismatchError: If downloaded file checksum differs from manifest.
        """
        manifest = self.get_candidate_manifest(model_version)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"eta_model_{model_version}.json"

        if self.is_remote:
            s3_model_key = f"{self._get_s3_key_prefix(model_version)}/eta_model.json"
            try:
                self._s3_client.download_file(
                    self.bucket_name, s3_model_key, str(target_file)
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    raise ArtifactNotFoundError(
                        f"Artifact for version {model_version} not found in S3"
                    ) from exc
                raise StorageError(f"S3 download failed: {exc}") from exc
        else:
            source_file = self.local_dir / self.prefix / model_version / "eta_model.json"
            if not source_file.is_file():
                raise ArtifactNotFoundError(
                    f"Local artifact not found for {model_version} at {source_file}"
                )
            shutil.copy2(source_file, target_file)

        # Integrity Check: SHA-256 Verification
        if not verify_checksum(target_file, manifest.checksum_sha256):
            # Clean up corrupted file
            if target_file.is_file():
                target_file.unlink()
            raise ChecksumMismatchError(
                f"Checksum mismatch for version {model_version}! "
                f"Expected {manifest.checksum_sha256}, download was corrupted or modified."
            )

        logger.info(
            "Candidate artifact downloaded and integrity verified",
            model_version=model_version,
            checksum_sha256=manifest.checksum_sha256,
            download_path=str(target_file),
        )
        return target_file, manifest

    def list_candidates(self) -> list[ModelManifest]:
        """List all candidate model manifests found in storage."""
        manifests: list[ModelManifest] = []

        if self.is_remote:
            prefix = f"{self.prefix}/"
            try:
                paginator = self._s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if key.endswith("/manifest.json"):
                            parts = key[len(prefix) :].split("/")
                            if parts:
                                version = parts[0]
                                try:
                                    manifests.append(
                                        self.get_candidate_manifest(version)
                                    )
                                except Exception:
                                    pass
            except ClientError as exc:
                logger.error("Failed to list S3 model candidates", error=str(exc))
        else:
            base_dir = self.local_dir / self.prefix
            if base_dir.is_dir():
                for version_path in base_dir.iterdir():
                    if version_path.is_dir():
                        manifest_file = version_path / "manifest.json"
                        if manifest_file.is_file():
                            try:
                                manifests.append(ModelManifest.from_file(manifest_file))
                            except Exception:
                                pass

        return sorted(manifests, key=lambda m: m.created_at_utc, reverse=True)


# Global S3 Model Storage Instance
s3_storage = S3ModelStorage()
