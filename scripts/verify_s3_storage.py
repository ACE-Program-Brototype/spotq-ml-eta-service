"""Verification script for S3 artifact storage, Infisical runtime configuration, and integrity verification.

Usage:
    # With Infisical:
    infisical run --env=dev -- uv run python scripts/verify_s3_storage.py

    # Or with local .env:
    uv run python scripts/verify_s3_storage.py
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import xgboost as xgb

from app.infrastructure.config.settings import settings
from app.infrastructure.ml_models.features import FEATURE_NAMES
from app.infrastructure.ml_models.validator import validate_candidate_for_promotion
from app.infrastructure.storage.s3_storage import S3ModelStorage, compute_sha256


def verify_storage_lifecycle() -> bool:
    """Perform end-to-end verification of S3 storage and candidate integrity."""
    print("=" * 65)
    print("SPOTQ S3 ARTIFACT STORAGE & INFISICAL VERIFICATION")
    print("=" * 65)

    print(f"Environment Mode : {settings.ENVIRONMENT}")
    print(f"S3 Bucket        : {settings.S3_BUCKET_NAME}")
    print(f"S3 Region        : {settings.S3_REGION}")
    print(f"Artifact Prefix  : {settings.S3_ARTIFACT_PREFIX}")
    print(f"AWS Credentials  : {'Found' if settings.AWS_ACCESS_KEY_ID else 'Not configured'}")
    print("-" * 65)

    if not settings.S3_BUCKET_NAME:
        print("ERROR: S3_BUCKET_NAME (or AWS_S3_BUCKET) is not configured.")
        print("Ensure secrets are set in Infisical or in local .env.")
        return False

    storage = S3ModelStorage()
    print(f"Storage Mode     : {'Remote S3' if storage.is_remote else 'Local Directory'}")
    if not storage.is_remote:
        print("ERROR: S3ModelStorage is not operating in remote S3 mode.")
        return False

    test_version = "verify-probe-v1"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        booster_file = temp_path / "candidate_model.json"

        # 1. Generate valid dummy booster with canonical 6 features
        print("\n1. Generating test candidate model...")
        X = np.asarray([[1.0, 0.0, 12.0, 0.0, 0.0, 20.0]], dtype=np.float32)
        y = np.asarray([15.0], dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
        booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=1)
        booster.set_attr(dataset_mode="synthetic", model_version=test_version)
        booster.save_model(str(booster_file))
        local_sha = compute_sha256(booster_file)
        print(f"   Candidate generated: SHA-256 = {local_sha[:16]}...")

        # 2. Upload to S3
        print(f"\n2. Uploading candidate to S3 prefix '{storage.prefix}/{test_version}'...")
        manifest = storage.upload_candidate(
            model_version=test_version,
            model_file_path=booster_file,
            dataset_mode="synthetic",
            metrics={"mae": 1.8, "rmse": 2.4},
            is_eligible=True,
        )
        print(f"   Upload successful: S3 URI = {manifest.s3_uri}")
        print(f"   Manifest Checksum : {manifest.checksum_sha256[:16]}...")
        assert manifest.checksum_sha256 == local_sha

        # 3. Download and verify integrity
        print("\n3. Downloading candidate artifact from S3 to isolated staging...")
        staging_dir = temp_path / "staging"
        downloaded_file, downloaded_manifest = storage.download_candidate(test_version, staging_dir)
        print(f"   Downloaded artifact : {downloaded_file}")
        print(f"   Downloaded manifest : model_version={downloaded_manifest.model_version}")

        # 4. Validate candidate for promotion
        print("\n4. Running promotion validator (SHA-256 + 6-feature schema + booster)...")
        result = validate_candidate_for_promotion(downloaded_manifest, downloaded_file)
        print(f"   Validation Result   : {'PASSED' if result.is_valid else 'FAILED'}")
        if not result.is_valid:
            print(f"   Rejection Reason    : {result.rejection_reason}")
            return False

        # 5. Clean up test probe from S3
        print("\n5. Cleaning up probe artifact from S3...")
        if storage._s3_client:
            storage._s3_client.delete_object(
                Bucket=storage.bucket_name,
                Key=f"{storage.prefix}/{test_version}/eta_model.json",
            )
            storage._s3_client.delete_object(
                Bucket=storage.bucket_name,
                Key=f"{storage.prefix}/{test_version}/manifest.json",
            )
            print("   S3 test probe cleaned up.")

    print("\n" + "=" * 65)
    print("ALL S3 STORAGE & INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = verify_storage_lifecycle()
    sys.exit(0 if success else 1)
