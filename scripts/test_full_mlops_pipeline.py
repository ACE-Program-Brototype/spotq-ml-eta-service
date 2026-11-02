"""End-to-End MLOps Pipeline Integration Verification Script.

Tests the full lifecycle with Infisical secrets and live S3:
1. Candidate Registry Listing (GET /api/v1/eta/admin/candidates)
2. Controlled Promotion & S3 Download & SHA-256 Validation (POST /api/v1/eta/admin/promote-candidate)
3. Active Model Status Check (GET /api/v1/eta/admin/model-status)
4. Live Prediction Inference (GET /api/v1/eta/predict)
5. Safe Rollback Mechanism (POST /api/v1/eta/admin/rollback-model)

Usage:
    infisical run --env=dev -- uv run python scripts/test_full_mlops_pipeline.py
"""

import sys
from fastapi.testclient import TestClient
from app.main import app
from app.infrastructure.config.settings import settings


def run_pipeline_test() -> bool:
    print("=" * 75)
    print("SPOTQ FULL MLOPS PIPELINE VERIFICATION")
    print("=" * 75)

    with TestClient(app) as client:
        admin_key = settings.ETA_MODEL_ADMIN_KEY
        if not admin_key:
            print("ERROR: ETA_MODEL_ADMIN_KEY is not configured in Infisical.")
            return False

        headers = {"X-Admin-Key": admin_key}

        # 1. Baseline status
        print("\n1. Checking baseline model status...")
        r = client.get("/api/v1/eta/admin/model-status", headers=headers)
        if r.status_code != 200:
            print(f"FAILED to query model status: {r.status_code} {r.text}")
            return False
        active_before = r.json()["active_model"]["model_version"]
        print(f"   Active model baseline: {active_before}")

        # 2. List candidates in S3 Registry
        print("\n2. Querying model candidates from S3 Registry...")
        r = client.get("/api/v1/eta/admin/candidates", headers=headers)
        if r.status_code != 200:
            print(f"FAILED to list candidates: {r.status_code} {r.text}")
            return False
        candidates = r.json().get("candidates", [])
        if not candidates:
            print("ERROR: No candidates found in S3 registry.")
            return False
        target_version = candidates[-1]["model_version"]
        print(f"   Found {len(candidates)} candidates in S3.")
        print(f"   Selected candidate for promotion: {target_version}")

        # 3. Promote candidate
        print(f"\n3. Promoting candidate '{target_version}' via Admin API...")
        r = client.post(
            "/api/v1/eta/admin/promote-candidate",
            headers=headers,
            json={"model_version": target_version, "approver_id": "mlops_tester"},
        )
        if r.status_code != 200:
            print(f"FAILED promotion: {r.status_code} {r.text}")
            return False
        print(f"   Promotion successful! Promoted version: {r.json()['promoted_version']}")

        # 4. Verify active version
        print("\n4. Verifying active model status...")
        r = client.get("/api/v1/eta/admin/model-status", headers=headers)
        active_now = r.json()["active_model"]["model_version"]
        print(f"   Currently serving version: {active_now}")
        assert active_now == target_version

        # 5. Live prediction test
        print("\n5. Testing live inference on promoted model (/api/v1/eta/predict)...")
        r = client.get(
            "/api/v1/eta/predict",
            params={
                "restaurant_id": "mock-restaurant-001",
                "party_size": 2,
                "total_people_ahead": 5,
            },
        )
        if r.status_code != 200:
            print(f"FAILED prediction: {r.status_code} {r.text}")
            return False
        pred_data = r.json()
        print(f"   Prediction served: {pred_data['estimated_wait_minutes']} minutes (HTTP 200)")

        # 6. Safe rollback
        print("\n6. Testing safe model rollback...")
        r = client.post(
            "/api/v1/eta/admin/rollback-model",
            headers=headers,
            json={"reason": "Automated verification rollback test"},
        )
        if r.status_code != 200:
            print(f"FAILED rollback: {r.status_code} {r.text}")
            return False
        print(f"   Rollback successful! Restored version: {r.json()['model_version']}")

        # 7. Final status check
        print("\n7. Confirming model status after rollback...")
        r = client.get("/api/v1/eta/admin/model-status", headers=headers)
        final_active = r.json()["active_model"]["model_version"]
        print(f"   Final active version: {final_active}")
        assert final_active == active_before

    print("\n" + "=" * 75)
    print("ALL MLOPS PIPELINE CHECKS PASSED: S3 REGISTRY, PROMOTION & ROLLBACK VERIFIED!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
