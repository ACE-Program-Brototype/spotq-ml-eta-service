# SpotQ ETA Microservice - Complete API Documentation

Comprehensive API reference for the **SpotQ ETA Microservice** (`spotq-ml-eta-service`), accurately verified against the active codebase for wait-time prediction, health checks, Prometheus metrics, model lifecycle management (SCRUM-1228), and automated model monitoring and retraining triggers (SCRUM-1229).

---

## 1. General Information & Conventions

### Base URLs
- **Local Development:** `http://127.0.0.1:8000`
- **Docker Container:** `http://localhost:8000`
- **Production Host:** `http://<service-host>:<port>`

### Interactive API Explorers
- **Swagger UI:** `/docs`
- **ReDoc UI:** `/redoc`
- **OpenAPI Schema (JSON):** `/openapi.json`

### Authentication Scheme
- **Public Endpoints:** Health checks (`/health`, `/healthz`, `/livez`, `/readyz`), Prometheus metrics (`/metrics`), and the real-time prediction endpoint (`/api/v1/eta/predict`) require no authentication.
- **Administrative Endpoints:** All endpoints under `/api/v1/eta/admin/*` require the `X-Admin-Key` HTTP header configured via the `ETA_MODEL_ADMIN_KEY` environment variable. Constant-time digest comparison (`hmac.compare_digest`) is enforced to prevent timing attacks.

```http
X-Admin-Key: <your-configured-admin-key>
```

### Standard Error Response Schemas
When request query parameters or JSON bodies fail validation (`400 Bad Request`):
```json
{
  "error": "invalid_request",
  "message": "Request parameters or body are invalid",
  "details": [
    {
      "location": ["query", "party_size"],
      "message": "Input should be greater than or equal to 1"
    }
  ]
}
```

When an administrative endpoint receives an unauthorized request (`401 Unauthorized`):
```json
{
  "detail": "Invalid or missing admin key"
}
```

When a model or database dependency is unavailable (`503 Service Unavailable`):
```json
{
  "detail": {
    "code": "eta_model_unavailable",
    "message": "A compatible ETA model is not loaded"
  }
}
```

---

## 2. System Health & Observability Endpoints

### 2.1 Root Status
#### `GET /`
Returns basic service identity, running state, and current operating environment.

- **Auth:** Public
- **Response (200 OK):**
```json
{
  "service": "spotq-eta-service",
  "status": "running",
  "environment": "development"
}
```

---

### 2.2 Process Liveness Probe
#### `GET /livez`
Fast liveness probe reporting whether the Python ASGI process and event loop are responsive. Used by Kubernetes/Docker container healthchecks.

- **Auth:** Public
- **Response (200 OK):**
```json
{
  "status": "alive",
  "service": "spotq-eta-service"
}
```

---

### 2.3 Dependency Readiness Probe
#### `GET /readyz`
Inspects upstream critical dependencies: MongoDB connection, loaded XGBoost model artifact, and Redis feature store. Redis disconnection returns `degraded_fallback_available` rather than failing readiness, because the inference engine can fall back to historical averages or cold-start defaults.

- **Auth:** Public
- **Response (200 OK if ready, 503 if not ready):**
```json
{
  "status": "ready",
  "model": "loaded",
  "mongodb": "connected",
  "redis": "connected"
}
```

---

### 2.4 Detailed Health Diagnostic
#### `GET /health` or `GET /healthz`
Provides end-to-end diagnostic status including connection latencies to MongoDB, Redis, and loaded model metadata.

- **Auth:** Public
- **Response (200 OK if healthy, 503 if degraded):**
```json
{
  "status": "healthy",
  "service": "spotq-eta-service",
  "environment": "development",
  "dependencies": {
    "redis_feature_store": {
      "status": "connected",
      "latency_ms": 1.25
    },
    "mongodb_buffer": {
      "status": "connected",
      "latency_ms": 3.42
    },
    "xgboost_model_artifact": {
      "status": "loaded_synthetic",
      "version": "eta_model_candidate",
      "artifact_id": "eta_model_candidate.json",
      "loaded_at": "2026-11-06T12:00:00.000000Z"
    }
  }
}
```

---

### 2.5 Prometheus Metrics Scrape
#### `GET /metrics`
Exposes system, HTTP latency, and custom machine learning metrics formatted for Prometheus scraping.

- **Auth:** Public
- **Format:** `text/plain; version=0.0.4`
- **Key Custom Metrics Exported:**
  - `spotq_eta_request_latency_seconds_bucket`: Latency histogram for the prediction endpoint.
  - `spotq_eta_model_inference_latency_seconds_bucket`: Pure XGBoost inference duration histogram.
  - `spotq_redis_feature_lookup_latency_seconds_bucket`: Redis lookup duration histogram.
  - `spotq_rabbitmq_messages_processed_total`: Count of consumed completed queue events.
  - `spotq_monitoring_evaluations_total{status="success"}`: Counter of completed monitoring cycles.
  - `spotq_monitoring_mae_minutes{model_version="..."}`: Rolling MAE gauge per model version.
  - `spotq_monitoring_rmse_minutes{model_version="..."}`: Rolling RMSE gauge per model version.
  - `spotq_monitoring_feature_drift_psi{feature_name="..."}`: Population Stability Index drift gauge per canonical feature.
  - `spotq_monitoring_retraining_triggers_total{trigger_type="...", status="..."}`: Retraining decisions counter.

---

## 3. Real-Time ETA Prediction API

### 3.1 Predict Waiting Time
#### `GET /api/v1/eta/predict`

Calculates estimated waiting time in minutes using the active XGBoost model, Redis feature store hierarchy, and chronological feature derivation. Asynchronously buffers prediction features to MongoDB `prediction_logs` for zero-latency monitoring and ground truth outcome correlation.

- **Auth:** Public
- **Target SLA:** Sub-50ms (p95)

#### Query Parameters

| Parameter | Type | Required | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `restaurant_id` | String | **Yes** | $1 \le \text{len} \le 128$ | Target restaurant identifier. |
| `total_people_ahead` | Integer | **Yes** | $0 \le x \le 1000$ | Number of people in queue ahead of this party. |
| `party_size` | Integer | **Yes** | $1 \le x \le 100$ | Number of dining guests in this party. |
| `queue_entry_id` | String | No | Max 128 chars | Correlation ID linking prediction to eventual completed queue event. |

#### Example Request
```http
GET /api/v1/eta/predict?restaurant_id=rest_101&total_people_ahead=4&party_size=2&queue_entry_id=entry_9988 HTTP/1.1
Host: 127.0.0.1:8000
```

#### Verified Response Contract (200 OK)
```json
{
  "restaurant_id": "rest_101",
  "estimated_wait_minutes": 18.5,
  "confidence_score": null,
  "timestamp": "2026-11-06T15:00:00.000000+00:00",
  "fallback_used": false,
  "prediction_id": "pred_c3d4e5f6a7b81234"
}
```

#### Response Field Reference
- `restaurant_id` (String): Confirmed restaurant identifier.
- `estimated_wait_minutes` (Float): Final estimated wait time in minutes (clamped between configured `ETA_MIN_WAIT_MINUTES` and `ETA_MAX_WAIT_MINUTES`).
- `confidence_score` (Float | null): Optional model confidence score (defaults to `null`).
- `timestamp` (String): ISO 8601 UTC timestamp when prediction was generated.
- `fallback_used` (Boolean): `true` if exact (restaurant + day + hour) bucket was missing in Redis and fallback was applied; `false` otherwise.
- `prediction_id` (String): Unique tracking ID logged in MongoDB `prediction_logs` for drift and performance monitoring.

---

### 3.2 Legacy Queue Event Ingestion
#### `POST /api/v1/eta/ingest`
HTTP ingestion endpoint to persist completed queue events directly to MongoDB `queue_events` collection (supplementing RabbitMQ ingestion).

- **Auth:** Public
- **Request Body (`ETAIngestRequest`):**
```json
{
  "queue_entry_id": "entry_9988",
  "restaurant_id": "rest_101",
  "party_size": 2,
  "total_people_ahead_at_join": 4,
  "joined_at": "2026-11-06T14:30:00.000000Z",
  "completed_at": "2026-11-06T14:48:30.000000Z",
  "actual_wait_time_minutes": 18.5
}
```
- **Response (200 OK):**
```json
{
  "status": "success",
  "message": "Queue event durably persisted for background processing"
}
```

---

## 4. Model Lifecycle Administration APIs (SCRUM-1228)

Authenticated endpoints for inspecting, promoting, rolling back, and reloading machine learning model artifacts. All endpoints require `X-Admin-Key`.

### 4.1 Inspect Active Model Status
#### `GET /api/v1/eta/admin/model-status`
Returns metadata of the model currently active in memory, SHA-256 integrity checksum, loaded timestamp, dataset mode, and storage backend details.

- **Header:** `X-Admin-Key: <secret>`
- **Response (200 OK):**
```json
{
  "active_model": {
    "status": "loaded_synthetic",
    "model_version": "eta_model_candidate",
    "artifact_id": "eta_model_candidate.json",
    "loaded_at": "2026-11-06T12:00:00.000000Z",
    "checksum_sha256": "4f828bb702e86fb7b79d8542c38865d6b4122d4f2481e18d6e3ec1694f6c449c",
    "dataset_mode": "synthetic",
    "is_healthy": true,
    "has_previous_model": false,
    "previous_model_version": null
  },
  "storage_mode": "local_fallback",
  "s3_bucket": null
}
```

---

### 4.2 List Registered Candidate Models
#### `GET /api/v1/eta/admin/candidates`
Scans S3 and local storage registries to list registered candidate model artifacts and their validation manifests.

- **Header:** `X-Admin-Key: <secret>`
- **Response (200 OK):**
```json
{
  "candidate_count": 1,
  "candidates": [
    {
      "model_version": "eta_candidate_20261105_180000",
      "artifact_id": "eta_candidate_20261105_180000.json",
      "checksum_sha256": "8a3e7b1c...",
      "dataset_mode": "synthetic",
      "created_at_utc": "2026-11-05T18:00:00.000000Z"
    }
  ]
}
```

---

### 4.3 Promote Candidate Model
#### `POST /api/v1/eta/admin/promote-candidate`
Authorizes promotion of a registered candidate. Verifies SHA-256 checksum, validates the 6 canonical feature names, executes an atomic in-memory hot-swap with zero downtime, and keeps the previous model in memory for instant rollback.

- **Header:** `X-Admin-Key: <secret>`
- **Request Body (`PromotionRequest`):**
```json
{
  "model_version": "eta_candidate_20261105_180000",
  "approver_id": "ml_engineer_jeswin",
  "notes": "Candidate retrained on latest dataset with superior validation MAE"
}
```
- **Response (200 OK):**
```json
{
  "message": "Candidate model successfully promoted and activated",
  "promoted_version": "eta_candidate_20261105_180000",
  "approver_id": "ml_engineer_jeswin",
  "promoted_at_utc": "2026-11-06T15:10:00.000000Z",
  "status": "loaded_synthetic",
  "model_version": "eta_candidate_20261105_180000",
  "artifact_id": "eta_candidate_20261105_180000.json",
  "loaded_at": "2026-11-06T15:10:00.000000Z",
  "checksum_sha256": "8a3e7b1c...",
  "dataset_mode": "synthetic",
  "is_healthy": true,
  "has_previous_model": true,
  "previous_model_version": "eta_model_candidate"
}
```
- **Error Responses:**
  - **401 Unauthorized:** Invalid or missing `X-Admin-Key`.
  - **404 Not Found:** Candidate version not found in storage.
  - **422 Unprocessable Entity:** SHA-256 integrity failure, feature schema incompatibility, or synthetic candidate rejected when `ALLOW_SYNTHETIC_MODEL=false`.

---

### 4.4 Rollback Active Model
#### `POST /api/v1/eta/admin/rollback-model`
Instantly restores the previously loaded model booster in sub-millisecond time. Does not require a request body.

- **Header:** `X-Admin-Key: <secret>`
- **Request Body:** None
- **Response (200 OK):**
```json
{
  "message": "Model rolled back successfully",
  "status": "loaded_synthetic",
  "model_version": "eta_model_candidate",
  "artifact_id": "eta_model_candidate.json",
  "loaded_at": "2026-11-06T15:15:00.000000Z",
  "checksum_sha256": "4f828bb7...",
  "dataset_mode": "synthetic",
  "is_healthy": true,
  "has_previous_model": false,
  "previous_model_version": null
}
```
- **Error Response (409 Conflict):** Returned if no previous model exists in memory for rollback.

---

### 4.5 Reload Current Model
#### `POST /api/v1/eta/admin/reload-model`
Forces re-reading, SHA-256 validation, and warm reloading of the configured model artifact from disk or S3. Does not require a request body.

- **Header:** `X-Admin-Key: <secret>`
- **Request Body:** None
- **Response (200 OK):**
```json
{
  "message": "Model reloaded successfully",
  "status": "loaded_synthetic",
  "model_version": "eta_model_candidate",
  "artifact_id": "eta_model_candidate.json",
  "loaded_at": "2026-11-06T15:20:00.000000Z",
  "checksum_sha256": "4f828bb7...",
  "dataset_mode": "synthetic",
  "is_healthy": true,
  "has_previous_model": false,
  "previous_model_version": null
}
```

---

## 5. Model Monitoring & Retraining APIs (SCRUM-1229)

Endpoints for inspecting feature drift, outcome accuracy, running on-demand monitoring cycles, and initiating controlled candidate retraining. All endpoints require `X-Admin-Key`.

### 5.1 Get Monitoring Health & Status
#### `GET /api/v1/eta/admin/monitoring/status`
Retrieves system monitoring health, active model version, concurrency training lock status, latest evaluation results (correlation counts, exclusions, drift per canonical feature, and MAE/RMSE), and latest retraining history.

- **Header:** `X-Admin-Key: <secret>`
- **Response (200 OK):**
```json
{
  "status": "operational",
  "active_model_version": "eta_model_candidate",
  "is_retraining_locked": false,
  "latest_evaluation": {
    "evaluation_id": "eval-8419f2a0-4a81-4321-9988-123456789abc",
    "evaluated_at": "2026-11-06T14:30:00.000000Z",
    "lookback_hours": 24,
    "active_model_version": "eta_model_candidate",
    "correlation_summary": {
      "total_predictions": 850,
      "total_completed_outcomes": 820,
      "correlated_count": 795,
      "uncorrelated_predictions_count": 55,
      "uncorrelated_outcomes_count": 25
    },
    "data_quality_report": {
      "total_evaluated": 795,
      "valid_count": 790,
      "excluded_count": 5,
      "is_sufficient_for_evaluation": true,
      "rejections": {
        "missing_features": 0,
        "non_finite_features": 0,
        "invalid_target_wait": 3,
        "invalid_predicted_wait": 0,
        "duplicate_queue_entry": 2,
        "invalid_timestamps": 0
      }
    },
    "drift_report": {
      "baseline_version": "baseline_v1",
      "sample_count": 790,
      "has_significant_drift": false,
      "drifted_features_count": 0,
      "feature_results": {
        "party_size": { "psi_score": 0.025, "drift_level": "stable", "is_drifted": false },
        "total_people_ahead": { "psi_score": 0.048, "drift_level": "stable", "is_drifted": false },
        "hour_of_day": { "psi_score": 0.031, "drift_level": "stable", "is_drifted": false },
        "day_of_week": { "psi_score": 0.018, "drift_level": "stable", "is_drifted": false },
        "is_weekend": { "psi_score": 0.011, "drift_level": "stable", "is_drifted": false },
        "historical_average_wait": { "psi_score": 0.062, "drift_level": "stable", "is_drifted": false }
      }
    },
    "performance_report": {
      "total_evaluated_samples": 790,
      "overall_mae_minutes": 2.68,
      "overall_rmse_minutes": 3.82,
      "has_degradation": false,
      "version_metrics": {
        "eta_model_candidate": {
          "model_version": "eta_model_candidate",
          "sample_count": 790,
          "mae_minutes": 2.68,
          "rmse_minutes": 3.82,
          "is_degraded": false
        }
      }
    }
  },
  "latest_retraining": null
}
```

---

### 5.2 Run On-Demand Monitoring Evaluation
#### `POST /api/v1/eta/admin/monitoring/evaluate`
Triggers an immediate evaluation cycle across a historical window. Performs correlation, data quality checks, PSI drift evaluation, MAE/RMSE calculations, updates Prometheus metrics, evaluates retraining triggers, and returns the full report.

- **Header:** `X-Admin-Key: <secret>`
- **Request Body (`EvaluateRequest`):**

| Field | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `lookback_hours` | Integer | `24` | $1 \le x \le 720$ | Hours of historical records to correlate and evaluate. |
| `min_samples` | Integer | `null` | $5 \le x \le 10000$ | Override minimum sample threshold (defaults to 30). |
| `restaurant_id` | String | `null` | Max 128 chars | Optional filter to evaluate a specific restaurant. |
| `auto_trigger` | Boolean | `true` | Boolean | Whether to dispatch candidate retraining if drift or degradation is confirmed. |
| `dataset_mode` | String | `"synthetic"` | `"synthetic"`, `"production"` | Dataset mode for candidate retraining. |

#### Example Request Body
```json
{
  "lookback_hours": 48,
  "min_samples": 50,
  "restaurant_id": null,
  "auto_trigger": true,
  "dataset_mode": "synthetic"
}
```

- **Response (200 OK):** Returns full `MonitoringCycleReport` JSON object.

---

### 5.3 Request Manual Candidate Retraining
#### `POST /api/v1/eta/admin/monitoring/trigger-retrain`
Manually initiates candidate model retraining through the SCRUM-1197 pipeline.

> ⚠️ **CRITICAL SAFETY BOUNDARY (AC6):**  
> Retraining generates a validated candidate artifact in `app/infrastructure/ml_models/artifacts/candidates/` and registers it in S3. It **NEVER** automatically promotes or activates the candidate. The active serving model remains unaffected until an operator explicitly calls `POST /api/v1/eta/admin/promote-candidate`.

- **Header:** `X-Admin-Key: <secret>`
- **Request Body (`ManualRetrainRequest`):**

| Field | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `reason` | String | Required | $3 \le \text{len} \le 500$ | Audit reason for requesting manual retraining. |
| `approver_id` | String | `"admin_operator"` | $1 \le \text{len} \le 128$ | Identifier of authorizing operator. |
| `dataset_mode` | String | `"synthetic"` | `"synthetic"`, `"production"` | Dataset source for candidate generation. |

#### Example Request Body
```json
{
  "reason": "Quarterly model retraining ahead of festival holiday peak",
  "approver_id": "ml_engineer_jeswin",
  "dataset_mode": "synthetic"
}
```

#### Response (200 OK)
```json
{
  "decision": {
    "trigger_id": "trig_4a1b8c9d0e1f",
    "should_retrain": true,
    "trigger_type": "manual",
    "status": "accepted",
    "reason": "Manual retraining requested by operator ml_engineer_jeswin: Quarterly model retraining ahead of festival holiday peak",
    "evaluated_at": "2026-11-06T15:30:00.000000Z"
  },
  "execution": {
    "trigger_id": "trig_4a1b8c9d0e1f",
    "status": "completed",
    "candidate_artifact_id": "eta_candidate_20261106_153010.json",
    "candidate_path": "app/infrastructure/ml_models/artifacts/candidates/eta_candidate_20261106_153010.json",
    "is_eligible_for_promotion": true,
    "completed_at": "2026-11-06T15:30:25.000000Z"
  }
}
```

---

## 6. HTTP Status Code Summary

| Status Code | Reason | Usage in Service |
| :--- | :--- | :--- |
| **200 OK** | Success | Request succeeded and result payload returned. |
| **400 Bad Request** | Validation Failure | Invalid query parameters or JSON body fields. |
| **401 Unauthorized** | Authentication Failure | Missing or incorrect `X-Admin-Key` header on administrative endpoints. |
| **404 Not Found** | Resource Missing | Candidate model version not found in storage. |
| **409 Conflict** | State Conflict | Retraining job already actively executing, or no previous model available for rollback. |
| **422 Unprocessable Entity** | Integrity / Schema Rejection | SHA-256 mismatch, feature mismatch, or synthetic model rejected when disallowed. |
| **503 Service Unavailable** | Dependency Offline | XGBoost model not loaded, or MongoDB/Redis disconnected. |

---

## 7. cURL Quick Reference Examples

### Real-Time ETA Prediction
```bash
curl -X GET "http://localhost:8000/api/v1/eta/predict?restaurant_id=rest_101&total_people_ahead=4&party_size=2&queue_entry_id=entry_9988"
```

### Inspect Active Model Status
```bash
curl -X GET "http://localhost:8000/api/v1/eta/admin/model-status" \
  -H "X-Admin-Key: your_admin_secret_key"
```

### Run On-Demand Monitoring Cycle (48-hour lookback)
```bash
curl -X POST "http://localhost:8000/api/v1/eta/admin/monitoring/evaluate" \
  -H "X-Admin-Key: your_admin_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"lookback_hours": 48, "min_samples": 30, "auto_trigger": true}'
```

### Manually Trigger Candidate Retraining
```bash
curl -X POST "http://localhost:8000/api/v1/eta/admin/monitoring/trigger-retrain" \
  -H "X-Admin-Key: your_admin_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual retraining verification", "approver_id": "operator_1"}'
```

### Promote Candidate Model
```bash
curl -X POST "http://localhost:8000/api/v1/eta/admin/promote-candidate" \
  -H "X-Admin-Key: your_admin_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"model_version": "eta_candidate_20261106_153010", "approver_id": "operator_1", "notes": "Approved"}'
```

### Rollback Model
```bash
curl -X POST "http://localhost:8000/api/v1/eta/admin/rollback-model" \
  -H "X-Admin-Key: your_admin_secret_key"
```
