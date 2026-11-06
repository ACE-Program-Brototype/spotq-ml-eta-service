# SpotQ ETA Microservice - Complete API Documentation

Comprehensive API reference for the **SpotQ ETA Microservice** (`spotq-ml-eta-service`), covering real-time wait-time prediction, health checks, Prometheus observability, authorized model lifecycle management (SCRUM-1228), and automated model monitoring and retraining triggers (SCRUM-1229).

---

## 1. General Information & Conventions

### Base URLs
- **Local Development:** `http://127.0.0.1:8000`
- **Docker Container:** `http://localhost:8000`
- **Kubernetes / Production:** `http://<service-host>:<port>`

### Interactive API Explorers
- **Swagger UI:** `/docs`
- **ReDoc UI:** `/redoc`
- **OpenAPI Schema (JSON):** `/openapi.json`

### Content Type
All POST/PUT endpoints consume and produce `application/json; charset=utf-8`.

### Authentication
- **Public Endpoints:** Health checks, Prometheus metrics, and the real-time prediction endpoint do not require credentials.
- **Administrative Endpoints:** All endpoints under `/api/v1/eta/admin/*` require the `X-Admin-Key` HTTP header configured via `ETA_MODEL_ADMIN_KEY`. Constant-time comparison is used to prevent timing attacks.

```http
X-Admin-Key: <your-configured-admin-key>
```

### Standard Error Response Schema
Whenever request validation fails (`400 Bad Request`), the service returns:

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

When an administrative request lacks authentication or supplies an invalid key (`401 Unauthorized`):

```json
{
  "detail": "Invalid or missing admin key"
}
```

---

## 2. System Health & Observability Endpoints

### 2.1 Root Status
#### `GET /`
Returns basic service identity and operating environment.

- **Auth:** None (Public)
- **Response Headers:** `Content-Type: application/json`
- **Success Response (200 OK):**
```json
{
  "service": "spotq-eta-service",
  "status": "running",
  "environment": "development"
}
```

---

### 2.2 Basic Liveness Probe
#### `GET /healthz`
Lightweight probe intended for container orchestrators (e.g., Kubernetes liveness probes).

- **Auth:** None (Public)
- **Success Response (200 OK):**
```json
{
  "status": "healthy"
}
```

---

### 2.3 Comprehensive Readiness Probe
#### `GET /health/ready`
Deep health inspection verifying upstream infrastructure dependencies: MongoDB connection, Redis feature store connection, loaded XGBoost model artifact, and RabbitMQ event consumer.

- **Auth:** None (Public)
- **Success Response (200 OK):**
```json
{
  "status": "ready",
  "timestamp": "2026-11-06T15:00:00.000000Z",
  "checks": {
    "redis": {
      "status": "healthy",
      "host": "localhost",
      "port": 6379
    },
    "mongodb": {
      "status": "healthy",
      "database": "spotq_production"
    },
    "model": {
      "status": "loaded",
      "model_version": "eta_model_candidate",
      "artifact_id": "eta_model_candidate.json",
      "dataset_mode": "synthetic"
    },
    "rabbitmq": {
      "status": "connected",
      "queue": "ml-eta-queue"
    }
  }
}
```
- **Error Response (503 Service Unavailable):** Returned if any critical dependency is offline.

---

### 2.4 Process Liveness Probe
#### `GET /health/live`
Confirms the ASGI event loop and application process are responsive.

- **Auth:** None (Public)
- **Success Response (200 OK):**
```json
{
  "status": "live",
  "uptime_seconds": 3600.5
}
```

---

### 2.5 Prometheus Metrics Scrape
#### `GET /metrics`
Exposes system, HTTP latency, and custom machine learning metrics in standard Prometheus text format.

- **Auth:** None (Public)
- **Format:** `text/plain; version=0.0.4`
- **Key Custom Metrics Exported:**
  - `spotq_eta_request_latency_seconds_bucket`: Prediction endpoint end-to-end latency histogram.
  - `spotq_eta_model_inference_latency_seconds_bucket`: XGBoost pure inference latency histogram.
  - `spotq_redis_feature_lookup_latency_seconds_bucket`: Redis lookup duration.
  - `spotq_rabbitmq_messages_processed_total`: Count of consumed completed queue events.
  - `spotq_monitoring_evaluations_total{status="success"}`: Counter for monitoring cycles.
  - `spotq_monitoring_mae_minutes{model_version="..."}`: Rolling MAE gauge per model version.
  - `spotq_monitoring_rmse_minutes{model_version="..."}`: Rolling RMSE gauge per model version.
  - `spotq_monitoring_feature_drift_psi{feature_name="..."}`: Population Stability Index drift gauge.
  - `spotq_monitoring_retraining_triggers_total{trigger_type="...", status="..."}`: Retraining decisions counter.

---

## 3. Real-Time ETA Prediction API

### 3.1 Predict Waiting Time
#### `GET /api/v1/eta/predict`

Calculates estimated waiting time in minutes using the active XGBoost model, Redis feature store hierarchy, and chronological feature derivation. Asynchronously buffers prediction features to MongoDB `prediction_logs` for zero-latency monitoring and ground truth outcome correlation.

- **Auth:** None (Public)
- **Target Latency:** Sub-50ms (p95)

#### Query Parameters

| Parameter | Type | Required | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `party_size` | Integer | **Yes** | $1 \le x \le 20$ | Number of dining guests. |
| `restaurant_id` | String | **Yes** | $1 \le \text{len} \le 128$ | Unique identifier of the restaurant. |
| `total_people_ahead` | Integer | No | $x \ge 0$ (Default: 0) | Count of people ahead in queue. |
| `queue_entry_id` | String | No | $1 \le \text{len} \le 128$ | Correlation ID linking prediction to eventual completed queue event. |

#### Example Request
```http
GET /api/v1/eta/predict?party_size=4&restaurant_id=rest_101&total_people_ahead=8&queue_entry_id=entry_abc123 HTTP/1.1
Host: 127.0.0.1:8000
```

#### Success Response (200 OK)
```json
{
  "predicted_wait_time_minutes": 24.5,
  "confidence_interval": {
    "lower_bound_minutes": 20.0,
    "upper_bound_minutes": 29.0
  },
  "party_size": 4,
  "people_ahead": 8,
  "restaurant_id": "rest_101",
  "historical_average_applied": 22.0,
  "is_fallback": false,
  "is_clamped": false,
  "prediction_id": "pred_c3d4e5f6a7b8",
  "model_version": "eta_model_candidate"
}
```

#### Response Fields
- `predicted_wait_time_minutes` (Float): Final predicted queue duration in minutes.
- `confidence_interval.lower_bound_minutes` (Float): Lower 90% confidence threshold (clamped to 0).
- `confidence_interval.upper_bound_minutes` (Float): Upper 90% confidence threshold.
- `party_size` (Integer): Verified input party size.
- `people_ahead` (Integer): Verified input queue depth.
- `restaurant_id` (String): Target restaurant ID.
- `historical_average_applied` (Float): Historical baseline retrieved from Redis hierarchy or default.
- `is_fallback` (Boolean): True if exact (restaurant + day + hour) bucket was missing in Redis and fallback was used.
- `is_clamped` (Boolean): True if raw model output was constrained by min/max boundaries.
- `prediction_id` (String): Unique identifier recorded in MongoDB `prediction_logs`.
- `model_version` (String): Version tag of the XGBoost booster that generated the inference.

#### Error Responses
- **400 Bad Request:** Missing required parameters or parameters out of allowed bounds.
- **503 Service Unavailable:** No model loaded or model in an unhealthy state.

---

## 4. Model Lifecycle Administration APIs (SCRUM-1228)

Endpoints for inspecting, promoting, rolling back, and reloading machine learning model artifacts. All endpoints require `X-Admin-Key`.

### 4.1 Inspect Active Model Status
#### `GET /api/v1/eta/admin/model-status`
Returns metadata of the model currently active in memory, SHA-256 integrity checksum, loaded timestamp, dataset mode, and storage backend details.

- **Auth:** `X-Admin-Key: <secret>`
- **Success Response (200 OK):**
```json
{
  "active_model": {
    "status": "ready",
    "model_version": "eta_model_candidate",
    "artifact_id": "eta_model_candidate.json",
    "loaded_at": "2026-11-06T12:00:00.000000Z",
    "checksum_sha256": "4f828bb702e86fb7b79d8542c38865d6b4122d4f2481e18d6e3ec1694f6c449c",
    "dataset_mode": "synthetic",
    "is_healthy": true,
    "has_previous_model": true,
    "previous_model_version": "eta_model_v1"
  },
  "storage_mode": "remote_s3",
  "s3_bucket": "spotq-ml-artifacts-prod"
}
```

---

### 4.2 List Registered Candidate Models
#### `GET /api/v1/eta/admin/candidates`
Scans S3 and local storage registries to list registered candidate models, evaluation metrics, and promotion eligibility status.

- **Auth:** `X-Admin-Key: <secret>`
- **Success Response (200 OK):**
```json
{
  "candidates_count": 2,
  "candidates": [
    {
      "model_version": "eta_candidate_20261105_180000",
      "artifact_path": "models/eta_candidate_20261105_180000/eta_model.json",
      "dataset_mode": "production",
      "mae_minutes": 2.45,
      "rmse_minutes": 3.65,
      "is_eligible_for_promotion": true,
      "registered_at": "2026-11-05T18:05:00Z"
    },
    {
      "model_version": "eta_candidate_synthetic_test",
      "artifact_path": "models/eta_candidate_synthetic_test/eta_model.json",
      "dataset_mode": "synthetic",
      "mae_minutes": 3.10,
      "rmse_minutes": 4.50,
      "is_eligible_for_promotion": false,
      "registered_at": "2026-11-04T10:00:00Z"
    }
  ]
}
```

---

### 4.3 Promote Candidate Model
#### `POST /api/v1/eta/admin/promote-candidate`
Authorizes promotion of a registered candidate. Verifies SHA-256 checksum, validates the 6 canonical feature names, executes an atomic in-memory hot-swap with zero downtime, and archives the previous model in memory for instant rollback.

- **Auth:** `X-Admin-Key: <secret>`
- **Request Body:**
```json
{
  "model_version": "eta_candidate_20261105_180000",
  "approver_id": "ml_engineer_jeswin",
  "notes": "Candidate retrained on latest 30-day production data beating baseline MAE"
}
```
- **Success Response (200 OK):**
```json
{
  "status": "promoted",
  "model_version": "eta_candidate_20261105_180000",
  "checksum_sha256": "8a3e7b1c...",
  "promoted_at": "2026-11-06T15:10:00.000000Z",
  "previous_model_version": "eta_model_candidate"
}
```
- **Error Responses:**
  - **401 Unauthorized:** Invalid or missing `X-Admin-Key`.
  - **404 Not Found:** Candidate version not found in S3 or local registry.
  - **422 Unprocessable Entity:** Checksum mismatch, feature schema incompatibility, or synthetic candidate rejected in production environment (`ALLOW_SYNTHETIC_MODEL=false`).

---

### 4.4 Rollback Active Model
#### `POST /api/v1/eta/admin/rollback-model`
Instantly restores the previously loaded model booster in sub-millisecond time without restarting the service or re-reading from disk.

- **Auth:** `X-Admin-Key: <secret>`
- **Request Body:**
```json
{
  "approver_id": "ml_engineer_jeswin",
  "reason": "Elevated tail latency observed on promoted candidate"
}
```
- **Success Response (200 OK):**
```json
{
  "status": "rolled_back",
  "active_model_version": "eta_model_candidate",
  "rolled_back_from": "eta_candidate_20261105_180000",
  "rolled_back_at": "2026-11-06T15:15:00.000000Z"
}
```
- **Error Response (400 Bad Request):** Returned if no previous model exists in memory.

---

### 4.5 Reload Current Model
#### `POST /api/v1/eta/admin/reload-model`
Forces re-reading, SHA-256 validation, and warm reloading of the configured model artifact from disk or S3.

- **Auth:** `X-Admin-Key: <secret>`
- **Success Response (200 OK):**
```json
{
  "status": "reloaded",
  "model_version": "eta_model_candidate",
  "reloaded_at": "2026-11-06T15:20:00.000000Z"
}
```

---

## 5. Model Monitoring & Retraining APIs (SCRUM-1229)

Endpoints for inspecting feature drift, outcome accuracy, running on-demand monitoring cycles, and initiating controlled retraining. All endpoints require `X-Admin-Key`.

### 5.1 Get Monitoring Health & Status
#### `GET /api/v1/eta/admin/monitoring/status`
Retrieves system monitoring health, active model version, concurrency training lock status, latest evaluation results (correlation counts, exclusions, drift per canonical feature, and MAE/RMSE), and latest retraining history.

- **Auth:** `X-Admin-Key: <secret>`
- **Success Response (200 OK):**
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

- **Auth:** `X-Admin-Key: <secret>`
- **Request Body:**

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

- **Success Response (200 OK):** Returns the full `MonitoringCycleReport` JSON object (matching the structure seen in `/status`).

---

### 5.3 Request Manual Candidate Retraining
#### `POST /api/v1/eta/admin/monitoring/trigger-retrain`
Manually initiates candidate model retraining through the SCRUM-1197 pipeline.

> ⚠️ **CRITICAL SAFETY BOUNDARY (AC6):**  
> Retraining generates a validated candidate artifact in `app/infrastructure/ml_models/artifacts/candidates/` and registers it in S3. It **NEVER** automatically promotes or activates the candidate. The active serving model remains unaffected until an operator explicitly calls `POST /api/v1/eta/admin/promote-candidate`.

- **Auth:** `X-Admin-Key: <secret>`
- **Request Body:**

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

#### Success Response (200 OK)
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

## 6. HTTP Status Code Reference

| Status Code | Description | Standard Usage |
| :--- | :--- | :--- |
| **200 OK** | Success | Request succeeded and body contains the result payload. |
| **400 Bad Request** | Invalid Input | Validation failed on query parameters or JSON body fields. |
| **401 Unauthorized** | Missing / Invalid Auth | Missing or incorrect `X-Admin-Key` header on administrative endpoints. |
| **404 Not Found** | Resource Missing | Requested candidate model version does not exist in registry. |
| **409 Conflict** | Concurrency Conflict | Retraining job already actively executing (`is_retraining_locked: true`). |
| **422 Unprocessable Entity** | Business Rule Rejection | Checksum failure, feature schema mismatch, or synthetic candidate rejected in production. |
| **503 Service Unavailable** | Service Not Ready | Model not loaded or critical upstream dependency (MongoDB, Redis) disconnected. |

---

## 7. cURL Quick Reference Examples

### Make an ETA Prediction
```bash
curl -X GET "http://localhost:8000/api/v1/eta/predict?party_size=2&restaurant_id=rest_101&total_people_ahead=4&queue_entry_id=entry_9988"
```

### Inspect Loaded Model Status
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
