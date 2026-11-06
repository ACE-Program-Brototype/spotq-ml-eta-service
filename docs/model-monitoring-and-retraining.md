# SpotQ ETA Model Monitoring, Drift Detection & Retraining Triggers (SCRUM-1229)

This document provides complete architectural and API documentation for the production ETA monitoring, feature drift detection, and automated retraining trigger pipeline implemented in `spotq-ml-eta-service`.

---

## 1. Architecture and Workflow Overview

The monitoring layer bridges real-time inference, ground-truth queue completions from RabbitMQ, data validation, statistical drift detection, performance monitoring, and model retraining.

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. Real-Time Prediction Path (Sub-50ms)                                │
 │    GET /api/v1/eta/predict?queue_entry_id=...                          │
 │    - Generates prediction_id                                           │
 │    - Evaluates XGBoost model                                           │
 │    - Returns ETA response immediately to client                        │
 │    - Dispatches non-blocking background task to MongoDB:               │
 │      db.prediction_logs.insert_one({ features, timestamp, version, ...})│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────┴────────────────────────────────────┐
 │ 2. Ground-Truth Event Completion (RabbitMQ Consumer)                   │
 │    Consumes spotq.events / ml-eta-queue completed events               │
 │    - Persists actual wait times into db.queue_events                   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. Production Prediction-to-Outcome Correlation (AC1)                  │
 │    app/application/monitoring/correlation.py                           │
 │    - Matches prediction_logs with queue_events via exact queue_entry_id│
 │    - Heuristic temporal fallback (±10 min window) for browsing queries │
 │    - Preserves prediction-time canonical features (zero leakage)       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 4. Data Quality & Exclusion Guard (AC2)                                │
 │    app/application/monitoring/data_quality.py                          │
 │    - Validates feature completeness & finite numerical values          │
 │    - Enforces target wait bounds (0, 120] minutes                      │
 │    - Deduplicates redundant completion events                          │
 │    - Enforces minimum sample threshold (N >= 30)                       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
 ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
 │ 5. Feature Drift Detection (AC3)│   │ 6. Outcome Performance (AC4)   │
 │    drift_detector.py            │   │    performance.py               │
 │    - Population Stability Index │   │    - Rolling MAE & RMSE         │
 │      (PSI) against baseline     │   │    - Strictly attributed to     │
 │    - Monitored: 6 canonical     │   │      generating model_version   │
 │      features with Laplace      │   │    - Evaluates against threshold│
 │      smoothing                  │   │      (MAE > 5m, RMSE > 8m)      │
 │    - Flags PSI >= 0.25 (Drift)  │   │    - Flags degradation          │
 └────────────────┬────────────────┘   └────────────────┬────────────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 7. Configurable Retraining Trigger Manager (AC5, AC6)                  │
 │    trigger_manager.py                                                  │
 │    - Evaluates triggers: Performance degradation, Drift, or Manual     │
 │    - Concurrency Lock: Prevents overlapping training executions        │
 │    - Cooldown Enforcer: Configurable 4-hour cooldown                   │
 │    - Invokes SCRUM-1197 training pipeline                              │
 │                                                                        │
 │  ⚠️ CRITICAL SAFETY BOUNDARY (AC6):                                    │
 │    Retraining produces CANDIDATE model artifacts ONLY.                 │
 │    Monitoring NEVER automatically promotes or activates a candidate.   │
 │    Promotion strictly requires operator authorization via              │
 │    POST /api/v1/eta/admin/promote-candidate (SCRUM-1228).              │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 8. Observability & Monitoring APIs (AC7, AC8)                          │
 │    - Prometheus gauges: PSI drift, MAE, RMSE, Trigger counters         │
 │    - Admin APIs: GET status, POST evaluate, POST trigger-retrain       │
 │    - CLI runner: python scripts/run_model_monitoring.py                │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Canonical Features & Drift Detection (AC3)

Drift monitoring tracks the 6 canonical features against the certified baseline stored in `config/feature_drift_baseline.json`:

| Feature Name | Type | Range / Categories | Baseline Source |
| :--- | :--- | :--- | :--- |
| `party_size` | Numerical | 1 to 20 people | Documented training distribution |
| `total_people_ahead` | Numerical | 0 to 100 people | Documented training distribution |
| `hour_of_day` | Categorical | 0 to 23 hours | Hourly distribution |
| `day_of_week` | Categorical | 0 (Monday) to 6 (Sunday) | Daily distribution |
| `is_weekend` | Categorical | 0 (Weekday) or 1 (Weekend) | Binary proportion |
| `historical_average_wait` | Numerical | 0 to 120 minutes | Historical feature distribution |

### Statistical Method: Population Stability Index (PSI)
Population Stability Index is calculated across all bins:
$$\text{PSI} = \sum_{k=1}^{K} (A_k - E_k) \times \ln\left(\frac{A_k}{E_k}\right)$$
Where:
- $A_k$ is the actual observed proportion in bin $k$ (with Laplace smoothing $\alpha = 0.5$).
- $E_k$ is the expected baseline proportion in bin $k$.

### Classification Bands:
- **$\text{PSI} < 0.10$**: Stable (no significant change).
- **$0.10 \le \text{PSI} < 0.25$**: Moderate Drift (slight covariate shift, monitor closely).
- **$\text{PSI} \ge 0.25$**: Significant Drift (substantial shift; triggers retraining evaluation).

---

## 3. Retraining Trigger Conditions & Safety Boundary (AC5, AC6)

### Trigger Conditions:
1. **Model Performance Degradation:** Observed rolling MAE exceeds `MONITORING_MAX_ACCEPTABLE_MAE_MINUTES` (default: 5.0m) or RMSE exceeds `MONITORING_MAX_ACCEPTABLE_RMSE_MINUTES` (default: 8.0m) with $N \ge N_{min}$.
2. **Significant Feature Drift:** One or more canonical features exhibit $\text{PSI} \ge 0.25$ with $N \ge N_{min}$.
3. **Manual Operator Request:** Authorized operator requests candidate retraining via API or CLI.

### Guards:
- **Sample Sufficiency Guard:** $N \ge 30$ valid samples required; small batches report `insufficient_samples` and will not trigger retraining.
- **Concurrency Lock:** `asyncio.Lock` ensures only one retraining pipeline executes at a time.
- **Cooldown Window:** `RETRAINING_COOLDOWN_HOURS` (default: 4.0 hours) prevents retraining loops.
- **Kill-Switch Toggle:** `ENABLE_AUTOMATIC_RETRAINING_TRIGGERS=false` disables automated triggers while permitting manual operator requests.

### 🛡️ Strict Lifecycle Safety Boundary (AC6)
Retraining triggered by the monitoring layer runs the standard SCRUM-1197 training pipeline and generates a new candidate artifact in `app/infrastructure/ml_models/artifacts/candidates/` and the S3 registry. **The active production model is never modified or hot-swapped by the monitoring process.** Promotion to production strictly requires explicit human operator authorization via `POST /api/v1/eta/admin/promote-candidate` protected by `X-Admin-Key`.

---

## 4. Full API Specification

All administrative endpoints are authenticated via the `X-Admin-Key` HTTP header.

### 4.1 Inference API

#### `GET /api/v1/eta/predict`
Calculates estimated waiting time in minutes. Asynchronously logs prediction features for outcome correlation without adding latency.

- **Query Parameters:**
  - `party_size` (int, 1-20, required): Number of guests.
  - `restaurant_id` (str, required): Restaurant identifier.
  - `total_people_ahead` (int, >= 0, optional): Count of people currently in queue ahead.
  - `queue_entry_id` (str, optional): Unique queue ticket ID to enable exact correlation when the customer is seated.

- **Success Response (200 OK):**
```json
{
  "predicted_wait_time_minutes": 18.5,
  "confidence_interval": {
    "lower_bound_minutes": 15.0,
    "upper_bound_minutes": 22.0
  },
  "party_size": 2,
  "people_ahead": 4,
  "restaurant_id": "rest_101",
  "historical_average_applied": 17.5,
  "is_fallback": false,
  "is_clamped": false,
  "prediction_id": "pred_9b83f06ac89e",
  "model_version": "eta_model_candidate"
}
```

---

### 4.2 Model Administration APIs (SCRUM-1228)

#### `GET /api/v1/eta/admin/model-status`
Inspect currently loaded active model, loaded checksum, and storage backend.
- **Header:** `X-Admin-Key: <secret>`

#### `GET /api/v1/eta/admin/candidates`
List all registered candidate models in storage eligible for promotion.
- **Header:** `X-Admin-Key: <secret>`

#### `POST /api/v1/eta/admin/promote-candidate`
Promote and atomically hot-swap an authorized candidate model into active serving.
- **Header:** `X-Admin-Key: <secret>`
- **Request Body:**
```json
{
  "model_version": "eta_candidate_20261105_120000",
  "approver_id": "ml_engineer_jeswin",
  "notes": "Retrained model beating baseline MAE"
}
```

#### `POST /api/v1/eta/admin/rollback-model`
Instantly roll back to the previously active model version.
- **Header:** `X-Admin-Key: <secret>`
- **Request Body:**
```json
{
  "approver_id": "ml_engineer_jeswin",
  "reason": "Observed elevated tail latencies"
}
```

---

### 4.3 Monitoring & Retraining APIs (SCRUM-1229)

#### `GET /api/v1/eta/admin/monitoring/status`
Retrieve current monitoring health, latest evaluation results, feature drift summary, and retraining status.
- **Header:** `X-Admin-Key: <secret>`
- **Success Response (200 OK):**
```json
{
  "status": "operational",
  "active_model_version": "eta_model_candidate",
  "is_retraining_locked": false,
  "latest_evaluation": {
    "evaluation_id": "eval-d3f18a24-789a",
    "evaluated_at": "2026-11-06T00:30:00Z",
    "lookback_hours": 24,
    "active_model_version": "eta_model_candidate",
    "correlation_summary": {
      "total_predictions": 1420,
      "total_completed_outcomes": 1390,
      "correlated_count": 1350,
      "uncorrelated_predictions_count": 70,
      "uncorrelated_outcomes_count": 40
    },
    "data_quality_report": {
      "total_evaluated": 1350,
      "valid_count": 1342,
      "excluded_count": 8,
      "is_sufficient_for_evaluation": true
    },
    "drift_report": {
      "has_significant_drift": false,
      "drifted_features_count": 0,
      "feature_results": {
        "party_size": { "psi_score": 0.024, "drift_level": "stable", "is_drifted": false },
        "total_people_ahead": { "psi_score": 0.041, "drift_level": "stable", "is_drifted": false },
        "hour_of_day": { "psi_score": 0.035, "drift_level": "stable", "is_drifted": false },
        "day_of_week": { "psi_score": 0.012, "drift_level": "stable", "is_drifted": false },
        "is_weekend": { "psi_score": 0.008, "drift_level": "stable", "is_drifted": false },
        "historical_average_wait": { "psi_score": 0.052, "drift_level": "stable", "is_drifted": false }
      }
    },
    "performance_report": {
      "overall_mae_minutes": 2.85,
      "overall_rmse_minutes": 4.12,
      "has_degradation": false
    }
  },
  "latest_retraining": {
    "trigger_id": "trig_a1b2c3d4",
    "trigger_type": "manual",
    "status": "completed",
    "completed_at": "2026-11-05T18:00:00Z"
  }
}
```

---

#### `POST /api/v1/eta/admin/monitoring/evaluate`
Trigger an on-demand evaluation cycle over a configurable lookback window.
- **Header:** `X-Admin-Key: <secret>`
- **Request Body:**
```json
{
  "lookback_hours": 24,
  "min_samples": 30,
  "restaurant_id": null,
  "auto_trigger": true,
  "dataset_mode": "synthetic"
}
```
- **Field Descriptions:**
  - `lookback_hours` (int, 1-720, default: 24): Hours of historical predictions and completed queue events to evaluate.
  - `min_samples` (int, 5-10000, optional): Override minimum valid samples required for statistical significance.
  - `restaurant_id` (str, optional): Optional restaurant identifier filter.
  - `auto_trigger` (bool, default: true): If true and degradation/drift is confirmed, automatically dispatches candidate retraining.
  - `dataset_mode` (str, default: "synthetic"): Dataset source for candidate generation ("synthetic" or "production").

- **Success Response (200 OK):** Returns full `MonitoringCycleReport`.

---

#### `POST /api/v1/eta/admin/monitoring/trigger-retrain`
Manually trigger candidate model retraining.
- **Header:** `X-Admin-Key: <secret>`
- **Request Body:**
```json
{
  "reason": "Operator observed upcoming seasonal shifts",
  "approver_id": "operator_jeswin",
  "dataset_mode": "synthetic"
}
```
- **Success Response (200 OK):**
```json
{
  "decision": {
    "trigger_id": "trig_manual_847291",
    "should_retrain": true,
    "trigger_type": "manual",
    "status": "accepted",
    "reason": "Manual retraining requested by operator operator_jeswin: Operator observed upcoming seasonal shifts",
    "evaluated_at": "2026-11-06T01:10:00Z"
  },
  "execution": {
    "trigger_id": "trig_manual_847291",
    "status": "completed",
    "candidate_artifact_id": "eta_candidate_20261106_011015.json",
    "candidate_path": "app/infrastructure/ml_models/artifacts/candidates/eta_candidate_20261106_011015.json",
    "is_eligible_for_promotion": true,
    "completed_at": "2026-11-06T01:10:25Z"
  }
}
```

---

## 5. Prometheus Metrics Reference

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `spotq_monitoring_evaluations_total` | Counter | `status` | Total monitoring evaluations executed |
| `spotq_monitoring_mae_minutes` | Gauge | `model_version` | Mean Absolute Error in minutes for evaluated model version |
| `spotq_monitoring_rmse_minutes` | Gauge | `model_version` | Root Mean Squared Error in minutes for evaluated model version |
| `spotq_monitoring_feature_drift_psi` | Gauge | `feature_name` | Current Population Stability Index per feature |
| `spotq_monitoring_retraining_triggers_total` | Counter | `trigger_type`, `status` | Retraining evaluation outcomes (`accepted`, `rejected_cooldown`, etc.) |

---

## 6. CLI Execution

For offline cron jobs or manual terminal inspection, run the standalone monitoring script:

```bash
# Basic run with default 24h lookback
uv run python scripts/run_model_monitoring.py

# Custom lookback window and evaluation without auto-triggering retraining
uv run python scripts/run_model_monitoring.py --lookback-hours 48 --no-auto-trigger

# Target specific restaurant
uv run python scripts/run_model_monitoring.py --restaurant-id rest_101
```

