# Automated ETA Model Training & Retraining Pipeline

This guide details the offline training, continuous retraining, chronological evaluation, candidate validation, and MLflow experiment tracking pipeline for the SpotQ wait-time prediction engine.

---

## 1. Overview and Objectives

The Automated ETA Model Training & Retraining Pipeline builds validated candidate model artifacts using historical queue data persisted in MongoDB without disrupting or overwriting the active production model.

Key architectural boundaries:
- **Separation of Concerns:** Training and retraining workflows are strictly decoupled from real-time synchronous inference. They are executed as scheduled or triggered jobs.
- **Continuous Ingestion to Retraining Loop:** MERN completed queue events are ingested via the background BullMQ consumer (`ml-eta-queue`) into MongoDB (`queue_events`), supplying fresh historical data for subsequent retraining runs.
- **Candidate Isolation:** Retrained artifacts are written to `app/infrastructure/ml_models/artifacts/candidates/` and never directly overwrite the active production model (`MODEL_PATH`).
- **Serving Contract Parity:** Candidate models use the canonical 6-feature schema, exact feature names, ordering, and types required by the real-time prediction endpoint.
- **Reproducible Tracking:** Every training and retraining run, its dataset provenance, hyperparameters, evaluation metrics, and output artifacts are recorded in MLflow. Incomplete or failed runs are explicitly flagged as `FAILED`.

---

## 2. Canonical Feature Schema

The model contract uses the canonical six-feature schema:

| Feature Name | Type | Description | Parity with Inference API |
| :--- | :--- | :--- | :--- |
| `party_size` | Integer (1–100) | Size of the dining party | Directly provided in request |
| `total_people_ahead` | Integer (0–1000) | Total guests waiting ahead in queue | Directly provided in request |
| `hour_of_day` | Integer (0–23) | Restaurant-local hour of join instant | Resolved via restaurant IANA timezone |
| `day_of_week` | Integer (0–6) | Restaurant-local weekday (0=Mon, 6=Sun) | Resolved via restaurant IANA timezone |
| `is_weekend` | Integer (0 or 1) | 1 if Saturday or Sunday, else 0 | Resolved via restaurant IANA timezone |
| `historical_average_wait` | Float (minutes) | Rolling average for `(restaurant, weekday, hour)` | Redis cache lookup with cold-start fallback |

### Leakage-Safe Feature Engineering
Training incorporates wait times strictly completed *before* each event's join timestamp. Historical averages only include previously completed queues, preventing future-data leakage.

---

## 3. Historical Data Validation

Data cleaning ([`data_cleaning.py`](file:///Users/jeswin07/Documents/Data Science AI ML/Data Science Btype/SpotQ_ML/spotq-ml-eta-service/app/infrastructure/ml_models/data_cleaning.py)) validates all records against strict quality gates before training begins:

- **Required Fields:** `queue_entry_id`, `restaurant_id`, `party_size`, `total_people_ahead_at_join`, `joined_at`, `completed_at`, `actual_wait_time_minutes`.
- **Target Boundaries:** $0 < \text{actual\_wait\_time\_minutes} \le 120$.
- **Timestamp Order:** `completed_at > joined_at` with mandatory timezone awareness.
- **Duration Agreement:** $|(\text{completed\_at} - \text{joined\_at}) - \text{actual\_wait\_time}| \le 0.1\text{ min}$.
- **Deduplication:** Unique on `queue_entry_id`.
- **Configurable Minimum Sample Size:** The job fails safely with `InsufficientDataError` (exit code 2) if valid samples are fewer than `--min-sample-size` (default: 50).

---

## 4. Candidate Eligibility Criteria

Trained models are evaluated on a chronological holdout (with embargo for in-flight queues) and compared against:
1. **Training Median Baseline**
2. **Historical Average Baseline**
3. **Random Forest Benchmark**

A candidate model is marked `is_eligible: true` only if it satisfies all configured gates:
- **MAE Threshold:** Holdout MAE $\le \text{max\_test\_mae}$ (default: 5.0 min).
- **RMSE Threshold:** Holdout RMSE $\le \text{max\_test\_rmse}$ (default: 8.0 min).
- **Baseline Improvement:** Holdout MAE must outperform the historical average baseline.
- **Provenance Safeguard:** Models trained on `synthetic` data are strictly marked `is_eligible: false` with the warning:
  > *"Synthetic data validates pipeline mechanics only; not eligible for production promotion."*

---

## 5. MLflow Experiment Tracking

The workflow records all metadata to the configured MLflow tracking server:
- **Parameters:** `dataset_mode`, `collection`, `min_sample_size`, `test_fraction`, `n_splits`, `seed`, `cold_start_average_minutes`, git revision SHA, and optimal XGBoost hyperparameters.
- **Metrics:** `input_records`, `accepted_records`, `rejected_records`, `cv_best_mae_minutes`, `test_mae_minutes`, `test_rmse_minutes`, `test_r2`, baseline MAEs, and percent improvements.
- **Tags:** `dataset_mode`, `is_synthetic`, `candidate_eligible`, `feature_contract_version`.
- **Artifacts:** `eta_model_candidate.json`, `eta_training_report.json`.
- **Fault Tolerance:** If training or validation raises an exception, the MLflow run is automatically marked as `FAILED`.

---

## 6. Execution Runbook

### Prerequisites
Start the database and MLflow tracking services:
```bash
cd spotq-infra
docker compose up -d mongodb redis mlflow
```

Ensure environment variables are configured (or use defaults):
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MONGO_URI="mongodb://localhost:27017"
export MONGO_COLLECTION_NAME="queue_events"
```

### Running Training / Retraining on Real Queue Data
```bash
cd spotq-ml-eta-service
uv run python -m scripts.train_eta_model \
  --dataset-mode real \
  --collection queue_events \
  --min-sample-size 100 \
  --max-test-mae 4.5 \
  --max-test-rmse 7.5
```

### Running Validation / Retraining on Synthetic Data
For CI pipelines, development testing, and mechanical pipeline verification:
```bash
cd spotq-ml-eta-service
uv run python -m scripts.train_eta_model \
  --dataset-mode synthetic \
  --min-sample-size 50 \
  --n-iter 4
  --max-test-rmse 7.5 \
  --timezone-map config/restaurant_timezones.json
```

### Running Training on Synthetic Data (Development & Verification)
```bash
cd spotq-eta-service
uv run python -m scripts.train_eta_model \
  --dataset-mode synthetic \
  --collection spotq_production \
  --min-sample-size 50 \
  --max-test-mae 5.0 \
  --max-test-rmse 8.0 \
  --timezone-map config/restaurant_timezones.json
```

### Command Arguments Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset-mode` | *(required)* | `real` or `synthetic` |
| `--collection` | `queue_events` | MongoDB collection storing queue outcomes |
| `--min-sample-size` | `50` | Minimum validated records required to proceed |
| `--test-fraction` | `0.20` | Fraction of latest chronological data for test holdout |
| `--n-splits` | `3` | Expanding-window purged time-series CV folds |
| `--n-iter` | `12` | Randomized search iterations for hyperparameter tuning |
| `--max-test-mae` | `5.0` | Maximum acceptable MAE for promotion eligibility |
| `--max-test-rmse` | `8.0` | Maximum acceptable RMSE for promotion eligibility |
| `--allow-worse-than-baseline` | `False` | Flag to waive requirement that candidate beats historical average |
| `--mlflow-tracking-uri` | `http://localhost:5000` | MLflow tracking server endpoint |
| `--mlflow-experiment-name` | `spotq-eta-training` | MLflow experiment name |
| `--output-dir` | `app/infrastructure/ml_models/artifacts/candidates` | Candidate output folder |

---

## 7. Artifacts and Output

Upon successful completion, the script generates:
1. `app/infrastructure/ml_models/artifacts/candidates/eta_model_candidate.json`:
   - Serialized XGBoost booster in standard JSON format.
   - Embeds attributes: `mlflow_run_id`, `feature_contract_version`, `dataset_mode`, `trained_at_utc`.
   - Verified with an automated smoke-prediction before write completion.
2. `app/infrastructure/ml_models/artifacts/candidates/eta_training_report.json`:
   - Full auditable summary including cleaning rejection reasons, split cutoffs, CV scores, baseline comparison, eligibility gate results, and slice diagnostics.

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `InsufficientDataError` (Exit code 2) | Fewer valid records than `--min-sample-size` after cleaning. | Check exclusion reasons in log; seed synthetic data or decrease `--min-sample-size` for testing. |
| `CandidateVerificationError` (Exit code 3) | Serialized candidate failed serving schema validation or smoke test. | Inspect booster feature names and count; verify `FEATURE_NAMES` contract. |
| `ConnectionRefusedError` (MLflow) | MLflow tracking server is not running at configured URI. | Run `docker compose up -d mlflow` or set `--mlflow-tracking-uri file://$(pwd)/mlruns` for local file logging. |
| Active model unchanged after training | Expected behavior; candidates are not auto-promoted. | Promotion is governed by administrative reload (`POST /api/v1/eta/admin/reload-model`) as part of SCRUM-1228. |

