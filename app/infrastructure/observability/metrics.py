"""Custom Prometheus metrics and health gauges."""

from prometheus_client import Counter, Gauge, Histogram, REGISTRY


def _get_or_create_gauge(name: str, documentation: str) -> Gauge:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, documentation)


def _get_or_create_histogram(name: str, documentation: str, buckets: tuple) -> Histogram:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, buckets=buckets)


REDIS_LOOKUP_LATENCY = _get_or_create_histogram(
    "spotq_redis_feature_lookup_latency_seconds",
    "Latency of Redis feature store lookups in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

MONGO_WRITE_LATENCY = _get_or_create_histogram(
    "spotq_mongodb_write_latency_seconds",
    "Latency of MongoDB queue buffer write operations in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

REDIS_HEALTH_GAUGE = _get_or_create_gauge(
    "spotq_redis_connection_status",
    "1 if Redis connection is healthy, 0 otherwise",
)

MONGO_HEALTH_GAUGE = _get_or_create_gauge(
    "spotq_mongodb_connection_status",
    "1 if MongoDB connection is healthy, 0 otherwise",
)

MODEL_LOADED_GAUGE = _get_or_create_gauge(
    "spotq_xgboost_model_loaded_status",
    "1 if XGBoost model artifact is loaded, 0 otherwise",
)


def _get_or_create_counter(name: str, documentation: str) -> Counter:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation)


ETA_REQUEST_LATENCY = _get_or_create_histogram(
    "spotq_eta_request_latency_seconds",
    "End-to-end latency of ETA prediction requests in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

ETA_INFERENCE_LATENCY = _get_or_create_histogram(
    "spotq_eta_model_inference_latency_seconds",
    "XGBoost model inference latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

ETA_FALLBACK_TOTAL = _get_or_create_counter(
    "spotq_eta_fallback_total",
    "ETA predictions that used fallback history rather than exact history",
)

ETA_CLAMPED_TOTAL = _get_or_create_counter(
    "spotq_eta_clamped_total",
    "ETA predictions clamped to configured minimum or maximum bounds",
)



# Model lifecycle metrics deliberately avoid model-version labels.
MODEL_LOAD_SUCCESS_TOTAL = _get_or_create_counter(
    "spotq_model_load_success_total",
    "Successful ETA model loads",
)
MODEL_LOAD_FAILURE_TOTAL = _get_or_create_counter(
    "spotq_model_load_failure_total",
    "Failed ETA model load attempts",
)
MODEL_RELOAD_SUCCESS_TOTAL = _get_or_create_counter(
    "spotq_model_reload_success_total",
    "Successful ETA model reloads",
)
MODEL_RELOAD_FAILURE_TOTAL = _get_or_create_counter(
    "spotq_model_reload_failure_total",
    "Failed ETA model reloads",
)
MODEL_RELOAD_DURATION = _get_or_create_histogram(
    "spotq_model_reload_duration_seconds",
    "Duration of ETA model reload attempts",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_LOAD_DURATION = _get_or_create_histogram(
    "spotq_model_load_duration_seconds",
    "Duration of initial ETA model load attempts",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_PROMOTION_SUCCESS_TOTAL = _get_or_create_counter(
    "spotq_model_promotion_success_total",
    "Total successful authorized candidate model promotions",
)

MODEL_PROMOTION_FAILURE_TOTAL = _get_or_create_counter(
    "spotq_model_promotion_failure_total",
    "Total failed candidate model promotion attempts",
)

MODEL_ROLLBACK_SUCCESS_TOTAL = _get_or_create_counter(
    "spotq_model_rollback_success_total",
    "Total successful model rollbacks to previous version",
)

MODEL_ROLLBACK_FAILURE_TOTAL = _get_or_create_counter(
    "spotq_model_rollback_failure_total",
    "Total failed model rollback attempts",
)

ETA_INFERENCE_FAILURE_TOTAL = _get_or_create_counter(
    "spotq_eta_inference_failure_total",
    "ETA prediction failures during model inference",
)

# RabbitMQ Event Integration Metrics (SCRUM-1257)
RABBITMQ_HEALTH_GAUGE = _get_or_create_gauge(
    "spotq_rabbitmq_health_status",
    "RabbitMQ consumer connection health (1 = connected, 0 = disconnected)",
)

RABBITMQ_MESSAGES_RECEIVED_TOTAL = _get_or_create_counter(
    "spotq_rabbitmq_messages_received_total",
    "Total RabbitMQ queue event messages received by consumer",
)

RABBITMQ_MESSAGES_PROCESSED_TOTAL = _get_or_create_counter(
    "spotq_rabbitmq_messages_processed_total",
    "Total RabbitMQ messages successfully validated and persisted",
)

RABBITMQ_MESSAGES_FAILED_TOTAL = _get_or_create_counter(
    "spotq_rabbitmq_messages_failed_total",
    "Total RabbitMQ messages that failed processing due to transient errors",
)

RABBITMQ_MESSAGES_REJECTED_TOTAL = _get_or_create_counter(
    "spotq_rabbitmq_messages_rejected_total",
    "Total RabbitMQ messages permanently rejected and routed to DLQ",
)

RABBITMQ_MESSAGE_PROCESSING_DURATION = _get_or_create_histogram(
    "spotq_rabbitmq_message_processing_duration_seconds",
    "Duration of RabbitMQ message validation and persistence in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)


def _get_or_create_counter_with_labels(name: str, documentation: str, labelnames: list[str]) -> Counter:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames=labelnames)


def _get_or_create_gauge_with_labels(name: str, documentation: str, labelnames: list[str]) -> Gauge:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, documentation, labelnames=labelnames)


# SCRUM-1229 Model Monitoring & Retraining Metrics
MONITORING_EVALUATIONS_TOTAL = _get_or_create_counter_with_labels(
    "spotq_monitoring_evaluations_total",
    "Total monitoring evaluations performed",
    labelnames=["status"],
)

MONITORING_MAE_GAUGE = _get_or_create_gauge_with_labels(
    "spotq_monitoring_mae_minutes",
    "Mean Absolute Error in minutes for model version",
    labelnames=["model_version"],
)

MONITORING_RMSE_GAUGE = _get_or_create_gauge_with_labels(
    "spotq_monitoring_rmse_minutes",
    "Root Mean Squared Error in minutes for model version",
    labelnames=["model_version"],
)

MONITORING_FEATURE_DRIFT_PSI = _get_or_create_gauge_with_labels(
    "spotq_monitoring_feature_drift_psi",
    "Population Stability Index drift per feature",
    labelnames=["feature_name"],
)

MONITORING_RETRAINING_TRIGGERS_TOTAL = _get_or_create_counter_with_labels(
    "spotq_monitoring_retraining_triggers_total",
    "Total retraining requests evaluated",
    labelnames=["trigger_type", "status"],
)

