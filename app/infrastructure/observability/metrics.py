"""Custom Prometheus metrics and health gauges."""

from prometheus_client import Gauge, Histogram

REDIS_LOOKUP_LATENCY = Histogram(
    "spotq_redis_feature_lookup_latency_seconds",
    "Latency of Redis feature store lookups in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

MONGO_WRITE_LATENCY = Histogram(
    "spotq_mongodb_write_latency_seconds",
    "Latency of MongoDB queue buffer write operations in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

REDIS_HEALTH_GAUGE = Gauge(
    "spotq_redis_connection_status",
    "1 if Redis connection is healthy, 0 otherwise",
)

MONGO_HEALTH_GAUGE = Gauge(
    "spotq_mongodb_connection_status",
    "1 if MongoDB connection is healthy, 0 otherwise",
)

MODEL_LOADED_GAUGE = Gauge(
    "spotq_xgboost_model_loaded_status",
    "1 if XGBoost model artifact is loaded, 0 otherwise",
)