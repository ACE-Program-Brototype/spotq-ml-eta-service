
# ETA Model Lifecycle Management

## Startup validation

The ETA service loads its configured XGBoost artifact during application
startup. The loader validates the artifact's feature count and ordered feature
names, provenance metadata, and a smoke prediction before activation. Invalid
artifacts must not be activated.

Synthetic artifacts are permitted only in development when
`ALLOW_SYNTHETIC_MODEL=true`. Production must use an artifact marked as
trained on real data.

## Reload and rollback

Model administration endpoints require the configured `X-Admin-Key` header.
The service does not accept arbitrary filesystem paths through these endpoints.

- `POST /api/v1/eta/admin/reload-model` validates the configured artifact
  before activation.
- `POST /api/v1/eta/admin/rollback-model` restores the previous model when one
  is available.
- A failed reload must leave the currently active model intact.
- Reload and rollback operations are process-local.

## Multiple workers and replicas

Each application worker or replica maintains its own active model and
previous-model state in memory. A successful reload request updates only the
process that handles that request; it does not automatically synchronize other
workers or replicas.

For multi-worker or multi-replica deployments, coordinate reloads through the
deployment or orchestration layer and verify model status on every instance.
Do not assume that one successful reload request updates the entire service.

## Health endpoints

- `/livez` reports whether the application process is alive.
- `/readyz` reports whether the service can serve requests based on its
  required dependencies and model state.
- `/healthz` provides dependency and model health details.

Health responses should not expose filesystem paths, credentials, or other
secrets.

## Scope

This lifecycle capability covers startup validation, controlled model reload,
rollback, health reporting, and operational metrics. It does not implement
automated retraining, model promotion based on evaluation metrics, drift
detection, or scheduled retraining.

For candidate training, chronological evaluation, and MLflow experiment tracking,
see [Model Training Workflow](model-training-workflow.md).
