import pandas as pd
import pytest

from scripts.train_eta_model import (
    compute_regression_metrics,
    validate_feature_columns,
)
from app.infrastructure.ml_models.features import FEATURE_NAMES


def test_regression_metrics():
    metrics = compute_regression_metrics(
        [10.0, 20.0, 30.0],
        [12.0, 18.0, 33.0],
    )

    assert metrics["mae_minutes"] == pytest.approx(7 / 3)
    assert metrics["rmse_minutes"] == pytest.approx((17 / 3) ** 0.5)
    assert metrics["r2"] is not None


def test_feature_matrix_schema_is_validated():
    frame = pd.DataFrame(
        [[2, 5, 12, 0, 0, 15.0]],
        columns=FEATURE_NAMES,
    )
    validate_feature_columns(frame)


def test_feature_matrix_rejects_wrong_order():
    frame = pd.DataFrame(
        [[2, 5, 12, 0, 0, 15.0]],
        columns=list(reversed(FEATURE_NAMES)),
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_feature_columns(frame)


def test_feature_matrix_rejects_missing_values():
    frame = pd.DataFrame(
        [[2, 5, 12, 0, 0, float("nan")]],
        columns=FEATURE_NAMES,
    )

    with pytest.raises(ValueError, match="missing values"):
        validate_feature_columns(frame)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_regression_metrics([1.0, 2.0], [1.0])
