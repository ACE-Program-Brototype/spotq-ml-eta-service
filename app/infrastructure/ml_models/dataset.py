"""Chronological, leakage-aware dataset splitting for SpotQ ETA."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable

from sklearn.model_selection import TimeSeriesSplit

from app.infrastructure.ml_models.features import parse_timestamp


@dataclass(frozen=True)
class ChronologicalSplit:
    """A sorted dataset split with an embargo for incomplete labels."""

    records: tuple[dict[str, Any], ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    embargo_indices: tuple[int, ...]
    cutoff: datetime

    @property
    def train_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.records[i] for i in self.train_indices)

    @property
    def test_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.records[i] for i in self.test_indices)

    @property
    def embargo_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.records[i] for i in self.embargo_indices)


@dataclass(frozen=True)
class PurgedTimeSeriesCV:
    """Expanding-window CV folds with completion-time purging."""

    records: tuple[dict[str, Any], ...]
    folds: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...
    ]


def _sort_and_validate(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate timestamps and sort records by join instant."""
    parsed_records = []

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every dataset record must be a dictionary")

        try:
            joined_at = parse_timestamp(record["joined_at"])
            completed_at = parse_timestamp(record["completed_at"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Every record must contain valid timezone-aware "
                "joined_at and completed_at timestamps"
            ) from exc

        if completed_at <= joined_at:
            raise ValueError("Completion must be after join time")

        parsed_records.append((record, joined_at, completed_at))

    if len(parsed_records) < 3:
        raise ValueError("At least three records are required")

    # Python's sort is stable, preserving input order for tied join times.
    parsed_records.sort(key=lambda item: item[1])

    return tuple(item[0] for item in parsed_records)


def chronological_train_test_split(
    records: Iterable[dict[str, Any]],
    test_fraction: float = 0.20,
) -> ChronologicalSplit:
    """Reserve the latest period for a final holdout.

    Records joining before the cutoff but completing at or after the cutoff
    are embargoed: their labels would not have been available at the cutoff.
    All records joining at the cutoff belong to the test period.
    """
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float))
        or not math.isfinite(test_fraction)
        or not 0 < test_fraction < 1
    ):
        raise ValueError("test_fraction must be strictly between 0 and 1")

    ordered = _sort_and_validate(records)
    requested_test_count = max(
        1, math.ceil(len(ordered) * test_fraction)
    )

    cutoff = parse_timestamp(
        ordered[len(ordered) - requested_test_count]["joined_at"]
    )

    train_indices = []
    test_indices = []
    embargo_indices = []

    for index, record in enumerate(ordered):
        joined_at = parse_timestamp(record["joined_at"])
        completed_at = parse_timestamp(record["completed_at"])

        if joined_at >= cutoff:
            test_indices.append(index)
        elif completed_at < cutoff:
            train_indices.append(index)
        else:
            embargo_indices.append(index)

    if not train_indices or not test_indices:
        raise ValueError(
            "Split produced an empty train or test set; "
            "provide more records or adjust test_fraction"
        )

    return ChronologicalSplit(
        records=ordered,
        train_indices=tuple(train_indices),
        test_indices=tuple(test_indices),
        embargo_indices=tuple(embargo_indices),
        cutoff=cutoff,
    )


def make_purged_time_series_cv(
    records: Iterable[dict[str, Any]],
    n_splits: int = 5,
    test_size: int | None = None,
) -> PurgedTimeSeriesCV:
    """Build expanding-window CV folds on unique join timestamps.

    Validation windows contain contiguous groups of join timestamps.
    Training records must join before, and complete strictly before,
    each validation window begins. Indices refer to the returned sorted
    records and can be passed to scikit-learn model-selection tools.
    """
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, int)
        or n_splits < 2
    ):
        raise ValueError("n_splits must be an integer >= 2")

    if test_size is not None and (
        isinstance(test_size, bool)
        or not isinstance(test_size, int)
        or test_size < 1
    ):
        raise ValueError("test_size must be a positive integer")

    ordered = _sort_and_validate(records)

    unique_times = sorted(
        {parse_timestamp(record["joined_at"]) for record in ordered}
    )

    if len(unique_times) <= n_splits:
        raise ValueError(
            "Insufficient unique join timestamps for the requested CV folds"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
    )

    folds = []

    try:
        time_splits = splitter.split(unique_times)

        for _, validation_time_indices in time_splits:
            validation_times = {
                unique_times[int(i)] for i in validation_time_indices
            }
            validation_start = min(validation_times)

            train_indices = []
            validation_indices = []

            for index, record in enumerate(ordered):
                joined_at = parse_timestamp(record["joined_at"])
                completed_at = parse_timestamp(record["completed_at"])

                if joined_at in validation_times:
                    validation_indices.append(index)
                elif joined_at < validation_start and completed_at < validation_start:
                    train_indices.append(index)

            if not train_indices or not validation_indices:
                raise ValueError(
                    "A CV fold has an empty train or validation set; "
                    "provide more data or reduce n_splits"
                )

            folds.append(
                (tuple(train_indices), tuple(validation_indices))
            )

    except ValueError as exc:
        raise ValueError(
            f"Could not create purged time-series CV: {exc}"
        ) from exc

    return PurgedTimeSeriesCV(
        records=ordered,
        folds=tuple(folds),
    )
