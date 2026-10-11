"""Read pre-warmed historical ETA averages from Redis."""

import math

from app.infrastructure.cache.redis_client import redis_manager


async def get_historical_average(
    restaurant_id: str,
    day_of_week: int,
    hour_of_day: int,
    *,
    cold_start_minutes: float = 20.0,
    client=None,
) -> tuple[float, str]:
    """Resolve exact, restaurant/weekday, restaurant, global, then cold start."""
    if not restaurant_id or not isinstance(restaurant_id, str):
        raise ValueError("restaurant_id must be a non-empty string")
    if isinstance(day_of_week, bool) or not isinstance(day_of_week, int) or not 0 <= day_of_week <= 6:
        raise ValueError("day_of_week must be an integer from 0 to 6")
    if isinstance(hour_of_day, bool) or not isinstance(hour_of_day, int) or not 0 <= hour_of_day <= 23:
        raise ValueError("hour_of_day must be an integer from 0 to 23")
    if (
        isinstance(cold_start_minutes, bool)
        or not isinstance(cold_start_minutes, (int, float))
        or not math.isfinite(cold_start_minutes)
        or cold_start_minutes < 0
    ):
        raise ValueError("cold_start_minutes must be a finite non-negative number")

    redis = client or redis_manager.client
    if redis is None:
        return float(cold_start_minutes), "cold_start"

    keys = (
        (
            f"eta:history:{restaurant_id}:{day_of_week}:{hour_of_day}",
            "exact",
        ),
        (
            f"eta:history:{restaurant_id}:weekday:{day_of_week}",
            "restaurant_weekday",
        ),
        (f"eta:history:{restaurant_id}:all", "restaurant"),
        ("eta:history:global", "global"),
    )

    for key, source in keys:
        try:
            raw = await redis.get(key)
            if raw is None:
                continue
            average = float(raw)
            if math.isfinite(average) and average > 0:
                return average, source
        except (TypeError, ValueError):
            # Ignore malformed cache values and try the next fallback.
            continue

    return float(cold_start_minutes), "cold_start"
