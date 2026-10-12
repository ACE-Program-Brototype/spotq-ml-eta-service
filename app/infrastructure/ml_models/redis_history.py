"""Read validated historical ETA averages from Redis within a strict time budget."""
import asyncio
import math
from time import perf_counter

from app.infrastructure.cache.redis_client import redis_manager
from app.infrastructure.config.settings import settings
from app.infrastructure.logging.logger import logger
from app.infrastructure.observability.metrics import REDIS_LOOKUP_LATENCY


async def get_historical_average(
    restaurant_id: str,
    day_of_week: int,
    hour_of_day: int,
    *,
    cold_start_minutes: float = 20.0,
    client=None,
    timeout_seconds: float | None = None,
) -> tuple[float, str]:
    """Resolve history in priority order without allowing Redis to stall a request.

    Sources are ``exact``, ``restaurant_weekday``, ``restaurant``, ``global``,
    ``cold_start``, ``redis_timeout``, and ``redis_error``.
    """
    if not isinstance(restaurant_id, str) or not restaurant_id.strip():
        raise ValueError("restaurant_id must be a non-empty string")
    if (
        isinstance(day_of_week, bool)
        or not isinstance(day_of_week, int)
        or not 0 <= day_of_week <= 6
    ):
        raise ValueError("day_of_week must be an integer from 0 to 6")
    if (
        isinstance(hour_of_day, bool)
        or not isinstance(hour_of_day, int)
        or not 0 <= hour_of_day <= 23
    ):
        raise ValueError("hour_of_day must be an integer from 0 to 23")
    if (
        isinstance(cold_start_minutes, bool)
        or not isinstance(cold_start_minutes, (int, float))
        or not math.isfinite(cold_start_minutes)
        or cold_start_minutes < 0
    ):
        raise ValueError("cold_start_minutes must be finite and non-negative")

    fallback = float(cold_start_minutes)
    redis = client if client is not None else redis_manager.client
    if redis is None:
        return fallback, "cold_start"

    keys = (
        (f"eta:history:{restaurant_id}:{day_of_week}:{hour_of_day}", "exact"),
        (f"eta:history:{restaurant_id}:weekday:{day_of_week}", "restaurant_weekday"),
        (f"eta:history:{restaurant_id}:all", "restaurant"),
        ("eta:history:global", "global"),
    )
    budget = (
        settings.REDIS_LOOKUP_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("timeout_seconds must be finite and positive")

    started = perf_counter()
    try:
        async with asyncio.timeout(budget):
            for key, source in keys:
                raw = await redis.get(key)
                if raw is None:
                    continue
                try:
                    average = float(raw)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(average) and average > 0:
                    return average, source
        return fallback, "cold_start"
    except TimeoutError:
        logger.warning(
            "ETA history lookup timed out",
            restaurant_id=restaurant_id,
            timeout_seconds=budget,
        )
        return fallback, "redis_timeout"
    except Exception as exc:
        logger.warning(
            "ETA history lookup failed; using cold-start value",
            restaurant_id=restaurant_id,
            error_type=type(exc).__name__,
        )
        return fallback, "redis_error"
    finally:
        REDIS_LOOKUP_LATENCY.observe(perf_counter() - started)
