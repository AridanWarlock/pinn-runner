"""Independent residual summaries; diagnostics never participate in training."""

from __future__ import annotations

import numpy as np

from .cases import Array


def residual_time_bins(
    times: Array, residuals: Array, interval: tuple[float, float], count: int = 10
) -> list[dict[str, float | int | None]]:
    """Partition a fixed holdout; right endpoint belongs to the final bin."""
    clock = times.reshape(-1)
    lower, upper = interval
    if count < 1 or lower >= upper or not np.isfinite([lower, upper]).all():
        raise ValueError("Residual bins require a finite increasing interval and positive count")
    if (
        residuals.ndim != 2
        or len(residuals) != len(clock)
        or residuals.shape[1] == 0
        or not np.isfinite(clock).all()
        or not np.isfinite(residuals).all()
        or np.any((clock < lower) | (clock > upper))
    ):
        raise ValueError("Residuals must be finite aligned columns with times inside the interval")
    edges = np.linspace(lower, upper, count + 1)
    membership = np.minimum(np.searchsorted(edges, clock, side="right") - 1, count - 1)
    results: list[dict[str, float | int | None]] = []
    for index in range(count):
        values = residuals[membership == index]
        results.append(
            {
                "time_lower": float(edges[index]),
                "time_upper": float(edges[index + 1]),
                "points": len(values),
                "rmse": float(np.sqrt(np.mean(values**2))) if len(values) else None,
                "max_absolute": float(np.max(np.abs(values))) if len(values) else None,
            }
        )
    return results


def worst_time_error(rows: list[dict[str, float | None]]) -> dict[str, float | int | None]:
    valid = [(row["time"], row["relative_l2"]) for row in rows if row["relative_l2"] is not None]
    if not valid:
        return {"worst_time_l2": None, "worst_time": None, "undefined_layers": len(rows)}
    instant, error = max(valid, key=lambda pair: float(pair[1] or 0))
    return {
        "worst_time_l2": error,
        "worst_time": instant,
        "undefined_layers": len(rows) - len(valid),
    }
