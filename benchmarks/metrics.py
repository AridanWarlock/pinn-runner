"""Error measures and strict, comparable numerical regression gates."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .cases import Array, BenchmarkCase


def solution_errors(
    case: BenchmarkCase, spatial: dict[str, Array], time: Array, prediction: Array
) -> dict[str, float]:
    expected = case.reference(spatial, time.T)
    if prediction.shape != expected.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction must have the reference shape and finite values")
    error = prediction - expected
    denominator = float(np.linalg.norm(expected))
    if denominator == 0:
        raise ValueError("Relative L2 requires a nonzero reference")
    boundary = np.zeros(len(prediction), dtype=bool)
    for name, values in spatial.items():
        boundary |= np.isclose(values[:, 0], 0)
        if case.name != "advection":
            boundary |= np.isclose(values[:, 0], 1)
    return {
        "relative_l2": float(np.linalg.norm(error) / denominator),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
        "initial_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "boundary_rmse": float(np.sqrt(np.mean(error[boundary] ** 2))),
        "final_relative_l2": float(np.linalg.norm(error[:, -1]) / np.linalg.norm(expected[:, -1])),
    }


def quality_failures(case: BenchmarkCase, metrics: Mapping[str, float]) -> list[str]:
    limits = {
        "relative_l2": case.relative_l2_limit,
        "max_error": case.max_error_limit,
        "pde_rmse": case.residual_rmse_limit,
        "initial_rmse": 0.05,
        "boundary_rmse": 0.05,
    }
    return [
        f"{name}={metrics[name]:.6g} exceeds {limit}"
        for name, limit in limits.items()
        if not math.isfinite(metrics[name]) or metrics[name] > limit
    ]


def regression_failures(current: Mapping[str, float], baseline: Mapping[str, float]) -> list[str]:
    # Absolute floors keep rounding/noise near zero from triggering relative alarms.
    tolerances = {"relative_l2": 0.002, "max_error": 0.005, "pde_rmse": 0.005}
    if any(
        not math.isfinite(values[name]) or values[name] < 0
        for values in (current, baseline)
        for name in tolerances
    ):
        raise ValueError("Regression metrics must be finite and nonnegative")
    return [
        f"{name} regressed: {current[name]:.6g} vs {baseline[name]:.6g}"
        for name, floor in tolerances.items()
        if current[name] > baseline[name] * 1.5 + floor
    ]
