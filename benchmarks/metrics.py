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
    boundary = case.boundary_mask(spatial)
    return {
        **case.additional_errors(spatial, time.T, error),
        "relative_l2": float(np.linalg.norm(error) / denominator),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
        "initial_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "boundary_rmse": float(np.sqrt(np.mean(error[boundary] ** 2))),
        "final_relative_l2": float(np.linalg.norm(error[:, -1]) / np.linalg.norm(expected[:, -1])),
    }


def quality_failures(case: BenchmarkCase, metrics: Mapping[str, float]) -> list[str]:
    return case.quality.failures(metrics)


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


def temporal_errors(
    case: BenchmarkCase, spatial: dict[str, Array], time: Array, prediction: Array
) -> list[dict[str, float | None]]:
    """Separate actual error growth from the decay of the reference norm."""
    expected = case.reference(spatial, time.T)
    if prediction.shape != expected.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction must have the reference shape and finite values")
    boundary = case.boundary_mask(spatial)
    error = prediction - expected
    results: list[dict[str, float | None]] = []
    for index, instant in enumerate(time.reshape(-1)):
        reference_norm = float(np.linalg.norm(expected[:, index]))
        results.append(
            {
                "time": float(instant),
                "reference_l2": reference_norm,
                "relative_l2": float(np.linalg.norm(error[:, index]) / reference_norm)
                if reference_norm > 0
                else None,
                "rmse": float(np.sqrt(np.mean(error[:, index] ** 2))),
                "max_error": float(np.max(np.abs(error[:, index]))),
                "boundary_rmse": float(np.sqrt(np.mean(error[boundary, index] ** 2)))
                if boundary.any()
                else None,
                "interior_rmse": float(np.sqrt(np.mean(error[~boundary, index] ** 2)))
                if (~boundary).any()
                else None,
            }
        )
    return results
