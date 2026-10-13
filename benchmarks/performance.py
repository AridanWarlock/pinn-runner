"""Warm, synchronized prediction measurements on inputs already on the device."""

from __future__ import annotations

import statistics
import time
from typing import Protocol

import torch

from pinnstorch.contracts import SamplerBatch, TensorOutputs


class Predictor(Protocol):
    def predict_step(self, batch: SamplerBatch, batch_idx: int) -> TensorOutputs: ...


def prediction_speed(
    model: Predictor, spatial: list[torch.Tensor], times: torch.Tensor, batch_size: int
) -> dict[str, float | int]:
    count = len(times)
    if count == 0 or batch_size < 1 or any(len(column) != count for column in spatial):
        raise ValueError(
            "Prediction timing requires aligned nonempty inputs and positive batch size"
        )
    durations: list[float] = []
    warmup, repeats = 3, 7
    for iteration in range(warmup + repeats):
        if times.device.type == "cuda":
            torch.cuda.synchronize(times.device)
        started = time.perf_counter()
        for index, start in enumerate(range(0, count, batch_size)):
            end = start + batch_size
            batch: SamplerBatch = ([column[start:end] for column in spatial], times[start:end], {})
            model.predict_step(batch, index)
        if times.device.type == "cuda":
            torch.cuda.synchronize(times.device)
        if iteration >= warmup:
            durations.append(time.perf_counter() - started)
    median = statistics.median(durations)
    return {
        "warm_prediction_seconds": median,
        "warm_prediction_points_per_second": count / median,
        "warm_prediction_min_seconds": min(durations),
        "warm_prediction_max_seconds": max(durations),
        "prediction_points": count,
        "prediction_batch_size": batch_size,
        "prediction_warmup_passes": warmup,
        "prediction_measurement_passes": repeats,
    }
