"""Nested fixed equation samples and a measured time budget for density studies."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from pinnstorch.runner.factory import ExecutionObjects
from pinnstorch.runner.settings import Reader, Refinement
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement

from .convergence import ConvergenceArm, ConvergencePlan, synchronize
from .fingerprints import objective_fingerprint, tensor_fingerprint


def nested_points(seed: int, count: int, lower: Any, upper: Any) -> np.ndarray[Any, Any]:
    """Independent local RNG; a shorter request is an exact prefix, including in FP32."""
    low, high = np.asarray(lower).reshape(-1), np.asarray(upper).reshape(-1)
    points = np.random.default_rng(seed + 20261017).random((count, len(low)))
    return np.asarray(low + (high - low) * points, dtype=np.float32)


class Deadline(Callback):
    def __init__(self, started: float, budget: float, device: str) -> None:
        self.started, self.budget, self.device = started, budget, device

    def expired(self) -> bool:
        synchronize(self.device)
        return time.perf_counter() - self.started >= self.budget

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.expired():
            trainer.should_stop = True


def density_training(
    plan: ConvergencePlan,
    arm: ConvergenceArm,
    seed: int,
    objects: ExecutionObjects,
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert plan.density is not None and arm.points is not None
    preparation_started = time.perf_counter()
    point_count = arm.points
    density = plan.density
    budget, interval = density["training_seconds"], density["snapshot_seconds"]
    data = objects.data
    if data.condition_names != ("equations",):
        raise ValueError("Density study requires hard initial and boundary conditions")
    dataset = data.train_datasets[0]
    if (
        getattr(dataset, "sampling_policy", None) is not None
        or getattr(dataset, "time_map", None) is not None
    ):
        raise ValueError("Density study requires fixed uniform sampling")
    points = nested_points(seed, arm.points, objects.mesh.lb, objects.mesh.ub)
    data.train_datasets[0].store(points[:, :-1], points[:, -1:], {})
    initial = tensor_fingerprint(objects.model.state_dict())
    # This pool is not the independent quality holdout and is never resampled.
    objective = FixedObjective.capture(data, torch.device(plan.device))
    prefix_count = density["prefix_points"]
    prefix = {
        name: ([x[:prefix_count] for x in space], clock[:prefix_count], targets)
        for name, (space, clock, targets) in objective.batches.items()
    }
    if plan.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(plan.device)
    preparation_seconds = time.perf_counter() - preparation_started
    started = time.perf_counter()
    deadline = Deadline(started, budget, plan.device)
    callbacks: Any = getattr(objects.trainer, "callbacks")
    callbacks.append(deadline)
    objects.trainer.fit(objects.model, datamodule=data)
    objects.model.to(plan.device)
    recipe = Refinement.read(
        Reader(dict(objects.model.hparams["refinement_settings"]), "training.refinement")
    )
    stage = LBFGSRefinement(
        objects.model,
        objective,
        recipe.learning_rate,
        recipe.history_size,
        recipe.tolerance_grad,
        recipe.tolerance_change,
    )
    rows: list[dict[str, Any]] = []
    overhead = 0.0

    def save() -> None:
        nonlocal overhead
        synchronize(plan.device)
        elapsed = time.perf_counter() - started
        write_started = time.perf_counter()
        torch.save(objects.model.state_dict(), directory / f"lb-{stage.steps}.pt")
        synchronize(plan.device)
        overhead += time.perf_counter() - write_started
        rows.append(
            dict(
                lb_steps=stage.steps,
                seconds=elapsed,
                closures=stage.closures,
                adam_residual_points=objects.model.adam_residual_points,
                lbfgs_residual_points=stage.closures * point_count,
            )
        )

    save()
    next_save = (int(rows[-1]["seconds"] / interval) + 1) * interval
    while not deadline.expired() and stage.steps < plan.snapshots[-1]:
        stage.run(1)
        if time.perf_counter() - started >= next_save:
            save()
            next_save = (int(rows[-1]["seconds"] / interval) + 1) * interval
    if rows[-1]["lb_steps"] != stage.steps:
        save()
    synchronize(plan.device)
    elapsed = time.perf_counter() - started
    return rows, dict(
        density_preparation_seconds=preparation_seconds,
        snapshot_seconds=overhead,
        training_seconds_with_snapshots=elapsed,
        initial_state_sha256=initial,
        fixed_objective_sha256=objective_fingerprint(objective.batches),
        objective_prefix_sha256=objective_fingerprint(prefix),
        adam_steps=objects.trainer.global_step,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if plan.device == "cuda" else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if plan.device == "cuda" else None,
        density=density,
        stop_reason="time_budget" if elapsed >= budget else "step_cap",
        budget_overshoot_seconds=max(0.0, elapsed - budget),
    )
