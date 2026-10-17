"""Snapshot training and post-training physical evaluation with explicit budgets."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pinnstorch.models.net.spatial_features import FeatureFCN
from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.checkpoint_io import load_checkpoint
from pinnstorch.runner.factory import ExecutionObjects, SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import Reader, Refinement, TaskSettings
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement

from .boundary_metrics import boundary_operator_errors
from .cases import BenchmarkCase
from .execution import holdout_inputs
from .fingerprints import objective_fingerprint, tensor_fingerprint
from .metrics import solution_errors, temporal_errors
from .performance import prediction_speed
from .variants import (
    Backward,
    HardBackground,
    HardExecution,
    HardLifting,
    SpatialEncoding,
    TrainingComposition,
)


@dataclass(frozen=True)
class ConvergenceArm:
    execution: str = "eager"
    lifting: str = "recursive"
    backward: str = "all"
    background: str = "general"
    encoding: str = "identity"

    @property
    def name(self) -> str:
        base = f"{self.execution}-{self.lifting}-{self.backward}"
        if self.background != "general":
            base += f"-{self.background}"
        return base if self.encoding == "identity" else f"{base}-{self.encoding}"

    def apply(self, config: dict[str, Any]) -> None:
        TrainingComposition(
            (
                HardExecution(self.execution),
                HardLifting(self.lifting),
                Backward(self.backward),
                HardBackground(self.background),
                SpatialEncoding(self.encoding),
            )
        ).apply(config)


@dataclass(frozen=True)
class ConvergencePlan:
    case: BenchmarkCase
    steps: int
    snapshots: tuple[int, ...]
    limits: dict[str, float]
    constraints: str = "hard_initial_dirichlet"
    points: int | None = None
    device: str = "cuda"
    tolerance_change: float = Refinement().tolerance_change

    def __post_init__(self) -> None:
        if (
            self.steps < 1
            or not self.snapshots
            or self.snapshots[0] != 0
            or any(b <= a for a, b in zip(self.snapshots, self.snapshots[1:]))
        ):
            raise ValueError(
                "Positive Adam steps and increasing snapshots starting at zero required"
            )
        if not self.limits or any(not math.isfinite(x) or x <= 0 for x in self.limits.values()):
            raise ValueError("Finite positive quality limits required")
        if self.points is not None and self.points < 1:
            raise ValueError("Collocation point count must be positive")
        if not math.isfinite(self.tolerance_change) or self.tolerance_change <= 0:
            raise ValueError("L-BFGS tolerance_change must be finite and positive")

    def config(self, seed: int, arm: ConvergenceArm, evaluating: bool = False) -> dict[str, Any]:
        config = self.case.config(self.device, self.steps, seed, evaluating)
        config["training"].update(
            constraints=self.constraints,
            refinement={
                "steps": self.snapshots[-1],
                "tolerance_change": self.tolerance_change,
            },
        )
        if self.points is not None:
            config["training"]["collocation"].update(points=self.points, sampling="lhs")
        arm.apply(config)
        return config


def crossing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Two adjacent successes confirm only an observed interval, not interpolation."""
    if not rows:
        raise ValueError("Crossing requires nonempty observations")
    for index in range(len(rows) - 1):
        if rows[index]["passed"] and rows[index + 1]["passed"]:
            return {
                "status": "observed",
                "lower_seconds": rows[index - 1]["seconds"] if index else None,
                "upper_seconds": rows[index]["seconds"],
                "confirmed_seconds": rows[index + 1]["seconds"],
                "left_censored": index == 0,
            }
    return {"status": "not_confirmed", "observed_until_seconds": rows[-1]["seconds"]}


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


class SnapshotTraining:
    def __init__(self, plan: ConvergencePlan, objects: ExecutionObjects, directory: Path) -> None:
        self.plan, self.objects, self.directory = plan, objects, directory

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        objects, device = self.objects, self.plan.device
        initial_fingerprint = tensor_fingerprint(objects.model.state_dict())
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        started = time.perf_counter()
        objects.trainer.fit(objects.model, datamodule=objects.data)
        objects.model.to(device)
        recipe = Refinement.read(
            Reader(dict(objects.model.hparams["refinement_settings"]), "training.refinement")
        )
        stage = LBFGSRefinement(
            objects.model,
            FixedObjective.capture(objects.data, objects.model.device),
            recipe.learning_rate,
            recipe.history_size,
            recipe.tolerance_grad,
            recipe.tolerance_change,
        )
        rows: list[dict[str, Any]] = []
        overhead = 0.0
        for step in self.plan.snapshots:
            stage.run(step - stage.steps)
            synchronize(device)
            elapsed = time.perf_counter() - started
            write_started = time.perf_counter()
            torch.save(objects.model.state_dict(), self.directory / f"lb-{step}.pt")
            synchronize(device)
            overhead += time.perf_counter() - write_started
            rows.append(
                {
                    "lb_steps": step,
                    "seconds": elapsed,
                    "closures": stage.closures,
                    "adam_residual_points": objects.model.adam_residual_points,
                    "lbfgs_residual_points": stage.closures
                    * len(stage.objective.batches["equations"][1]),
                }
            )
        training_seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated() if device == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved() if device == "cuda" else None
        return rows, {
            "snapshot_seconds": overhead,
            "training_seconds_with_snapshots": training_seconds,
            "initial_state_sha256": initial_fingerprint,
            "fixed_objective_sha256": objective_fingerprint(stage.objective.batches),
            "adam_steps": objects.trainer.global_step,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }


class SnapshotEvaluation:
    def __init__(self, plan: ConvergencePlan, factory: SolverFactory) -> None:
        self.plan, self.factory = plan, factory
        self.objects = factory.build(predicting=True)
        self.model = self.objects.model.to(plan.device).eval()
        mesh = self.objects.mesh
        spatial, clock, _ = mesh.flatten_mesh(None)
        self.features = [
            torch.tensor(spatial[:, i : i + 1], dtype=torch.float32, device=plan.device)
            for i in range(spatial.shape[1])
        ]
        self.clock = torch.tensor(clock.copy(), dtype=torch.float32, device=plan.device)
        self.axes = {
            name: mesh.spatial_domain_mesh[:, 0, i : i + 1].astype(np.float64)
            for i, name in enumerate(factory.settings.domain.space)
        }
        self.grid_time = np.asarray(mesh.time_domain, dtype=np.float64).reshape(-1, 1)
        self.holdout = holdout_inputs(factory.settings)

    def evaluate(self, checkpoint: Path) -> tuple[dict[str, float], dict[str, Any]]:
        self.model.load_state_dict(load_checkpoint(checkpoint, map_location=self.plan.device))
        chunks = []
        for index, start in enumerate(range(0, len(self.clock), 2048)):
            outputs = self.model.predict_step(
                (
                    [x[start : start + 2048] for x in self.features],
                    self.clock[start : start + 2048],
                    {},
                ),
                index,
            )
            chunks.append(outputs["u"].detach().cpu().numpy().astype(np.float64))
        values = np.concatenate(chunks).reshape(len(next(iter(self.axes.values()))), -1)
        metrics = solution_errors(self.plan.case, self.axes, self.grid_time, values)
        columns = [
            torch.tensor(
                self.holdout[:, i : i + 1],
                dtype=torch.float32,
                device=self.plan.device,
                requires_grad=True,
            )
            for i in range(self.holdout.shape[1])
        ]
        with torch.set_grad_enabled(True):
            physical = dict(self.model(columns[:-1], columns[-1]))
            residuals = self.factory.adapter.residuals(physical, *columns)
            residual = torch.cat(
                [residuals[name] for name in self.factory.problem.equations], dim=1
            )
            metrics["pde_rmse"] = float(residual.detach().square().mean().sqrt().cpu())
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite snapshot metric")
        return metrics, {
            "temporal": temporal_errors(self.plan.case, self.axes, self.grid_time, values),
            "boundary_operators": [
                asdict(row)
                for row in boundary_operator_errors(
                    self.factory.settings, self.factory.adapter, self.model, self.plan.device
                )
            ],
        }

    def performance(self, model: PINNModule | None = None) -> dict[str, int | float]:
        predictor = self.model if model is None else model
        return prediction_speed(predictor, self.features, self.clock, 2048)


def convergence_trial(
    plan: ConvergencePlan, arm: ConvergenceArm, seed: int, directory: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=False)
    config = plan.config(seed, arm)
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    problem = load_problem(plan.case.source)
    factory = SolverFactory(TaskSettings.read(config), problem, directory, plan.case.source.parent)
    objects = factory.build(predicting=False)
    setup_seconds = time.perf_counter() - started
    rows, performance = SnapshotTraining(plan, objects, directory).run()
    evaluation_started = time.perf_counter()
    # Separate evaluator: force physical-autograd control, preserve the actual lifting.
    evaluator_config = plan.config(
        seed, ConvergenceArm("eager", arm.lifting, "all", "general", arm.encoding), evaluating=True
    )
    evaluator = SnapshotEvaluation(
        plan,
        SolverFactory(
            TaskSettings.read(evaluator_config), problem, directory, plan.case.source.parent
        ),
    )
    for row in rows:
        metrics, diagnostics = evaluator.evaluate(directory / f"lb-{row['lb_steps']}.pt")
        missing = plan.limits.keys() - metrics.keys()
        if missing:
            raise ValueError(f"Unknown quality metrics: {sorted(missing)}")
        row.update(
            metrics=metrics,
            **diagnostics,
            passed=all(metrics[name] <= limit for name, limit in plan.limits.items()),
        )
    # Quality uses the independent control; deployment timing uses the trained executor.
    performance.update(evaluator.performance(objects.model.eval()))
    control_prediction = evaluator.performance()
    performance.update(
        setup_seconds=setup_seconds,
        evaluation_seconds=time.perf_counter() - evaluation_started,
        trial_wall_seconds=time.perf_counter() - started,
    )
    return {
        "case": plan.case.name,
        "backend": arm.execution,
        "lifting": arm.lifting,
        "backward": arm.backward,
        "background": arm.background,
        "spatial_encoding": arm.encoding,
        "feature_derivatives": objects.model.net.features.derivative_rule
        if isinstance(objects.model.net, FeatureFCN)
        else None,
        "control_feature_derivatives": evaluator.model.net.features.derivative_rule
        if isinstance(evaluator.model.net, FeatureFCN)
        else None,
        "network": config["network"],
        "trainable_parameters": sum(
            p.numel() for p in objects.model.parameters() if p.requires_grad
        ),
        "resolved_background": objects.model.hparams["hard_background"],
        "resolved_backward": objects.model.hparams["backward_mode"],
        "prediction_execution": objects.model.hparams["hard_execution"],
        "control_prediction_execution": evaluator.model.hparams["hard_execution"],
        "control_prediction_background": evaluator.model.hparams["hard_background"],
        "control_prediction_performance": control_prediction,
        "seed": seed,
        "lbfgs_tolerance_change": plan.tolerance_change,
        "snapshots": rows,
        "crossing": crossing(rows),
        **performance,
    }
