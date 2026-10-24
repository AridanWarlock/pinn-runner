"""Snapshot training and post-training physical evaluation with explicit budgets."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, replace
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
from .burgers_quality import LayerEvaluation, evaluation_identity
from .cases import BenchmarkCase
from .catalog import UNFORCED_VISCOSITIES
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
    points: int | None = None
    coefficient_representation: str | None = None

    @property
    def name(self) -> str:
        base = f"{self.execution}-{self.lifting}-{self.backward}"
        if self.background != "general":
            base += f"-{self.background}"
        base = base if self.encoding == "identity" else f"{base}-{self.encoding}"
        if self.coefficient_representation is not None:
            base += f"-{self.coefficient_representation}"
        return base if self.points is None else f"{base}-n{self.points}"

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
    density: dict[str, Any] | None = None

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
        points = self.points if arm.points is None else arm.points
        if points is not None:
            config["training"]["collocation"].update(points=points, sampling="lhs")
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
        initial_parameters = tensor_fingerprint(dict(objects.model.named_parameters()))
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
            "initial_parameters_sha256": initial_parameters,
            "fixed_objective_sha256": objective_fingerprint(stage.objective.batches),
            "adam_steps": objects.trainer.global_step,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }


class SnapshotEvaluation:
    def __init__(self, plan: ConvergencePlan, factory: SolverFactory) -> None:
        self.plan, self.factory = plan, factory
        if plan.case.name in ("heat_constant_surrogate", "heat_spatial_scale_surrogate"):
            domain = factory.settings.domain
            parameter = "a" if plan.case.name == "heat_spatial_scale_surrogate" else "k"
            if (
                tuple(domain.space) != ("x",)
                or tuple(domain.space["x"].interval) != (0, 1)
                or tuple(domain.time.interval) != (0, 1)
                or tuple(domain.parameters) != (parameter,)
                or tuple(domain.parameters[parameter].interval) != (0.05, 0.25)
            ):
                raise ValueError("Surrogate evaluation requires registered x/t/k intervals")
        if plan.case.name in ("heat_quadratic_shapes", "heat_quadratic_clock"):
            from .catalog import QUADRATIC_CLOCK_TRAINING_ROWS, QUADRATIC_TRAINING_ROWS

            training_rows = (
                QUADRATIC_CLOCK_TRAINING_ROWS
                if plan.case.name == "heat_quadratic_clock"
                else QUADRATIC_TRAINING_ROWS
            )

            domain = factory.settings.domain
            if (
                tuple(domain.space) != ("x",)
                or tuple(domain.space["x"].interval) != (0, 1)
                or tuple(domain.time.interval) != (0, 1)
                or tuple(domain.parameters) != ("a", "b")
                or tuple(domain.parameters["a"].interval) != (0.05, 0.15)
                or tuple(domain.parameters["b"].interval) != (1, 2)
                or domain.parameter_rows != tuple(tuple(row) for row in training_rows)
            ):
                raise ValueError(
                    "Whole-code evaluation requires registered domain and training rows"
                )
            expected_clock = (
                {"parameter": "a", "reference": 0.15}
                if plan.case.name == "heat_quadratic_clock"
                else None
            )
            if (
                asdict(factory.settings.network.time_scale)
                if factory.settings.network.time_scale
                else None
            ) != expected_clock:
                raise ValueError("Whole-code evaluation requires registered time scale")
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
        if plan.case.name in UNFORCED_VISCOSITIES and (
            list(factory.settings.domain.space["x"].interval) != [-1, 1]
            or list(factory.settings.domain.time.interval) != [0, 1]
        ):
            raise ValueError("Unforced evaluator requires the registered physical domain")
        self.layer = (
            LayerEvaluation(
                UNFORCED_VISCOSITIES[plan.case.name], (len(self.axes["x"]), len(self.grid_time))
            )
            if plan.case.name in UNFORCED_VISCOSITIES
            else None
        )

    def evaluate(self, checkpoint: Path) -> tuple[dict[str, float], dict[str, Any]]:
        self.model.load_state_dict(load_checkpoint(checkpoint, map_location=self.plan.device))
        if self.plan.case.name in ("heat_quadratic_shapes", "heat_quadratic_clock"):
            from .quadratic_surrogate_quality import evaluate as evaluate_quadratic

            return evaluate_quadratic(
                self.model, self.factory.adapter, self.plan.device, self.plan.case.name
            )
        if self.plan.case.name == "heat_spatial_scale_surrogate":
            from .spatial_surrogate_quality import evaluate as evaluate_spatial

            return evaluate_spatial(self.model, self.factory.adapter, self.plan.device)
        if self.plan.case.name == "heat_constant_surrogate":
            from .surrogate_quality import evaluate

            return evaluate(self.model, self.factory.adapter, self.plan.device)
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
        if self.plan.case.name.startswith("heat_conductivity_beta"):
            flux_settings = replace(
                self.factory.settings,
                boundaries=tuple(
                    replace(boundary, kind="heat_flux", conductivity="conductivity")
                    for boundary in self.factory.settings.boundaries
                ),
            )
            metrics["physical_flux_max"] = max(
                row.max_error
                for row in boundary_operator_errors(
                    flux_settings, self.factory.adapter, self.model, self.plan.device
                )
            )
        temporal = temporal_errors(self.plan.case, self.axes, self.grid_time, values)
        if self.layer is not None:
            layer_metrics, layer_temporal = self.layer.evaluate(
                self.model, self.factory.adapter, self.plan.limits, self.plan.device
            )
            metrics.update(layer_metrics)
            temporal = layer_temporal  # type: ignore[assignment]
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite snapshot metric")
        return metrics, {
            "temporal": temporal,
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
    source = plan.case.prepare_source(directory, arm.coefficient_representation)
    problem = load_problem(source)
    factory = SolverFactory(TaskSettings.read(config), problem, directory, source.parent)
    objects = factory.build(predicting=False)
    setup_seconds = time.perf_counter() - started
    if plan.density is None:
        rows, performance = SnapshotTraining(plan, objects, directory).run()
    else:
        from .density import density_training

        rows, performance = density_training(plan, arm, seed, objects, directory)
    evaluation_started = time.perf_counter()
    # Separate evaluator: force physical-autograd control, preserve the actual lifting.
    evaluator_config = plan.config(
        seed, ConvergenceArm("eager", arm.lifting, "all", "general", arm.encoding), evaluating=True
    )
    evaluator = SnapshotEvaluation(
        plan,
        SolverFactory(TaskSettings.read(evaluator_config), problem, directory, source.parent),
    )
    for row in rows:
        metrics, diagnostics = evaluator.evaluate(directory / f"lb-{row['lb_steps']}.pt")
        metrics["worst_time_relative_l2"] = max(
            item["relative_l2"]
            for item in diagnostics["temporal"]
            if item["relative_l2"] is not None
        )
        missing = plan.limits.keys() - metrics.keys()
        if missing:
            raise ValueError(f"Unknown quality metrics: {sorted(missing)}")
        row.update(
            metrics=metrics,
            **diagnostics,
            passed=all(metrics[name] <= limit for name, limit in plan.limits.items()),
        )
        if plan.density is not None:
            row["passed"] &= row["seconds"] <= plan.density["training_seconds"]
    # Quality uses the independent control; deployment timing uses the trained executor.
    performance.update(evaluator.performance(objects.model.eval()))
    control_prediction = evaluator.performance()
    performance.update(
        setup_seconds=setup_seconds,
        evaluation_seconds=time.perf_counter() - evaluation_started,
        trial_wall_seconds=time.perf_counter() - started,
    )
    physical_identity: dict[str, Any] = {}
    if arm.coefficient_representation is not None:
        from .conductivity import physics, source_sha256

        actual_source = hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
        if actual_source != source_sha256(plan.case.name, arm.coefficient_representation):
            raise ValueError("Materialized problem differs from registered source")
        physical_identity = dict(
            coefficient_representation=arm.coefficient_representation,
            physics=physics(plan.case.name),
            problem_source_sha256=actual_source,
        )
    return {
        **(
            {
                "layer_evaluation": evaluation_identity(
                    evaluator.layer.coarse if evaluator.layer is not None else None
                )
            }
            if plan.case.name in UNFORCED_VISCOSITIES
            else {}
        ),
        **physical_identity,
        **({"points_override": arm.points} if plan.density is not None else {}),
        "case": plan.case.name,
        "constraints": objects.model.hparams["constraint_kind"],
        "backend": arm.execution,
        "lifting": arm.lifting,
        "backward": arm.backward,
        "background": arm.background,
        "spatial_encoding": arm.encoding,
        "problem_source_sha256": hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "feature_derivatives": objects.model.net.features.derivative_rule
        if isinstance(objects.model.net, FeatureFCN)
        else "autograd",
        "control_feature_derivatives": evaluator.model.net.features.derivative_rule
        if isinstance(evaluator.model.net, FeatureFCN)
        else "autograd",
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
