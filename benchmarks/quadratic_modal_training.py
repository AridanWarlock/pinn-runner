"""Исследовательское обучение общего модального суррогата по интегралу PDE."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from pinnstorch.models.net.quadratic_modal import QuadraticModalFCN
from pinnstorch.runner.factory import ExecutionObjects
from pinnstorch.runner.problem import ProblemAdapter
from pinnstorch.runner.settings import Reader, Refinement

from .fingerprints import tensor_fingerprint
from .quadratic_modal_objective import ModalResidualObjective

if TYPE_CHECKING:
    from .convergence import ConvergencePlan


def modal_training(
    plan: ConvergencePlan, objects: ExecutionObjects, adapter: ProblemAdapter, directory: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = objects.model.to(device=torch.device(plan.device), dtype=torch.float32).train()
    net = model.net
    if not isinstance(net, QuadraticModalFCN) or net.normalization != "mode_gap":
        raise ValueError("Registered modal training requires mode_gap normalization")
    rows = plan.case.config(plan.device, plan.steps, 11)["domain"]["parameter_rows"]
    if any(a != 0.15 for a, _ in rows):
        raise ValueError("Registered modal training requires reference scale 0.15")
    objective = ModalResidualObjective(adapter.initial_tensors, [float(b) for _, b in rows]).to(
        device=torch.device(plan.device), dtype=torch.float32
    )
    recipe = Refinement.read(Reader(dict(model.hparams["refinement_settings"]), "refinement"))
    parameters = tuple(model.parameters())
    initial_state = tensor_fingerprint(model.state_dict())
    initial_parameters = tensor_fingerprint(dict(model.named_parameters()))
    fixed_objective = tensor_fingerprint(objective.state_dict())
    if plan.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    adam = torch.optim.Adam(parameters, lr=0.001, weight_decay=0)

    def loss() -> torch.Tensor:
        value = cast(torch.Tensor, objective(net.coefficients(objective.forms)))
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite modal loss")
        return value

    for _ in range(plan.steps):
        adam.zero_grad(set_to_none=True)
        model.backward(loss())
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("Nonfinite modal gradient")
        adam.step()
    lbfgs = torch.optim.LBFGS(
        parameters,
        lr=recipe.learning_rate,
        history_size=recipe.history_size,
        tolerance_grad=recipe.tolerance_grad,
        tolerance_change=recipe.tolerance_change,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )
    closures = 0

    def closure() -> torch.Tensor:
        nonlocal closures
        lbfgs.zero_grad(set_to_none=True)
        value = loss()
        model.backward(value)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("Nonfinite modal gradient")
        closures += 1
        return value.detach()

    snapshots = []
    completed, overhead = 0, 0.0
    for step in plan.snapshots:
        for _ in range(step - completed):
            lbfgs.step(closure)
        completed = step
        if plan.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        writing = time.perf_counter()
        torch.save(model.state_dict(), directory / f"lb-{step}.pt")
        overhead += time.perf_counter() - writing
        with torch.no_grad():
            full_loss = float(objective.full_integral(net.coefficients(objective.forms)))
        snapshots.append(
            dict(
                lb_steps=step,
                seconds=elapsed,
                closures=closures,
                adam_residual_points=0,
                lbfgs_residual_points=0,
                modal_forms_per_evaluation=len(rows),
                modal_integral=full_loss,
                modal_form_evaluations=(plan.steps + closures) * len(rows),
            )
        )
    return snapshots, dict(
        snapshot_seconds=overhead,
        training_seconds_with_snapshots=time.perf_counter() - started,
        initial_state_sha256=initial_state,
        initial_parameters_sha256=initial_parameters,
        fixed_objective_sha256=fixed_objective,
        adam_steps=plan.steps,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if plan.device == "cuda" else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if plan.device == "cuda" else None,
        modal_training=dict(
            kind="continuous_pde_initial_projections",
            quadrature=256,
            forms=len(rows),
            normalization="mode_gap",
            xt_sampling=False,
        ),
    )
