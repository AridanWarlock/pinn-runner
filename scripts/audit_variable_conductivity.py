"""CPU diagnostic of the actual PINN's conservative residual and every weight gradient."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.api import Coordinates  # noqa: E402
from pinnstorch.models.pinn_module import PINNModule  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

SOURCE = ROOT / "benchmarks/problems/heat_variable_conductivity/functions.py"
K0 = 1 / 8
STEP = 1e-4


def points(dimensions: int, dtype: torch.dtype) -> torch.Tensor:
    """Fixed interior, faces/corners, table knots and their two-sided neighbourhoods."""
    interior = torch.rand((64, dimensions + 1), generator=torch.Generator().manual_seed(2718))
    axes = [
        [0, 0.125 - 1e-4, 0.125, 0.125 + 1e-4, 0.5 - 1e-4, 0.5, 0.5 + 1e-4, 1],
        [0, 0.25 - 1e-4, 0.25, 0.25 + 1e-4, 0.75 - 1e-4, 0.75, 0.75 + 1e-4, 1],
    ][:dimensions]
    boundary = torch.tensor(list(itertools.product(*axes, [0, 0.37, 1])), dtype=dtype)
    return torch.cat((interior.to(dtype), boundary))


def numpy_source(values: NDArray[np.float64], beta: int) -> NDArray[np.float64]:
    """Independent closed polynomial formula; no Equation or coefficient provider."""
    xs = values[:, :-1]
    dimensions = xs.shape[1]
    q = 3 * xs**2 - 2 * xs**3
    product = np.prod(q, axis=1)
    laplacian = np.zeros(len(xs))
    gradient = np.zeros(len(xs))
    for axis in range(dimensions):
        others = np.prod(np.delete(q, axis, axis=1), axis=1)
        laplacian += (6 - 12 * xs[:, axis]) * others
        gradient += 6 * xs[:, axis] * (1 - xs[:, axis]) * others
    kappa = K0 * (1 + beta * np.mean(xs, axis=1))
    return cast(
        NDArray[np.float64],
        np.exp(-values[:, -1:])
        * (-product - kappa * laplacian - K0 * beta / dimensions * gradient)[:, None],
    )


def make_factory(
    directory: Path,
    dimensions: int,
    beta: int,
    representation: str,
    seed: int,
    device: str = "cpu",
) -> SolverFactory:
    directory.mkdir(parents=True, exist_ok=False)
    text = SOURCE.read_text()
    for before, after in (
        ("DIMENSIONS = 2", f"DIMENSIONS = {dimensions}"),
        ("BETA = 1", f"BETA = {beta}"),
        ('REPRESENTATION = "formula"', f'REPRESENTATION = "{representation}"'),
    ):
        assert text.count(before) == 1
        text = text.replace(before, after)
    path = directory / "functions.py"
    path.write_text(text)
    config = CASES[f"heat_neumann_polynomial{dimensions}d"].config(device, 1, seed)
    config["domain"]["time"]["interval"] = [0, 1]
    config["network"].update(hidden_layers=[32, 32, 32], spatial_encoding="neumann_polynomial")
    config["training"].update(
        constraints="hard_initial_robin", hard_execution="eager", hard_background="general"
    )
    config["runtime"].update(metrics=False, progress=False)
    return SolverFactory(TaskSettings.read(config), load_problem(path), directory, directory)


def differentiate(value: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(value.sum(), coordinate, create_graph=True, retain_graph=True)[0]


def residuals(
    model: PINNModule, factory: SolverFactory, values: torch.Tensor, beta: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    columns = [values[:, i : i + 1].clone().requires_grad_() for i in range(values.shape[1])]
    spatial, clock = columns[:-1], columns[-1]
    field = model(spatial, clock)["u"]
    actual = factory.adapter.residuals({"u": field}, *columns)["heat"]
    # Control directly differentiates the physical field. No Equation cache,
    # provider derivative or implementation of the divergence is reused here.
    gradient = [differentiate(field, x) for x in spatial]
    laplacian = sum(differentiate(g, x) for g, x in zip(gradient, spatial))
    kappa = K0 * (1 + beta / len(spatial) * sum(spatial))
    forcing = torch.tensor(
        numpy_source(values.detach().cpu().double().numpy(), beta),
        dtype=values.dtype,
        device=values.device,
    )
    control = (
        differentiate(field, clock)
        - kappa * laplacian
        - K0 * beta / len(spatial) * sum(gradient)
        - forcing
    )
    return actual, control, field, gradient, kappa


@dataclass
class Audit:
    row: dict[str, Any]
    field: torch.Tensor
    residual: torch.Tensor
    gradient: torch.Tensor


def audit(
    directory: Path,
    dimensions: int,
    dtype: torch.dtype,
    beta: int,
    kind: str,
    seed: int,
    device: str = "cpu",
) -> Audit:
    factory = make_factory(directory, dimensions, beta, kind, seed, device)
    model = factory.build(predicting=True).model.to(device=device, dtype=dtype).eval()
    assert model.device.type == device
    assert all(p.device.type == device for p in model.parameters())
    values = points(dimensions, dtype).to(device)
    parameters = tuple(model.parameters())
    before = tensor_fingerprint(model.state_dict())
    parameter_sha = tensor_fingerprint(dict(model.named_parameters()))
    rtol, atol = (2e-4, 5e-6) if dtype == torch.float32 else (1e-10, 1e-11)
    actual, control, field, spatial_gradient, kappa = residuals(model, factory, values, beta)
    torch.testing.assert_close(actual, control, rtol=rtol, atol=atol)
    loss, reference_loss = actual.square().mean(), control.square().mean()
    assert float(loss.detach()) > 1e-5, "The gradient check must not use a solved zero residual"
    grads = torch.autograd.grad(loss, parameters, retain_graph=True)
    reference_grads = torch.autograd.grad(reference_loss, parameters, retain_graph=True)
    for (name, _), g, expected in zip(model.named_parameters(), grads, reference_grads):
        assert torch.isfinite(g).all(), f"Nonfinite gradient: {name}"
        torch.testing.assert_close(
            g, expected, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}"
        )
    flat, reference_flat = (
        torch.cat([g.flatten() for g in grads]),
        torch.cat([g.flatten() for g in reference_grads]),
    )
    assert float(flat.norm()) > 1e-5
    # Boundary flux has physical normal signs; zero flux also covers corners.
    flux_max = 0.0
    for axis, derivative in enumerate(spatial_gradient):
        for face, normal in ((0, -1), (1, 1)):
            mask = values[:, axis : axis + 1] == face
            flux_max = max(
                flux_max, float((kappa.mul(-normal) * derivative)[mask].detach().abs().max())
            )
    assert flux_max <= (2e-5 if dtype == torch.float32 else 1e-11)
    initial_mask = values[:, -1] == 0
    initial = factory.problem.initial(
        Coordinates(dict(zip("xy"[:dimensions], values[:, :-1].split(1, dim=1))), values[:, -1:])
    )["u"]
    initial_error = float((field[initial_mask] - initial[initial_mask]).detach().abs().max())
    assert initial_error <= (2e-6 if dtype == torch.float32 else 1e-11)
    direction_error: float | None = None
    directional_derivative: float | None = None
    if dtype == torch.float64:
        direction = torch.randn(
            flat.shape, dtype=dtype, generator=torch.Generator().manual_seed(815)
        ).to(device)
        direction /= direction.norm()
        originals = [p.detach().clone() for p in parameters]
        ends = list(itertools.accumulate(p.numel() for p in parameters))
        directions = [direction[a:b] for a, b in zip([0, *ends[:-1]], ends)]
        losses = []
        try:
            for sign in (-1, 1):
                with torch.no_grad():
                    for p, original, delta in zip(parameters, originals, directions):
                        p.copy_(original + sign * STEP * delta.reshape_as(p))
                r = residuals(model, factory, values, beta)[0]
                losses.append(float(r.square().mean().detach()))
        finally:
            with torch.no_grad():
                for p, original in zip(parameters, originals):
                    p.copy_(original)
        finite_difference = (losses[1] - losses[0]) / (2 * STEP)
        directional_derivative = float(torch.dot(flat, direction))
        direction_error = abs(finite_difference - directional_derivative)
        assert direction_error <= 5e-7 + 2e-5 * abs(directional_derivative)
    assert tensor_fingerprint(model.state_dict()) == before
    difference = flat - reference_flat
    return Audit(
        dict(
            dimensions=dimensions,
            dtype=str(dtype),
            beta=beta,
            representation=kind,
            seed=seed,
            points=len(values),
            parameters=flat.numel(),
            parameter_sha256=parameter_sha,
            state_unchanged=True,
            loss=float(loss.detach()),
            gradient_l2=float(flat.norm()),
            residual_control_max_abs=float((actual - control).detach().abs().max()),
            gradient_control_max_abs=float(difference.abs().max()),
            gradient_control_relative_l2=float(difference.norm() / reference_flat.norm()),
            initial_max_abs=initial_error,
            physical_flux_max_abs=flux_max,
            directional_derivative=directional_derivative,
            directional_absolute_error=direction_error,
        ),
        field.detach(),
        actual.detach(),
        flat.detach(),
    )


def audit_pair(
    directory: Path,
    dimensions: int,
    dtype: torch.dtype,
    beta: int,
    seed: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    formula, table = [
        audit(directory / kind, dimensions, dtype, beta, kind, seed, device)
        for kind in ("formula", "table")
    ]
    assert formula.row["parameter_sha256"] == table.row["parameter_sha256"]
    assert torch.equal(formula.field, table.field)
    rtol, atol = (2e-4, 5e-6) if dtype == torch.float32 else (1e-10, 1e-11)
    torch.testing.assert_close(table.residual, formula.residual, rtol=rtol, atol=atol)
    torch.testing.assert_close(table.gradient, formula.gradient, rtol=rtol, atol=atol)
    for result in (formula, table):
        result.row.update(
            pair_residual_max_abs=float((table.residual - formula.residual).abs().max()),
            pair_gradient_max_abs=float((table.gradient - formula.gradient).abs().max()),
        )
    return [formula.row, table.row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = None
        for dimensions, dtype, beta, seed in itertools.product(
            (1, 2), (torch.float32, torch.float64), (0, 1), (19, 1234, 4321)
        ):
            label = f"{dimensions}d-{dtype}-beta{beta}-seed{seed}"
            pair = audit_pair(args.output / label, dimensions, dtype, beta, seed)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(pair[0]), lineterminator="\n")
                writer.writeheader()
            writer.writerows(pair)
            stream.flush()
            rows.extend(pair)
    summary = dict(
        device="cpu",
        torch=torch.__version__,
        cases=len(rows),
        passed=True,
        training_runs=0,
        optimizer_steps=0,
        gpu_jobs=0,
        finite_difference_step=STEP,
        finite_difference_checks=sum(r["directional_absolute_error"] is not None for r in rows),
        all_states_unchanged=all(r["state_unchanged"] for r in rows),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
