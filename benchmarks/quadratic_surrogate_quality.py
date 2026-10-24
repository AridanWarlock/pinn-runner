"""Независимая оценка общих весов по целым кодам квадратичного коэффициента."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.api import Equation
from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.problem import ProblemAdapter

from .catalog import SURROGATE_EVALUATIONS
from .fingerprints import tensor_fingerprint
from .quadratic_reference import reference_fields

Array = NDArray[np.float64]


def evaluation_identity(case: str = "heat_quadratic_shapes") -> dict[str, Any]:
    if case not in ("heat_quadratic_shapes", "heat_quadratic_clock"):
        raise ValueError("Expected registered quadratic evaluation")

    def digest(name: str) -> str:
        return hashlib.sha256(
            Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()

    return {
        **deepcopy(SURROGATE_EVALUATIONS[case]),
        "reference_sha256": digest("quadratic_surrogate_quality.py"),
        "analytical_sha256": digest("quadratic_reference.py"),
    }


def relative(actual: Array, expected: Array) -> float:
    denominator = float(np.linalg.norm(expected))
    if not denominator > 0:
        raise ValueError("Zero reference norm")
    return float(np.linalg.norm(actual - expected) / denominator)


def evaluate(
    model: PINNModule, adapter: ProblemAdapter, device: str, case: str = "heat_quadratic_shapes"
) -> tuple[dict[str, float], dict[str, Any]]:
    identity = evaluation_identity(case)
    before = tensor_fingerprint(model.state_dict())
    x = np.linspace(0, 1, identity["grid"][0])[:, None]
    t = np.linspace(0, 1, identity["grid"][1])[None, :]
    xx, tt = np.broadcast_arrays(x, t)
    generator = np.random.default_rng(identity["pde_seed"])
    probes = generator.uniform(size=(identity["pde_points"], 2))
    rows: list[dict[str, Any]] = []
    temporal: list[dict[str, float]] = []
    fields: list[Array] = []
    references: list[Array] = []
    faces = {"lower": 0.0, "upper": 0.0}

    def query(a: float, b: float, gradient: bool) -> tuple[Array, Array | None]:
        columns = [
            torch.tensor(
                v.reshape(-1, 1).copy(), dtype=torch.float32, device=device, requires_grad=gradient
            )
            for v in (xx, np.full(xx.shape, a), np.full(xx.shape, b), tt)
        ]
        values, derivatives = [], []
        for start in range(0, len(columns[0]), 2048):
            batch = [v[start : start + 2048] for v in columns]
            with torch.set_grad_enabled(gradient):
                output = model(batch[:3], batch[3])
                if gradient:
                    e = Equation(output, adapter.context(batch[:3], batch[3]))
                    derivatives.append(e.diff("u", "x").detach().cpu().numpy().astype(np.float64))
            values.append(output["u"].detach().cpu().numpy().astype(np.float64))
        value = cast(Array, np.concatenate(values).reshape(xx.shape))
        dx = cast(Array, np.concatenate(derivatives).reshape(xx.shape)) if gradient else None
        return value, dx

    for index, ((a, b), split) in enumerate(zip(identity["hypotheses"], identity["splits"])):
        actual, dx = query(a, b, True)
        assert dx is not None
        exact = reference_fields(x, t, a=a, b=b, modes=identity["reference_modes"])
        fields.append(actual)
        references.append(exact.temperature)
        k = a * (1 + x / b) ** 2
        time_error = np.linalg.norm(actual - exact.temperature, axis=0) / np.linalg.norm(
            exact.temperature, axis=0
        )
        columns = [
            torch.tensor(v.copy(), dtype=torch.float32, device=device, requires_grad=True)
            for v in (
                probes[:, :1],
                np.full((len(probes), 1), a),
                np.full((len(probes), 1), b),
                probes[:, 1:],
            )
        ]
        with torch.set_grad_enabled(True):
            output = model(columns[:3], columns[3])
            residual = adapter.residuals(output, *columns)["heat"]
        ignored_a = query(identity["ignored_a"], b, False)[0]
        ignored_b = query(a, identity["ignored_b"], False)[0]
        rows.append(
            dict(
                a=a,
                b=b,
                split=split,
                relative_l2=relative(actual, exact.temperature),
                final_relative_l2=float(time_error[-1]),
                worst_time_relative_l2=float(time_error.max()),
                worst_time=float(t[0, int(time_error.argmax())]),
                dx_relative_l2=relative(dx, exact.dx),
                flux_relative_l2=relative(-k * dx, -k * exact.dx),
                pde_rmse=float(residual.detach().square().mean().sqrt().cpu()),
                initial_rmse=float(
                    np.sqrt(np.mean((actual[:, 0] - exact.temperature[:, 0]) ** 2))
                ),
                boundary_max=float(np.abs(actual[[0, -1]]).max()),
                ignored_a_relative_l2=relative(ignored_a, exact.temperature),
                ignored_b_relative_l2=relative(ignored_b, exact.temperature),
            )
        )
        temporal.extend(
            dict(hypothesis_index=float(index), time=float(instant), relative_l2=float(error))
            for instant, error in zip(t[0], time_error)
        )
        for side, position in (("lower", 0), ("upper", -1)):
            faces[side] = max(faces[side], float(np.abs(actual[position]).max()))
    responses = [
        dict(
            left=left,
            right=right,
            relative_l2=relative(
                fields[right] - fields[left], references[right] - references[left]
            ),
        )
        for left, right in identity["response_pairs"]
    ]
    keys = (
        "relative_l2",
        "final_relative_l2",
        "worst_time_relative_l2",
        "dx_relative_l2",
        "flux_relative_l2",
        "pde_rmse",
        "initial_rmse",
    )
    metrics = {key: max(row[key] for row in rows) for key in keys}
    metrics["coefficient_response_relative_l2"] = max(row["relative_l2"] for row in responses)
    if not all(
        np.isfinite(value) for row in rows for key, value in row.items() if key != "split"
    ) or not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite whole-code metric")
    if tensor_fingerprint(model.state_dict()) != before:
        raise ValueError("Evaluation changed model state")
    return metrics, dict(
        hypotheses=rows,
        responses=responses,
        temporal=temporal,
        hypothesis_evaluation=identity,
        evaluation_state_unchanged=True,
        boundary_operators=[
            dict(field="u", kind="dirichlet", axis="x", side=side, max_error=value)
            for side, value in faces.items()
        ],
    )
