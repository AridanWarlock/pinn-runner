"""Independent analytical slices for the first shared constant-k surrogate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.problem import ProblemAdapter

from .catalog import CONSTANT_EVALUATION
from .fingerprints import tensor_fingerprint


def evaluation_identity() -> dict[str, Any]:
    return {
        **CONSTANT_EVALUATION,
        "reference_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    }


def reference(x: NDArray[np.float64], t: NDArray[np.float64], k: float) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.sin(np.pi * x) * np.exp(-(np.pi**2) * k * t))


def evaluate(
    model: PINNModule, adapter: ProblemAdapter, device: str
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate each fixed hypothesis; never pool errors before acceptance."""
    identity = evaluation_identity()
    before = tensor_fingerprint(model.state_dict())
    x = np.linspace(0, 1, identity["grid"][0], dtype=np.float64)[:, None]
    t = np.linspace(0, 1, identity["grid"][1], dtype=np.float64)[None, :]
    xx, tt = np.broadcast_arrays(x, t)
    generator = np.random.default_rng(identity["pde_seed"])
    probes = generator.uniform(size=(identity["pde_points"], 2))
    rows: list[dict[str, float]] = []
    temporal: list[dict[str, float]] = []
    values_by_k: dict[float, NDArray[np.float64]] = {}
    face_maxima = {"lower": 0.0, "upper": 0.0}

    def field(k: float) -> NDArray[np.float64]:
        columns = [
            torch.tensor(v.reshape(-1, 1).copy(), dtype=torch.float32, device=device)
            for v in (xx, np.full(xx.shape, k), tt)
        ]
        chunks = []
        for start in range(0, len(columns[0]), 2048):
            with torch.no_grad():
                output = model(
                    [v[start : start + 2048] for v in columns[:2]],
                    columns[2][start : start + 2048],
                )
            chunks.append(output["u"].detach().cpu().numpy().astype(np.float64))
        return cast(NDArray[np.float64], np.concatenate(chunks).reshape(xx.shape))

    ignored = field(identity["ignored_k"])
    for k in identity["hypotheses"]:
        actual = field(k)
        values_by_k[k] = actual
        for side, index in (("lower", 0), ("upper", -1)):
            face_maxima[side] = max(face_maxima[side], float(np.abs(actual[index]).max()))
        expected = reference(x, t, k)
        error = actual - expected
        temporal_relative = np.linalg.norm(error, axis=0) / np.linalg.norm(expected, axis=0)
        columns = [
            torch.tensor(v, dtype=torch.float32, device=device, requires_grad=True)
            for v in (probes[:, :1], np.full((len(probes), 1), k), probes[:, 1:])
        ]
        with torch.set_grad_enabled(True):
            outputs = model(columns[:2], columns[2])
            residual = adapter.residuals(outputs, *columns)["heat"]
        row = {
            "k": k,
            "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(expected)),
            "final_relative_l2": float(temporal_relative[-1]),
            "worst_time_relative_l2": float(temporal_relative.max()),
            "pde_rmse": float(residual.detach().square().mean().sqrt().cpu()),
            "initial_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "boundary_max": float(np.abs(actual[[0, -1]]).max()),
            "ignored_k_relative_l2": float(
                np.linalg.norm(ignored - expected) / np.linalg.norm(expected)
            ),
        }
        rows.append(row)
        temporal.extend(
            {"k": k, "time": float(instant), "relative_l2": float(relative)}
            for instant, relative in zip(t[0], temporal_relative)
        )
    # A field independent of k gives exactly 1 for this response metric.
    lo, hi = identity["hypotheses"][0], identity["hypotheses"][-1]
    expected_response = reference(x, t, hi) - reference(x, t, lo)
    response_error = values_by_k[hi] - values_by_k[lo] - expected_response
    metrics = {
        key: max(row[key] for row in rows)
        for key in (
            "relative_l2",
            "final_relative_l2",
            "worst_time_relative_l2",
            "pde_rmse",
            "initial_rmse",
        )
    }
    metrics["coefficient_response_relative_l2"] = float(
        np.linalg.norm(response_error) / np.linalg.norm(expected_response)
    )
    if not all(np.isfinite(value) for row in rows for value in row.values()) or not all(
        np.isfinite(v) for v in metrics.values()
    ):
        raise ValueError("Nonfinite per-hypothesis surrogate metric")
    if tensor_fingerprint(model.state_dict()) != before:
        raise ValueError("Evaluation modified shared-model weights")
    return metrics, {
        "hypotheses": rows,
        "hypothesis_evaluation": identity,
        "evaluation_state_unchanged": True,
        "temporal": temporal,
        "boundary_operators": [
            dict(
                field="u",
                kind="dirichlet",
                axis="x",
                side=side,
                max_error=face_maxima[side],
            )
            for side in ("lower", "upper")
        ],
    }
