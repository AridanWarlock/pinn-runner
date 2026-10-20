"""Physical-autograd layer control, separate from the training objective."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.problem import ProblemAdapter

from .burgers import grid_reference
from .catalog import UNFORCED_EVALUATION


def evaluation_identity(shape: tuple[int, int] | None = None) -> dict[str, Any]:
    result = dict(
        deepcopy(UNFORCED_EVALUATION),
        reference_sha256=hashlib.sha256(
            Path(__file__).with_name("burgers.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )
    if shape is not None:
        result.update(coarse_grid=list(shape), fine_grid=[2 * n - 1 for n in shape])
    return result


def field_metrics(
    values: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]
) -> dict[str, float]:
    if values.shape != expected.shape or not np.isfinite(values).all():
        raise ValueError("Finite field with the reference shape required")
    errors = np.linalg.norm(values - expected, axis=0) / np.linalg.norm(expected, axis=0)
    return dict(
        relative_l2=float(np.linalg.norm(values - expected) / np.linalg.norm(expected)),
        final_relative_l2=float(errors[-1]),
        worst_time_relative_l2=float(errors.max()),
    )


class LayerEvaluation:
    def __init__(self, viscosity: float, shape: tuple[int, int]) -> None:
        if any(type(n) is not int or n < 3 for n in shape):
            raise ValueError("At least three nodes per physical coordinate required")
        self.viscosity = viscosity
        self.coarse = shape
        self.shape = (2 * shape[0] - 1, 2 * shape[1] - 1)
        self.x = np.linspace(-1, 1, self.shape[0])
        self.t = np.linspace(0, 1, self.shape[1])
        self.field = grid_reference(tuple(self.x), tuple(self.t), viscosity)
        self.slope = grid_reference(tuple(self.x), tuple(self.t), viscosity, True)

    def evaluate(
        self, model: PINNModule, adapter: ProblemAdapter, limits: dict[str, float], device: str
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        xx, tt = np.meshgrid(self.x, self.t, indexing="ij")
        values, slopes = [], []
        for start in range(0, xx.size, 2048):
            x, t = [
                torch.tensor(
                    a.reshape(-1)[start : start + 2048, None],
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )
                for a in (xx, tt)
            ]
            with torch.set_grad_enabled(True):
                u = model([x], t)["u"]
                ux = torch.autograd.grad(u, x, torch.ones_like(u))[0]
            values.append(u.detach().cpu().numpy())
            slopes.append(ux.detach().cpu().numpy())
        prediction = np.concatenate(values).astype(np.float64).reshape(self.shape)
        gradient = np.concatenate(slopes).astype(np.float64).reshape(self.shape)
        fine = field_metrics(prediction, self.field)
        coarse = field_metrics(prediction[::2, ::2], self.field[::2, ::2])
        gradient_error = np.sqrt(
            np.trapz((gradient - self.slope) ** 2, self.x, axis=0)
            / np.trapz(self.slope**2, self.x, axis=0)
        )
        coarse_gradient_error = np.sqrt(
            np.trapz((gradient[::2, ::2] - self.slope[::2, ::2]) ** 2, self.x[::2], axis=0)
            / np.trapz(self.slope[::2, ::2] ** 2, self.x[::2], axis=0)
        )
        key = "worst_time_gradient_relative_l2"
        fine[key], coarse[key] = float(gradient_error.max()), float(coarse_gradient_error.max())
        fine["grid_refinement_ratio"] = max(abs(fine[k] - coarse[k]) / limits[k] for k in fine)
        points = np.random.default_rng(20261014).uniform(size=(2048, 2))
        points[:, 0] = 0.2 * points[:, 0] - 0.1
        points[:, 1] = 0.15 + 0.85 * points[:, 1]
        columns = [
            torch.tensor(
                points[:, i : i + 1], dtype=torch.float32, device=device, requires_grad=True
            )
            for i in range(2)
        ]
        with torch.set_grad_enabled(True):
            physical = dict(model(columns[:1], columns[1]))
            residual = adapter.residuals(physical, *columns)["burgers"]
        fine["regional_pde_rmse"] = float(residual.detach().square().mean().sqrt().cpu())
        fine["initial_rmse"] = float(np.sqrt(np.mean((prediction[:, 0] - self.field[:, 0]) ** 2)))
        time_errors = np.linalg.norm(prediction - self.field, axis=0) / np.linalg.norm(
            self.field, axis=0
        )
        temporal = [
            dict(
                time=float(t),
                relative_l2=float(time_errors[j]),
                gradient_relative_l2=float(gradient_error[j]),
            )
            for j, t in enumerate(self.t)
        ]
        if not all(np.isfinite(v) for v in fine.values()):
            raise ValueError("Nonfinite layer metric")
        return fine, temporal
