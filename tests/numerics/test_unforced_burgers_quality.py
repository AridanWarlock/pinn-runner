"""Explicit smooth fields test independent field, gradient and regional metrics."""

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.burgers_quality import LayerEvaluation
from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


class SmoothField(torch.nn.Module):
    def forward(self, spatial, time):
        return {"u": -torch.exp(-time) * torch.sin(torch.pi * spatial[0])}


@pytest.mark.parametrize(
    "nu,name", [(0.1, "burgers_unforced_nu010"), (0.01, "burgers_unforced_nu001")]
)
def test_layer_metrics_match_independent_explicit_derivatives(
    tmp_path: Path, nu: float, name: str
) -> None:
    case = CASES[name]
    factory = SolverFactory(
        TaskSettings.read(case.config("cpu", 1, 11)),
        load_problem(case.source),
        tmp_path,
        case.source.parent,
    )
    control = LayerEvaluation(nu, (21, 11))
    limits = dict(
        relative_l2=0.002,
        final_relative_l2=0.003,
        worst_time_relative_l2=0.005,
        worst_time_gradient_relative_l2=0.02,
    )
    actual, temporal = control.evaluate(SmoothField(), factory.adapter, limits, "cpu")
    x, t = np.meshgrid(control.x, control.t, indexing="ij")
    u = -np.exp(-t) * np.sin(np.pi * x)
    ux = -np.pi * np.exp(-t) * np.cos(np.pi * x)
    gradient_error = np.sqrt(
        np.trapz((ux - control.slope) ** 2, control.x, axis=0)
        / np.trapz(control.slope**2, control.x, axis=0)
    )
    assert actual["worst_time_gradient_relative_l2"] == pytest.approx(
        gradient_error.max(), rel=2e-6
    )
    assert actual["final_relative_l2"] == pytest.approx(
        np.linalg.norm(u[:, -1] - control.field[:, -1]) / np.linalg.norm(control.field[:, -1]),
        rel=2e-6,
    )
    points = np.random.default_rng(20261014).uniform(size=(2048, 2))
    x = 0.2 * points[:, 0] - 0.1
    t = 0.15 + 0.85 * points[:, 1]
    residual = np.exp(-t) * (1 - nu * np.pi**2) * np.sin(np.pi * x) + np.pi * np.exp(
        -2 * t
    ) * np.sin(np.pi * x) * np.cos(np.pi * x)
    assert actual["regional_pde_rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)), rel=2e-6)
    assert len(temporal) == 21 and actual["initial_rmse"] < 2e-7
    assert actual["grid_refinement_ratio"] > 0
