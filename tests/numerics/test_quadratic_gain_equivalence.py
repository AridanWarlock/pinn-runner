"""Масштабирование последней карты сохраняет поле и физические производные."""

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.fingerprints import tensor_fingerprint
from pinnstorch.api import Equation
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gain8_preserves_function_space_under_last_affine_mapping(tmp_path: Path, dtype):
    case = CASES["heat_quadratic_shapes"]
    factory = SolverFactory(
        TaskSettings.read(case.config("cpu", 1, 11)),
        load_problem(case.source),
        tmp_path,
        case.source.parent,
    )
    model = factory.build(predicting=False).model.to(dtype=dtype).eval()
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fingerprint = tensor_fingerprint(saved)
    rng = np.random.default_rng(11)
    query = np.column_stack(
        (
            rng.uniform(0.01, 0.99, 37),
            rng.uniform(0.05, 0.15, 37),
            rng.uniform(1, 2, 37),
            rng.uniform(0.01, 0.99, 37),
        )
    )

    def fields(mapped):
        x, a, b, t = [
            torch.tensor(query[:, i : i + 1], dtype=dtype, requires_grad=True) for i in range(4)
        ]
        if mapped:
            output = model([x, a, b], t)
        else:
            raw = model.net([x, a, b], t)["raw"]
            output = {"u": torch.sin(torch.pi * x) ** 3 + t * x * (1 - x) * raw}
        e = Equation(output, factory.adapter.context([x, a, b], t))
        return [
            value.detach().clone()
            for value in (
                output["u"],
                e.diff("u", "x"),
                e.diff("u", "x", 2),
                e.dt("u"),
                factory.adapter.residuals(output, x, a, b, t)["heat"],
            )
        ]

    baseline = fields(False)
    with torch.no_grad():
        model.net.model.output.weight.div_(8)
        model.net.model.output.bias.div_(8)
    mapped = fields(True)
    atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (2e-13, 2e-12)
    for actual, expected in zip(mapped, baseline):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    model.load_state_dict(saved)
    assert tensor_fingerprint(model.state_dict()) == fingerprint
