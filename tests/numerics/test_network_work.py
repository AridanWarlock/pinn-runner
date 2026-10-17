"""Verify the derived 3**d projected rows against actual nested network calls."""

from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.network_work import count_network_work
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("encoding", ["identity", "neumann_cosine", "neumann_polynomial"])
def test_neumann_projection_work_matches_combinatorial_derivation(
    tmp_path: Path, dimensions: int, encoding: str
) -> None:
    case = CASES[f"heat_neumann_homogeneous{dimensions}d"]
    config = case.config("cpu", 1, 19)
    config["network"]["spatial_encoding"] = encoding
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution="robin_grouped" if encoding == "identity" else "eager",
    )
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(case.source), tmp_path, case.source.parent
    )
    model = factory.build(predicting=True).model
    spatial = [torch.full((5, 1), 0.3, requires_grad=True) for _ in range(dimensions)]
    clock = torch.full((5, 1), 0.2, requires_grad=True)
    work = count_network_work(model, lambda: model(spatial, clock))
    assert work.calls == (2**dimensions if encoding == "identity" else 1)
    assert work.rows == 5 * (3**dimensions if encoding == "identity" else 1)
