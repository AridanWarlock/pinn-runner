"""Resolved guarantees and objective agree for each supported execution adapter."""

from pathlib import Path

import pytest

from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.hard_execution import FusedPredictHard, ProductRuleHard


@pytest.mark.parametrize("backend", ["eager", "fused_predict", "product_rule"])
def test_hard_plan_guarantees_do_not_depend_on_execution(tmp_path: Path, backend: str) -> None:
    case = CASES["heat3d"]
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    config = case.config("cpu", 3, 19)
    config["training"].update(constraints="hard_initial_dirichlet", hard_execution=backend)
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(tmp_path / "functions.py"), tmp_path
    )
    plan = factory.constraints
    assert plan.exact_initial == ("u",)
    assert len(plan.exact_faces) == 6
    assert {(face.axis, face.side) for face in plan.exact_faces} == {
        (axis, side) for axis in ("x", "y", "z") for side in ("lower", "upper")
    }
    assert plan.remaining_losses == ("equations",)
    objects = factory.build(predicting=False)
    assert tuple(objects.data.condition_names) == plan.remaining_losses
    assert objects.model.output_fn is plan.output
    if isinstance(plan.output, (FusedPredictHard, ProductRuleHard)):
        assert plan.output.recipe is plan.recipe


def test_soft_plan_has_no_exact_conditions(tmp_path: Path) -> None:
    case = CASES["heat2d"]
    factory = SolverFactory(
        TaskSettings.read(case.config("cpu", 3, 19)), load_problem(case.source), tmp_path
    )
    assert factory.constraints.exact_initial == factory.constraints.exact_faces == ()
    assert factory.constraints.remaining_losses == ("initial", "equations", "boundaries")
    assert factory.constraints.output is None
    assert factory.constraints.recipe is None
