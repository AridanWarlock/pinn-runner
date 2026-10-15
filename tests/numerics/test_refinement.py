"""L-BFGS closure stability, real derivative objective, and stage restoration."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import Reader, Refinement, TaskSettings, Training
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement


def make_stage(directory: Path, hard: bool = False, backward: str = "all") -> LBFGSRefinement:
    case = CASES["burgers"]
    config = case.config("cpu", 3, 19)
    config["network"]["hidden_layers"] = [8, 8]
    config["training"]["collocation"]["points"] = 32
    config["training"]["backward"] = backward
    if hard:
        config["training"]["constraints"] = "hard_initial_dirichlet"
    directory.mkdir(exist_ok=True)
    (directory / "functions.py").write_bytes(case.source.read_bytes())
    torch.manual_seed(19)
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(directory / "functions.py"), directory
    )
    objects = factory.build(predicting=False)
    objects.data.setup("fit")
    objects.model.function_mapping = objects.data.function_mapping
    return LBFGSRefinement(
        objects.model,
        FixedObjective.capture(objects.data, objects.model.device),
        1,
        10,
        1e-7,
        1e-9,
    )


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("backward", ["all", "parameters"])
def test_fixed_objective_derivatives_and_exact_stage_restore(
    tmp_path: Path, hard: bool, backward: str
) -> None:
    stage = make_stage(tmp_path / "original", hard, backward)
    before = stage.closure().item()
    gradients = [p.grad.clone() for p in stage.model.parameters() if p.grad is not None]
    assert stage.closure().item() == before
    for expected, parameter in zip(gradients, stage.model.parameters()):
        torch.testing.assert_close(parameter.grad, expected, atol=0, rtol=0)
    snapshot = [c.clone() for c in stage.objective.batches["equations"][0]]
    metrics = stage.run(4)
    state = deepcopy(stage.state_dict(metrics))
    model_state = deepcopy(stage.model.state_dict())
    stage.run(3)
    restored = make_stage(
        tmp_path / "restored", hard, "parameters" if backward == "all" else "all"
    )
    restored.model.load_state_dict(model_state)
    restored.load_state_dict(state)
    restored_metrics = restored.run(3)
    for expected, observed in zip(stage.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    assert restored_metrics["lbfgs_steps"] == 7
    assert restored.closures == stage.closures
    assert restored.closure().item() < before
    for expected, observed in zip(snapshot, stage.objective.batches["equations"][0]):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    assert (
        restored_metrics["lbfgs_residual_points"]
        == restored_metrics["lbfgs_closure_evaluations"] * 32
    )


@pytest.mark.parametrize(
    "options",
    [
        {"steps": -1},
        {"history_size": 0},
        {"learning_rate": 0},
        {"tolerance_grad": float("nan")},
        {"steps": True},
    ],
)
def test_invalid_refinement_settings(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Refinement.read(Reader(options, "refinement"))


def test_stage_requires_explicit_adam_budget_and_matching_objective() -> None:
    for options in (
        {"refinement": {"steps": 2}},
        {"steps": 3, "refinement": {"steps": 2}, "optimizer": {"weight_decay": 0.1}},
    ):
        with pytest.raises(ValueError, match="refinement requires"):
            Training.read(Reader(options, "training"))
