"""Real L-BFGS trajectories and exact restoration around timing warmup."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from benchmarks.cases import CASES
from benchmarks.optimizer_timing import measure_steps, measure_total, preserved_state, summarize
from pinnstorch.models.net.spatial_features import FeatureFCN
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement
from tests.numerics.test_refinement import make_stage


def assert_same(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("backward", ["all", "parameters"])
def test_instrumentation_preserves_real_physical_stage(
    tmp_path: Path, hard: bool, backward: str
) -> None:
    reference = make_stage(tmp_path / "reference", hard, backward)
    measured = make_stage(tmp_path / "measured", hard, backward)
    reference.run(4)
    rows = measure_steps(measured.optimizer, measured.closure, 4, lambda: None)
    assert_same(reference.model.state_dict(), measured.model.state_dict())
    assert_same(reference.optimizer.state_dict(), measured.optimizer.state_dict())
    report = summarize(rows)
    assert report["closure_calls"] == measured.closures == reference.closures
    assert report["optimizer_evaluations"] == report["closure_calls"]
    assert report["closure_calls"] > report["steps"]  # Actual line-search reevaluations.
    assert 0 < report["closure_seconds"] <= report["seconds"]
    assert report["outside_closure_seconds"] >= 0
    assert report["instrumented"] is True
    bare = make_stage(tmp_path / "bare", hard, backward)
    total = measure_total(bare.optimizer, bare.closure, 4, lambda: None)
    assert_same(reference.model.state_dict(), bare.model.state_dict())
    assert_same(reference.optimizer.state_dict(), bare.optimizer.state_dict())
    assert total["closure_calls"] == reference.closures
    assert total["seconds"] > 0


@pytest.mark.parametrize("fail", [False, True])
def test_restore_populated_history_gradients_buffers_modes_and_rng(fail: bool) -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1)).double()
    model.register_buffer("sentinel", torch.tensor(3.0))
    model[0].eval()  # Preserve mixed module modes, not just the root flag.
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=1, line_search_fn="strong_wolfe", tolerance_grad=0
    )
    x = torch.tensor([[0.3, 0.7], [0.8, 0.1]], dtype=torch.float64)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - 0.4).square().mean()
        loss.backward()
        return loss.detach()

    measure_steps(optimizer, closure, 3, lambda: None)
    # Include both None and non-None gradient states.
    next(model.parameters()).grad = None
    weights = deepcopy(model.state_dict())
    history = deepcopy(optimizer.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    try:
        with preserved_state(model, optimizer):
            model.eval()
            model.sentinel.add_(7)
            torch.rand(5)
            measure_steps(optimizer, closure, 2, lambda: None)
            if fail:
                raise RuntimeError("interrupted warmup")
    except RuntimeError:
        assert fail
    assert_same(weights, model.state_dict())
    assert_same(history, optimizer.state_dict())
    assert_same(gradients, [p.grad for p in model.parameters()])
    assert modes == [m.training for m in model.modules()]
    assert_same(rng, torch.get_rng_state())


def test_restore_empty_optimizer_history_and_reject_foreign_parameters() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    with preserved_state(model, optimizer):
        optimizer.state[next(model.parameters())]["func_evals"] = 3
    assert not optimizer.state
    other = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="belong"):
        with preserved_state(other, optimizer):
            pass


@pytest.mark.parametrize("steps", [0, -1, True])
def test_reject_invalid_counts(steps: int) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    with pytest.raises(ValueError):
        measure_steps(optimizer, lambda: torch.tensor(0.0), steps, lambda: None)


def test_reject_nonfinite_closure_and_wrong_line_search() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    with pytest.raises(FloatingPointError):
        measure_steps(optimizer, lambda: torch.tensor(float("nan")), 1, lambda: None)
    optimizer = torch.optim.LBFGS(model.parameters())
    with pytest.raises(ValueError, match="strong_wolfe"):
        measure_steps(optimizer, lambda: torch.tensor(0.0), 1, lambda: None)
    with pytest.raises(ValueError):
        summarize([])


@pytest.mark.parametrize("kind", ["homogeneous", "polynomial"])
def test_actual_neumann_three_dimensional_stage_and_warmup_replay(
    tmp_path: Path, kind: str
) -> None:
    case = CASES[f"heat_neumann_{kind}3d"]
    config = case.config("cpu", 1, 19)
    config["network"].update(hidden_layers=[6, 6], spatial_encoding="neumann_polynomial")
    config["training"].update(
        constraints="hard_initial_robin",
        hard_execution="eager",
        backward="parameters",
        collocation={"sampling": "lhs", "points": 17},
    )
    for axis in config["domain"]["space"].values():
        axis["points"] = 3
    config["domain"]["time"]["points"] = 3
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(case.source), tmp_path, case.source.parent
    )
    objects = factory.build(predicting=False)
    objects.data.setup("fit")
    model = objects.model.double().eval()
    model.function_mapping = objects.data.function_mapping
    assert isinstance(model.net, FeatureFCN)
    generator = torch.Generator().manual_seed(19)
    points = torch.rand((17, 4), generator=generator, dtype=torch.float64)
    points[:, -1] *= config["domain"]["time"]["interval"][1]
    objective = FixedObjective(
        {"equations": ([points[:, i : i + 1] for i in range(3)], points[:, -1:], {})}
    )
    initial = deepcopy(model.state_dict())
    losses, parameters = [], []
    for rule in ("autograd", "analytic"):
        model.load_state_dict(initial)
        model.net.features = replace(model.net.features, derivative_rule=rule)
        warmup = LBFGSRefinement(model, objective, 1, 10, 1e-7, 1e-9)
        with preserved_state(model, warmup.optimizer):
            warmup.run(1)
        assert_same(initial, model.state_dict())
        assert not warmup.optimizer.state
        stage = LBFGSRefinement(model, objective, 1, 10, 1e-7, 1e-9)
        rows = measure_steps(stage.optimizer, stage.closure, 3, lambda: None)
        assert sum(r.closure_calls for r in rows) == stage.closures
        losses.append(stage.closure())
        parameters.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(parameters[0], parameters[1], rtol=1e-9, atol=1e-10)
