"""Terminal AD targets preserve PDE graphs, weight gradients and optimizer updates."""

import os
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.refinement import FixedObjective, LBFGSRefinement


def build(
    directory: Path, case_name: str, constraints: str, lifting: str = "recursive"
) -> tuple[SolverFactory, LBFGSRefinement]:
    directory.mkdir(exist_ok=True)
    case = CASES[case_name]
    config = case.config(os.environ.get("PINN_TEST_DEVICE", "cpu"), 3, 19)
    config["network"]["hidden_layers"] = [5, 5]
    config["training"].update(constraints=constraints, hard_lifting=lifting)
    config["training"]["collocation"]["points"] = 7
    (directory / "functions.py").write_bytes(case.source.read_bytes())
    torch.manual_seed(19)
    factory = SolverFactory(
        TaskSettings.read(config), load_problem(directory / "functions.py"), directory
    )
    objects = factory.build(predicting=False)
    objects.data.setup("fit")
    objects.model.to(config["runtime"]["device"])
    objects.model.function_mapping = objects.data.function_mapping
    stage = LBFGSRefinement(
        objects.model, FixedObjective.capture(objects.data, objects.model.device), 1, 5, 1e-7, 1e-9
    )
    return factory, stage


@pytest.mark.parametrize(
    "case_name,constraints,lifting",
    [
        ("burgers", "soft", "recursive"),
        ("heat2d", "hard_initial_dirichlet", "recursive"),
        ("heat_robin", "hard_initial_robin", "recursive"),
        ("heat_neumann2d", "hard_initial_robin", "expanded"),
        ("heat_mixed3d", "hard_initial_robin", "recursive"),
        ("heat_robin3d", "hard_initial_robin", "expanded"),
    ],
)
def test_parameter_targets_preserve_gradient_and_updates(
    tmp_path: Path, case_name: str, constraints: str, lifting: str
) -> None:
    _, stage = build(tmp_path, case_name, constraints, lifting)
    model = stage.model
    initial_state = {key: value.clone() for key, value in model.state_dict().items()}
    observations = []
    for mode in ("all", "parameters"):
        model.load_state_dict(initial_state)
        model.hparams["backward_mode"] = mode
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            batch = stage.objective.fresh()
            loss, _ = model.model_step(batch)
            assert all(
                column.requires_grad
                for columns, clock, _ in batch.values()
                for column in [*columns, clock]
            )
            model.backward(loss)
            grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
            coordinate_grads = [
                column.grad for columns, clock, _ in batch.values() for column in [*columns, clock]
            ]
            if mode == "parameters":
                assert all(g is None for g in coordinate_grads)
            else:
                assert any(g is not None for g in coordinate_grads)
            optimizer.step()
        observations.append(
            (loss.detach().clone(), grads, [p.detach().clone() for p in model.parameters()])
        )
    for before, after in zip(observations[0], observations[1]):
        if isinstance(before, list):
            for a, b in zip(before, after):
                if a is None:
                    assert b is None
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    # Repeated line-search closures must use the same policy and fixed coordinates.
    states = []
    for mode in ("all", "parameters"):
        model.load_state_dict(initial_state)
        model.hparams["backward_mode"] = mode
        fresh_stage = LBFGSRefinement(model, stage.objective, 1, 5, 1e-7, 1e-9)
        first = fresh_stage.closure()
        torch.testing.assert_close(fresh_stage.closure(), first, rtol=0, atol=0)
        metrics = fresh_stage.run(2)
        states.append(
            (
                [p.detach().clone() for p in model.parameters()],
                metrics["lbfgs_closure_evaluations"],
            )
        )
    assert states[0][1] == states[1][1]
    for a, b in zip(states[0][0], states[1][0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_frozen_parameters_are_excluded_and_explicit_targets_are_respected(tmp_path: Path) -> None:
    _, stage = build(tmp_path, "heat_robin", "hard_initial_robin")
    model = stage.model
    model.hparams["backward_mode"] = "parameters"
    frozen = next(model.parameters())
    frozen.requires_grad_(False)
    model.backward(model.model_step(stage.objective.fresh())[0])
    assert frozen.grad is None
    assert any(p.grad is not None for p in model.parameters())
    batch = stage.objective.fresh()
    loss, _ = model.model_step(batch)
    clock = batch["equations"][1]
    model.backward(loss, inputs=(clock,))
    assert clock.grad is not None
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable parameters"):
        model.backward(torch.ones((), requires_grad=True))


def test_parameter_backward_config_and_backend_guards(tmp_path: Path) -> None:
    case = CASES["heat2d"]
    config = case.config("cpu", 3, 19)
    assert TaskSettings.read(config).training.backward == "all"
    config["training"]["backward"] = "invalid"
    with pytest.raises(ValueError, match="backward"):
        TaskSettings.read(config)
    config["training"]["backward"] = "parameters"
    source = tmp_path / "functions.py"
    source.write_bytes(case.source.read_bytes())
    for changes in ({"device": "mps"}, {"devices": 2}, {"precision": "bf16-mixed"}):
        invalid = case.config("cpu", 3, 19)
        invalid["training"]["backward"] = "parameters"
        invalid["runtime"].update(changes)
        factory = SolverFactory(TaskSettings.read(invalid), load_problem(source), tmp_path)
        with pytest.raises(ValueError, match="Parameter backward requires"):
            factory.build(predicting=False)
