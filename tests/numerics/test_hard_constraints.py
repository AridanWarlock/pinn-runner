"""Hard constraint values, coordinate derivatives and gradient flow."""

from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from pinnstorch.contracts import TensorOutputs
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.constraints import HardBoxInitial
from pinnstorch.training.hard_execution import FusedPredictHard


def initial(spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
    x = spatial[0]
    return {"u": x.mul(x.neg().add(1))}


def test_hard_values_and_second_derivatives_allow_parameter_gradients() -> None:
    transform = HardBoxInitial([(0, 1)], (0, 1), initial)
    x = torch.tensor([[0.0], [0.3], [1.0]], dtype=torch.float64, requires_grad=True)
    t = torch.full_like(x, 0.5, requires_grad=True)
    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    output = transform({"u": torch.ones_like(x).mul(parameter)}, x, t)["u"]
    assert output[0].item() == output[-1].item() == 0
    at_initial = transform({"u": torch.ones_like(x).mul(parameter)}, x, torch.zeros_like(t))["u"]
    torch.testing.assert_close(at_initial, initial([x], t)["u"], atol=0, rtol=0)
    ux = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    torch.testing.assert_close(uxx, torch.full_like(x, -10), atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(uxx.sum(), parameter)[0]
    assert gradient.item() == pytest.approx(-12)


@pytest.mark.parametrize("invalid", ["neumann", "missing_face", "incompatible_initial"])
def test_incompatible_hard_constraints_are_rejected(tmp_path: Path, invalid: str) -> None:
    case = CASES["heat2d"]
    config = case.config("cpu", 3, 19)
    config["training"]["constraints"] = "hard_initial_dirichlet"
    source = case.source.read_text()
    if invalid == "neumann":
        config["boundaries"][0]["kind"] = "neumann"
    elif invalid == "missing_face":
        config["boundaries"][0]["side"] = "lower"
    else:
        source = source.replace('return {"u":', 'return {"u": 1 +')
    (tmp_path / "functions.py").write_text(source)
    with pytest.raises(ValueError, match="Hard constraints|Initial values"):
        SolverFactory(TaskSettings.read(config), load_problem(tmp_path / "functions.py"), tmp_path)


def test_hard_checkpoint_rejects_changed_physical_transform(tmp_path: Path) -> None:
    import logging

    from pinnstorch.runner.artifacts import ArtifactWriter
    from pinnstorch.runner.execution import SolverExecution

    case = CASES["heat2d"]
    (tmp_path / "functions.py").write_bytes(case.source.read_bytes())
    config = case.config("cpu", 3, 19)
    config["training"]["constraints"] = "hard_initial_dirichlet"
    settings = TaskSettings.read(config)
    problem = load_problem(tmp_path / "functions.py")
    objects = SolverFactory(settings, problem, tmp_path).build(True)
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": dict(objects.model.hparams),
            "state_dict": objects.model.state_dict(),
        },
        checkpoint,
    )
    execution = SolverExecution(
        settings, objects, ArtifactWriter(tmp_path, settings, problem), logging.getLogger("test")
    )
    execution._load(checkpoint, False)
    config["training"]["hard_execution"] = "fused_predict"
    fused = TaskSettings.read(config)
    fused_objects = SolverFactory(fused, problem, tmp_path).build(True)
    SolverExecution(
        fused, fused_objects, ArtifactWriter(tmp_path, fused, problem), logging.getLogger("test")
    )._load(checkpoint, False)
    for name, value in objects.model.state_dict().items():
        torch.testing.assert_close(fused_objects.model.state_dict()[name], value, rtol=0, atol=0)
    config["training"]["hard_execution"] = "eager"
    config["training"]["constraints"] = "soft"
    soft = TaskSettings.read(config)
    soft_objects = SolverFactory(soft, problem, tmp_path).build(True)
    with pytest.raises(ValueError, match="constraint"):
        SolverExecution(
            soft, soft_objects, ArtifactWriter(tmp_path, soft, problem), logging.getLogger("test")
        )._load(checkpoint, False)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_predict_preserves_values_and_training_graph(
    dimension: int, dtype: torch.dtype
) -> None:
    bounds = [(float(i - 2), float(2 * i + 1)) for i in range(dimension)]
    axes = [
        torch.tensor([left, left + (right - left) * 1e-6, (left + right) / 2, right], dtype=dtype)
        for left, right in bounds
    ]
    grid = torch.cartesian_prod(*axes, torch.tensor([2.0, 2.1, 4.0, 5.0], dtype=dtype))
    columns = [column.detach().requires_grad_() for column in grid.split(1, dim=1)]
    coordinates, time = columns[:-1], columns[-1]

    def shifted_initial(spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        field = torch.ones_like(time)
        for column, (left, right) in zip(spatial, bounds):
            field = field * (column - left) * (right - column)
        return {"u": field}

    eager = HardBoxInitial(bounds, (2, 5), shifted_initial)
    candidate = FusedPredictHard(eager)
    weight = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    outputs = {"u": weight * torch.sin(sum(coordinates) + time)}
    reference = eager(outputs, *coordinates, time)["u"]
    actual = candidate(outputs, *coordinates, time)["u"]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    first = torch.autograd.grad(actual.sum(), coordinates + [time], create_graph=True)
    laplacian = sum(
        torch.autograd.grad(part.sum(), coordinate, create_graph=True, retain_graph=True)[0]
        for part, coordinate in zip(first[:-1], coordinates)
    )
    residual = first[-1] - 0.1 * laplacian
    assert torch.isfinite(torch.autograd.grad(residual.square().sum(), weight)[0])
    with torch.no_grad():
        actual_predict = candidate(outputs, *coordinates, time)["u"]
        expected_predict = eager(outputs, *coordinates, time)["u"]
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual_predict, expected_predict, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_predict[[0, -1]], expected_predict[[0, -1]], rtol=0, atol=0)


def test_fused_predict_requires_hard_configuration() -> None:
    config = CASES["heat2d"].config("cpu", 3, 19)
    config["training"]["hard_execution"] = "fused_predict"
    with pytest.raises(ValueError, match="requires hard"):
        TaskSettings.read(config)
    config["training"]["constraints"] = "hard_initial_dirichlet"
    assert TaskSettings.read(config).training.hard_execution == "fused_predict"
