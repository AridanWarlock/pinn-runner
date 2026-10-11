"""Intersection ownership and normal derivative signs for explicit conditions."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml

from pinnstorch.contracts import SpatialTensors, TensorOutputs
from pinnstorch.data import BoundaryConditions, PointCloud, create_interval_data
from tests.integration.test_runner_files import launch, prepare_input


def mesh(dimension: int = 2, count: int = 3) -> PointCloud:
    kwargs = {"y_interval": [0, 1], "y_points": count} if dimension >= 2 else {}
    if dimension == 3:
        kwargs.update(z_interval=[0, 1], z_points=count)
    return PointCloud(lambda: create_interval_data([0, 1], [0, 1], count, 2, **kwargs))


def condition(kind: str, axis: str, **kwargs: object) -> dict[str, object]:
    return {"kind": kind, "axis": axis, "side": "both", "field": "T", **kwargs}


def loss(
    dataset: BoundaryConditions, forward: Callable[[SpatialTensors, torch.Tensor], TensorOutputs]
) -> torch.Tensor:
    columns, time, targets = dataset[:]
    columns = [column.requires_grad_() for column in columns]
    time = time.requires_grad_()
    dataset.functions = {"forward": forward}
    result, _ = dataset._loss_fn((columns, time, targets), 0.0)
    assert isinstance(result, torch.Tensor)
    return result


@pytest.mark.parametrize("dimension, expected", [(2, 8), (3, 26)])
def test_dirichlet_union(dimension: int, expected: int) -> None:
    dataset = BoundaryConditions(
        mesh(dimension), [condition("dirichlet", axis) for axis in ("x", "y", "z")[:dimension]]
    )
    assert len(dataset) == expected * 2
    columns = torch.cat(dataset.spatial_domain_sampled + [dataset.time_domain_sampled], dim=1)
    assert len(torch.unique(columns, dim=0)) == len(dataset)
    torch.testing.assert_close(
        loss(dataset, lambda columns, time: {"T": columns[0] * 0 + 2}), torch.tensor(4.0)
    )


def test_conflicting_dirichlet_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting Dirichlet"):
        BoundaryConditions(
            mesh(), [condition("dirichlet", "x", value=1), condition("dirichlet", "y", value=2)]
        )


@pytest.mark.parametrize("dimension, expected", [(1, 4), (2, 8), (3, 12)])
def test_neumann_excludes_intersections(dimension: int, expected: int) -> None:
    dataset = BoundaryConditions(
        mesh(dimension), [condition("neumann", axis) for axis in ("x", "y", "z")[:dimension]]
    )
    assert len(dataset) == expected
    points = torch.cat(dataset.spatial_domain_sampled, dim=1)
    assert torch.all(((points == 0) | (points == 1)).sum(dim=1) == 1)


def test_normal_signs_and_gradients() -> None:
    def target(columns: SpatialTensors, time: torch.Tensor) -> torch.Tensor:
        return torch.where(columns[0] == 0, -1.0, 1.0)

    dataset = BoundaryConditions(
        mesh(), [condition("neumann", "x", value_fn="normal")], {"normal": target}
    )
    scale = torch.tensor(2.0, requires_grad=True)
    result = loss(dataset, lambda columns, time: {"T": scale * columns[0]})
    torch.testing.assert_close(result, torch.tensor(1.0))
    result.backward()
    torch.testing.assert_close(scale.grad, torch.tensor(2.0))
    torch.testing.assert_close(
        loss(dataset, lambda columns, time: {"T": columns[0]}), torch.tensor(0.0)
    )


def test_robin_and_flux() -> None:
    def robin_value(columns: SpatialTensors, time: torch.Tensor) -> torch.Tensor:
        return 2 * columns[0] + 3 * torch.where(columns[0] == 0, -1.0, 1.0)

    dataset = BoundaryConditions(
        mesh(), [condition("robin", "x", alpha=2, beta=3, value_fn="r")], {"r": robin_value}
    )
    torch.testing.assert_close(
        loss(dataset, lambda columns, time: {"T": columns[0]}), torch.tensor(0.0)
    )
    flux = BoundaryConditions(
        mesh(), [{**condition("heat_flux", "x", conductivity=0.3, value=-0.3), "side": "upper"}]
    )
    torch.testing.assert_close(
        loss(flux, lambda columns, time: {"T": columns[0]}), torch.tensor(0.0)
    )


def test_parameterized_flux() -> None:
    data = create_interval_data(
        [0, 1], [0, 1], 3, 2, parameters={"k": {"interval": [0.1, 0.5], "points": 3}}
    )
    dataset = BoundaryConditions(
        PointCloud(lambda: data),
        [{**condition("heat_flux", "x", conductivity="k", value_fn="flux"), "side": "upper"}],
        {"flux": lambda columns, time: -columns[1]},
    )
    torch.testing.assert_close(
        loss(dataset, lambda columns, time: {"T": columns[0]}), torch.tensor(0.0)
    )


def test_mixed_corner_policy() -> None:
    dataset = BoundaryConditions(mesh(), [condition("dirichlet", "x"), condition("neumann", "y")])
    assert len(dataset) == 16
    _, _, values = dataset[:]
    assert torch.count_nonzero(values["weight_0"]) == 12
    assert torch.count_nonzero(values["weight_1"]) == 4


def test_insufficient_face_grid() -> None:
    with pytest.raises(ValueError, match="no eligible points"):
        BoundaryConditions(mesh(count=2), [condition("neumann", "x")])


def test_shared_dirichlet_weights_use_maximum() -> None:
    dataset = BoundaryConditions(
        mesh(), [condition("dirichlet", "x", weight=2), condition("dirichlet", "y", weight=3)]
    )
    _, _, values = dataset[:]
    assert values["weight_0"].max() == 3
    assert len(dataset.constraints) == 1


@pytest.mark.parametrize("parameterized", [False, True])
def test_runner_explicit_conditions(tmp_path: Path, parameterized: bool) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    cfg = yaml.safe_load((input_dir / "config.yaml").read_text())
    cfg["domain"]["space"]["y"] = {"interval": [0, 1], "points": 3}
    if parameterized:
        cfg["domain"]["parameters"] = {"k": {"interval": [0.1, 0.5], "points": 3}}
    cfg["boundaries"] = [
        condition("dirichlet", "x"),
        condition("neumann", "y", value_function="insulated"),
    ]
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (input_dir / "functions.py").write_text("""import torch
from pinnstorch.api import Coordinates, Equation, Problem, TensorFields

def initial(c: Coordinates) -> TensorFields:
    return {"T": torch.sin(torch.pi * c.x)}

def heat(e: Equation) -> torch.Tensor:
    return e.dt("T") - e.parameters.get("k", 0.3) * e.laplacian("T")

def insulated(c: Coordinates) -> torch.Tensor:
    return torch.zeros_like(c.t)

problem = Problem(fields=("T",), equations={"heat": heat}, initial=initial, boundary_values={"insulated": insulated})
""")
    result = launch(input_dir, tmp_path / "output")
    assert result.returncode == 0, result.stdout + result.stderr


def test_constant_field_neumann() -> None:
    dataset = BoundaryConditions(mesh(), [condition("neumann", "x")])
    torch.testing.assert_close(
        loss(dataset, lambda columns, time: {"T": torch.ones_like(time)}), torch.tensor(0.0)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"weight": 0},
        {"weight": float("nan")},
        {"unexpected": 1},
        {"alpha": 2},
        {"value_fn": "absent"},
    ],
)
def test_invalid_boundary_options(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BoundaryConditions(mesh(), [condition("dirichlet", "x", **options)])
