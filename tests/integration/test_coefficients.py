"""Analytic interpolation, autograd, dependency and prescribed-input contracts."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from pinnstorch.api import Coordinates, Equation, Problem
from pinnstorch.coefficients import RegularGridCoefficient
from pinnstorch.data import BoundaryConditions
from pinnstorch.runner.coefficients import CoefficientBank
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import ProblemAdapter
from pinnstorch.runner.settings import TaskSettings
from tests.integration.test_explicit_boundary import condition, loss, mesh


def domain(dimension: int = 2) -> TaskSettings:
    return TaskSettings.read(
        {
            "version": 1,
            "domain": {
                "space": {
                    axis: {"interval": [0, 1], "points": 3} for axis in ("x", "y", "z")[:dimension]
                },
                "time": {"interval": [0, 1], "points": 2},
            },
        }
    )


def coordinates(dimension: int) -> Coordinates:
    spatial = {
        axis: torch.linspace(0.13 + i * 0.07, 0.83 - i * 0.05, 7).reshape(-1, 1).requires_grad_()
        for i, axis in enumerate(("x", "y", "z")[:dimension])
    }
    return Coordinates(spatial, torch.ones_like(spatial["x"]).requires_grad_())


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_multilinear_values_derivatives_and_backprop(dimension: int) -> None:
    axes = {
        axis: torch.linspace(0, 1, 3 + i) for i, axis in enumerate(("x", "y", "z")[:dimension])
    }
    grids = torch.meshgrid(*axes.values(), indexing="ij")
    values = torch.ones_like(grids[0])
    for i, grid in enumerate(grids):
        values = values * (1 + (i + 1) * grid)
    table = RegularGridCoefficient(axes, values)
    c = coordinates(dimension)
    actual = table.forward(c)
    expected = torch.ones_like(c.t)
    for i, point in enumerate(c.spatial.values()):
        expected = expected * (1 + (i + 1) * point)
    torch.testing.assert_close(actual, expected)
    equation = Equation({"coefficient": actual}, c)
    for axis, point in c.spatial.items():
        derivative = torch.autograd.grad(
            expected, point, torch.ones_like(expected), create_graph=True, retain_graph=True
        )[0]
        torch.testing.assert_close(equation.diff("coefficient", axis), derivative)
    scale = torch.tensor(2.0, requires_grad=True)
    field = scale * sum(point.square() for point in c.spatial.values())
    heat = Equation({"T": field}, c)
    flux = {axis: actual * derivative for axis, derivative in heat.gradient("T").items()}
    residual = heat.divergence(flux)
    residual.square().mean().backward()
    assert scale.grad is not None and torch.isfinite(scale.grad) and scale.grad > 0
    assert not list(table.parameters())


def test_formula_cache_dependencies_and_lifecycle() -> None:
    base = Mock(side_effect=lambda c: 1 + c.x)
    bank = CoefficientBank({"base": base, "conductivity": lambda c: 2 * c.coefficients["base"]})
    c = bank.context(coordinates(1))
    assert c.coefficients["base"] is c.coefficients["base"]
    torch.testing.assert_close(c.coefficients["conductivity"], 2 + 2 * c.x)
    assert base.call_count == 1
    another = bank.context(coordinates(1))
    another.coefficients["base"]
    assert base.call_count == 2
    # The cache must not form a context→view→context reference cycle holding graphs.
    assert not c.coefficients.coordinates.coefficients
    with pytest.raises(KeyError):
        c.coefficients["missing"]
    cyclic = CoefficientBank(
        {"a": lambda c: c.coefficients["b"], "b": lambda c: c.coefficients["a"]}
    ).context(coordinates(1))
    with pytest.raises(ValueError, match="Cyclic"):
        cyclic.coefficients["a"]


def test_conservative_variable_conductivity_residual() -> None:
    table = RegularGridCoefficient(
        {"x": torch.tensor([0.0, 0.3, 1.0])}, torch.tensor([1.0, 1.3, 2.0])
    )
    c = CoefficientBank({"conductivity": table}).context(coordinates(2))
    e = Equation({"T": c.x.square() + c.y.square()}, c)
    flux = {
        axis: c.coefficients["conductivity"] * derivative
        for axis, derivative in e.gradient("T").items()
    }
    torch.testing.assert_close(e.divergence(flux), 4 + 6 * c.x)


def test_grid_nodes_and_nonuniform_cells() -> None:
    table = RegularGridCoefficient(
        {"x": torch.tensor([0.0, 0.2, 1.0])}, torch.tensor([0.0, 2.0, 6.0])
    )
    x = torch.tensor([[0.0], [0.1], [0.2], [0.6], [1.0]], requires_grad=True)
    c = Coordinates({"x": x}, torch.zeros_like(x))
    torch.testing.assert_close(table.forward(c), torch.tensor([[0.0], [1.0], [2.0], [4.0], [6.0]]))
    torch.testing.assert_close(
        Equation({"v": table.forward(c)}, c).diff("v", "x"),
        torch.tensor([[10.0], [10.0], [5.0], [5.0], [5.0]]),
    )


@pytest.mark.parametrize(
    "axes,values",
    [
        ({}, torch.ones(2)),
        ({"t": torch.tensor([0.0, 1.0])}, torch.ones(2)),
        ({"x": torch.tensor([0.0, 0.0])}, torch.ones(2)),
        ({"x": torch.tensor([0.0, 1.0])}, torch.ones(3)),
        ({"x": torch.tensor([0.0, 1.0])}, torch.tensor([1.0, float("nan")])),
        ({"x": torch.tensor([1.0, 1.0 + 1e-10], dtype=torch.float64)}, torch.ones(2)),
    ],
)
def test_invalid_tables(axes: dict[str, torch.Tensor], values: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        RegularGridCoefficient(axes, values)


def test_npz_roundtrip_identity_and_coverage(tmp_path: Path) -> None:
    path = tmp_path / "field.npz"
    np.savez(path, x=[0.0, 1.0], values=[1.0, 2.0])
    first = RegularGridCoefficient.load(path)
    np.savez(path, values=[1.0, 2.0], x=[0.0, 1.0])
    assert RegularGridCoefficient.load(path).signature == first.signature
    np.savez(path, x=[0.0, 1.0], values=[1.0, 3.0])
    assert RegularGridCoefficient.load(path).signature != first.signature
    assert "values" in first.state_dict()
    CoefficientBank({"k": first}).validate_domain(domain(1).domain)
    with pytest.raises(ValueError, match="cover"):
        CoefficientBank(
            {"k": RegularGridCoefficient({"x": torch.tensor([0.1, 0.9])}, torch.ones(2))}
        ).validate_domain(domain(1).domain)
    with pytest.raises(ValueError, match="absent"):
        CoefficientBank(
            {"k": RegularGridCoefficient({"z": torch.tensor([0.0, 1.0])}, torch.ones(2))}
        ).validate_domain(domain(1).domain)
    np.savez(path, x=[0.0, 1.0], values=[1.0, 2.0], unwanted=[3])
    with pytest.raises(ValueError, match="only"):
        RegularGridCoefficient.load(path)


def test_named_field_heat_flux_and_weight_gradients() -> None:
    def k(columns: list[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        return 1 + columns[0]

    dataset = BoundaryConditions(
        mesh(),
        [{**condition("heat_flux", "x", conductivity="k", value=-2.0), "side": "upper"}],
        coefficient_values={"k": k},
    )
    scale = torch.tensor(2.0, requires_grad=True)
    result = loss(dataset, lambda columns, time: {"T": scale * columns[0]})
    torch.testing.assert_close(result, torch.tensor(4.0))
    result.backward()
    torch.testing.assert_close(scale.grad, torch.tensor(8.0))
    with pytest.raises(ValueError, match="positive"):
        BoundaryConditions(
            mesh(),
            [condition("heat_flux", "x", conductivity="k")],
            coefficient_values={"k": lambda columns, time: -torch.ones_like(time)},
        )


def test_unknown_conductivity_and_name_conflicts(tmp_path: Path) -> None:
    settings = domain(1)
    problem = Problem(
        fields=("T",),
        initial=lambda c: {"T": torch.ones_like(c.t)},
        equations={"heat": lambda e: e.field("T")},
    )
    source = {
        "version": 1,
        "domain": {
            "space": {"x": {"interval": [0, 1], "points": 3}},
            "time": {"interval": [0, 1], "points": 2},
        },
        "boundaries": [condition("heat_flux", "x", conductivity="missing")],
    }
    with pytest.raises(ValueError, match="Unknown conductivity"):
        SolverFactory(TaskSettings.read(source), problem, tmp_path)
    with pytest.raises(ValueError, match="Coefficient names"):
        ProblemAdapter(problem, settings.domain, CoefficientBank({"x": lambda c: 1.0}))


@pytest.mark.parametrize("value", [torch.ones(3), True, {"bad": 1}])
def test_invalid_formula_result(value: object) -> None:
    bank = CoefficientBank({"k": lambda c: value})
    with pytest.raises(ValueError, match="scalar or Tensor"):
        bank.context(coordinates(1)).coefficients["k"]


def test_scalar_parameter_can_scale_a_tabulated_field() -> None:
    table = RegularGridCoefficient({"x": torch.tensor([0.0, 1.0])}, torch.tensor([1.0, 2.0]))
    base = coordinates(1)
    amplitude = torch.full_like(base.t, 0.3).requires_grad_()
    c = CoefficientBank(
        {
            "base": table,
            "conductivity": lambda c: c.parameters["amplitude"] * c.coefficients["base"],
        }
    ).context(Coordinates(base.spatial, base.t, {"amplitude": amplitude}))
    k = c.coefficients["conductivity"]
    e = Equation({"k": k}, c)
    torch.testing.assert_close(e.diff("k", "x"), amplitude)
    torch.testing.assert_close(e.diff("k", "amplitude"), 1 + c.x)


def test_table_buffers_follow_module_dtype() -> None:
    bank = CoefficientBank(
        {
            "values": RegularGridCoefficient(
                {"x": torch.tensor([0.0, 1.0])}, torch.tensor([1.0, 2.0])
            )
        }
    )
    bank.double()
    assert bank.table("values").values.dtype == torch.float64
    x = torch.tensor([[0.25]], dtype=torch.float64, requires_grad=True)
    value = bank.context(Coordinates({"x": x}, torch.zeros_like(x))).coefficients["values"]
    torch.testing.assert_close(value, torch.tensor([[1.25]], dtype=torch.float64))
    assert value.dtype == torch.float64


@pytest.mark.parametrize(
    "specification",
    [{"file": "/tmp/field.npz"}, {"file": "field.mat"}, {"file": "field.npz", "other": 1}],
)
def test_invalid_coefficient_config(specification: dict[str, object]) -> None:
    source = {
        "version": 1,
        "domain": {
            "space": {"x": {"interval": [0, 1], "points": 3}},
            "time": {"interval": [0, 1], "points": 2},
        },
        "coefficients": {"k": specification},
    }
    with pytest.raises(ValueError):
        TaskSettings.read(source)


def test_table_file_paths_duplicates_and_settings_roundtrip(tmp_path: Path) -> None:
    from pinnstorch.runner.settings import TableCoefficient

    with pytest.raises(ValueError, match="inside input-dir"):
        CoefficientBank.build(
            {}, {"k": TableCoefficient("../field.npz")}, tmp_path, domain(1).domain
        )
    with pytest.raises(ValueError, match="both YAML"):
        CoefficientBank.build(
            {"k": lambda c: 1.0}, {"k": TableCoefficient("field.npz")}, tmp_path, domain(1).domain
        )
    source = {
        "version": 1,
        "domain": {
            "space": {"x": {"interval": [0, 1], "points": 3}},
            "time": {"interval": [0, 1], "points": 2},
        },
        "coefficients": {"k": {"file": "field.npz"}},
    }
    settings = TaskSettings.read(source)
    settings.save(tmp_path / "config.yaml")
    assert TaskSettings.load(tmp_path / "config.yaml") == settings


def test_initial_and_boundary_callbacks_receive_coefficients() -> None:
    from pinnstorch.api import TensorFields
    from pinnstorch.data import InitialCondition, PointCloud

    def initial(c: Coordinates) -> TensorFields:
        return {"T": c.coefficients["k"]}

    problem = Problem(
        fields=("T",),
        initial=initial,
        equations={"heat": lambda e: e.field("T")},
        coefficients={"k": lambda c: 1 + c.x},
        boundary_values={"wall": lambda c: c.coefficients["k"]},
    )
    settings = domain(1)
    adapter = ProblemAdapter(problem, settings.domain)
    grid = PointCloud(settings.domain.grid)
    initial_dataset = InitialCondition(grid, solution=["T"], initial_fun=adapter.initial)
    columns, time, values = initial_dataset[:]
    torch.testing.assert_close(values["T"], 1 + columns[0])
    torch.testing.assert_close(adapter.boundary_value("wall", columns, time), values["T"])
