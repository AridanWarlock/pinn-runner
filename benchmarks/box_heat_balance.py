"""Independent volume/face quadrature for Cartesian heat and boundary operators."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from pinnstorch.api import Coordinates, Equation

BoxField = Callable[[list[torch.Tensor], torch.Tensor], torch.Tensor]
Array = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialQuadrature:
    nodes: torch.Tensor
    weights: Array

    @classmethod
    def build(cls, bounds: tuple[float, float], points: int, kind: str) -> "SpatialQuadrature":
        lower, upper = bounds
        if points < 2:
            raise ValueError("Quadrature requires at least two points")
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError("Quadrature requires a finite positive interval")
        if kind == "gauss_legendre":
            legendre_rule = cast(
                Callable[[int], tuple[Array, Array]], np.polynomial.legendre.leggauss
            )
            nodes, weights = legendre_rule(points)
            return cls(
                torch.tensor(lower + (nodes + 1) * (upper - lower) / 2, dtype=torch.float32),
                weights * (upper - lower) / 2,
            )
        if kind != "trapezoid":
            raise ValueError("Unknown spatial quadrature")
        nodes = torch.linspace(lower, upper, points)
        positions = nodes.numpy().astype(np.float64)
        weights = np.empty(points, dtype=np.float64)
        weights[0], weights[-1] = (
            (positions[1] - positions[0]) / 2,
            (positions[-1] - positions[-2]) / 2,
        )
        weights[1:-1] = (positions[2:] - positions[:-2]) / 2
        return cls(nodes, weights)


def _integrate(values: Array, axes: tuple[SpatialQuadrature, ...]) -> Array:
    for axis in axes:
        values = cast(Array, np.tensordot(axis.weights, values, axes=(0, 0)))
    return values


def _sample(
    field: BoxField,
    columns: list[torch.Tensor],
    axis: str,
    device: str,
    batch_size: int,
) -> tuple[Array, Array]:
    values, derivatives = [], []
    for start in range(0, len(columns[0]), batch_size):
        spatial = [
            x[start : start + batch_size].to(device).detach().requires_grad_()
            for x in columns[:-1]
        ]
        time = columns[-1][start : start + batch_size].to(device).detach().requires_grad_()
        with torch.set_grad_enabled(True):
            physical = field(spatial, time)
            equation = Equation({"u": physical}, Coordinates(dict(zip("xyz", spatial)), time))
            derivative = equation.diff("u", axis)
        values.append(physical.detach().cpu().numpy().astype(np.float64))
        derivatives.append(derivative.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(values).reshape(-1), np.concatenate(derivatives).reshape(-1)


def box_heat_balance(
    field: BoxField,
    targets: tuple[tuple[BoxField, BoxField], ...],
    source: BoxField,
    bounds: tuple[tuple[float, float], ...],
    time_interval: tuple[float, float],
    operators: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    device: str,
    conductivity: float = 0.1,
    space_points: int = 17,
    time_points: int = 21,
    batch_size: int = 256,
    quadrature: str = "trapezoid",
) -> dict[str, float]:
    if not 1 <= len(bounds) <= 3 or len(targets) != len(bounds) or len(operators) != len(bounds):
        raise ValueError(
            "Balance diagnostic requires one to three axes with matching operator data"
        )
    if space_points < 2 or time_points < 2 or batch_size < 1:
        raise ValueError("Quadrature counts must exceed one and batch size must be positive")
    axes = tuple(
        SpatialQuadrature.build(interval, space_points, quadrature) for interval in bounds
    )
    instants = torch.linspace(*time_interval, time_points)
    dimensions = len(bounds)
    mesh = torch.meshgrid(*(axis.nodes for axis in axes), instants, indexing="ij")
    columns = [x.reshape(-1, 1) for x in mesh]
    values, rates = _sample(field, columns, "t", device, batch_size)
    shape = (space_points,) * dimensions + (time_points,)
    clock = instants.numpy().astype(np.float64)
    # Prescribed source/data are evaluated independently on CPU, not teacher values.
    forcing = source(columns[:-1], columns[-1]).detach().numpy().astype(np.float64).reshape(shape)
    content = _integrate(values.reshape(shape), axes)
    rate = _integrate(rates.reshape(shape), axes)
    supply = _integrate(forcing, axes)
    wall_errors = []
    for axis, axis_name in enumerate("xyz"[:dimensions]):
        for side in (0, 1):
            face_nodes = [rule.nodes for rule in axes]
            face_nodes[axis] = torch.tensor([bounds[axis][side]], dtype=torch.float32)
            face_mesh = torch.meshgrid(*face_nodes, instants, indexing="ij")
            face_columns = [x.reshape(-1, 1) for x in face_mesh]
            face_values, derivative = _sample(field, face_columns, axis_name, device, batch_size)
            normal = derivative * (-1 if side == 0 else 1)
            alpha, beta = operators[axis][side]
            prescribed = (
                targets[axis][side](face_columns[:-1], face_columns[-1])
                .detach()
                .numpy()
                .reshape(-1)
            )
            wall_errors.append(alpha * face_values + beta * normal - prescribed)
            face_shape = (space_points,) * (dimensions - 1) + (time_points,)
            supply = supply + conductivity * _integrate(
                normal.reshape(face_shape),
                tuple(rule for index, rule in enumerate(axes) if index != axis),
            )
    wall_error = np.concatenate(wall_errors)
    balance = rate - supply
    accumulated = content - content[0] - cumulative_trapezoid(supply, clock, initial=0)
    return {
        "boundary_operator_rmse": float(np.sqrt(np.mean(wall_error**2))),
        "boundary_operator_max_error": float(np.max(np.abs(wall_error))),
        "heat_balance_rate_rmse": float(np.sqrt(np.mean(balance**2))),
        "heat_balance_rate_max_error": float(np.max(np.abs(balance))),
        "heat_balance_accumulated_max_error": float(np.max(np.abs(accumulated))),
        "initial_heat_content": float(content[0]),
        "final_heat_content": float(content[-1]),
    }
