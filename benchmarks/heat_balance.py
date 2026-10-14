"""Independent one-dimensional heat content and normal-derivative diagnostics."""

from collections.abc import Callable

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from pinnstorch.api import Coordinates, Equation

PhysicalField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def heat_balance(
    field: PhysicalField,
    bounds: tuple[float, float],
    time_interval: tuple[float, float],
    device: str,
    alpha: float,
    beta: float,
    target: PhysicalField,
    source: PhysicalField,
    conductivity: float = 0.1,
    space_points: int = 257,
    time_points: int = 31,
) -> dict[str, float]:
    space = torch.linspace(*bounds, space_points, dtype=torch.float32, device=device)
    clock = torch.linspace(*time_interval, time_points, dtype=torch.float32, device=device)
    xx, tt = torch.meshgrid(space, clock, indexing="ij")
    x = xx.reshape(-1, 1).detach().requires_grad_()
    t = tt.reshape(-1, 1).detach().requires_grad_()
    with torch.set_grad_enabled(True):
        physical = field(x, t)
        equation = Equation({"u": physical}, Coordinates({"x": x}, t))
        u_t = equation.dt("u")
        u_x = equation.diff("u", "x")
        rhs = source(x, t)
        targets = target(x, t)
    values, time_derivative, space_derivative, forcing, prescribed = [
        value.detach().cpu().numpy().astype(np.float64).reshape(space_points, time_points)
        for value in (physical, u_t, u_x, rhs, targets)
    ]
    positions = space.cpu().numpy().astype(np.float64)
    instants = clock.cpu().numpy().astype(np.float64)
    normal = np.stack((-space_derivative[0], space_derivative[-1]))
    wall_values = np.stack((values[0], values[-1]))
    wall_targets = np.stack((prescribed[0], prescribed[-1]))
    wall_error = alpha * wall_values + beta * normal - wall_targets
    content = np.trapz(values, positions, axis=0)
    content_rate = np.trapz(time_derivative, positions, axis=0)
    supply = conductivity * normal.sum(axis=0) + np.trapz(forcing, positions, axis=0)
    balance = content_rate - supply
    accumulated = content - content[0] - cumulative_trapezoid(supply, instants, initial=0)
    return {
        "boundary_operator_rmse": float(np.sqrt(np.mean(wall_error**2))),
        "boundary_operator_max_error": float(np.max(np.abs(wall_error))),
        "heat_balance_rate_rmse": float(np.sqrt(np.mean(balance**2))),
        "heat_balance_rate_max_error": float(np.max(np.abs(balance))),
        "heat_balance_accumulated_max_error": float(np.max(np.abs(accumulated))),
        "initial_heat_content": float(content[0]),
        "final_heat_content": float(content[-1]),
    }
