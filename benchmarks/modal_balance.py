"""Project physical heat residuals onto an eigenmode and account for boundary defects."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from pinnstorch.api import Coordinates, Equation

from .box_heat_balance import Array, BoxField, SpatialQuadrature, _integrate, _sample


@dataclass(frozen=True)
class HeatEigenmode:
    frequencies: tuple[float, ...]
    shifts: tuple[float, ...]
    conductivity: float = 0.1

    def __post_init__(self) -> None:
        if not 1 <= len(self.frequencies) <= 3 or len(self.shifts) != len(self.frequencies):
            raise ValueError("Eigenmode requires one to three matching frequencies/shifts")
        if not all(np.isfinite((*self.frequencies, *self.shifts, self.conductivity))):
            raise ValueError("Eigenmode coefficients must be finite")
        if self.conductivity <= 0:
            raise ValueError("Conductivity must be positive")

    @property
    def decay_rate(self) -> float:
        return self.conductivity * sum(frequency**2 for frequency in self.frequencies)

    def spatial(self, columns: list[torch.Tensor]) -> torch.Tensor:
        value = torch.ones_like(columns[0])
        for x, frequency, shift in zip(columns, self.frequencies, self.shifts):
            value = value * x.sub(shift).mul(frequency).cos()
        return value

    def normal_derivative(self, columns: list[torch.Tensor], axis: int, side: int) -> torch.Tensor:
        value = torch.ones_like(columns[0])
        for index, (x, frequency, shift) in enumerate(zip(columns, self.frequencies, self.shifts)):
            phase = x.sub(shift).mul(frequency)
            value = value * (-frequency * phase.sin() if index == axis else phase.cos())
        return cast(torch.Tensor, value * (-1 if side == 0 else 1))


@dataclass(frozen=True)
class AffineHeatSolution:
    mode: HeatEigenmode
    background_rate: float

    def source(self, spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(clock)
        for column in spatial:
            result = result + column.mul(self.background_rate)
        return result

    def reference(self, spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        background = torch.ones_like(clock) + clock * self.source(spatial, clock)
        return cast(
            torch.Tensor,
            background + self.mode.spatial(spatial) * clock.mul(-self.mode.decay_rate).exp(),
        )


@dataclass(frozen=True)
class ModalBalance:
    rows: list[dict[str, float]]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ModalGrid:
    axes: tuple[SpatialQuadrature, ...]
    instants: torch.Tensor
    bounds: tuple[tuple[float, float], ...]

    def columns(self, face: tuple[int, int] | None = None) -> list[torch.Tensor]:
        nodes = [axis.nodes for axis in self.axes]
        if face is not None:
            axis, side = face
            nodes[axis] = torch.tensor([self.bounds[axis][side]], dtype=torch.float32)
        mesh = torch.meshgrid(*nodes, self.instants, indexing="ij")
        return [column.reshape(-1, 1) for column in mesh]

    def integrate(self, flat: Array, face_axis: int | None = None) -> Array:
        rules = tuple(rule for i, rule in enumerate(self.axes) if i != face_axis)
        shape = tuple(len(rule.nodes) for rule in rules) + (len(self.instants),)
        return _integrate(flat.reshape(shape), rules)


@dataclass(frozen=True)
class ModalVolume:
    norm: Array
    solution: Array
    rate: Array
    source: Array
    residual: Array
    error: Array | None
    error_squared_integral: Array | None
    initial: float | None


@dataclass(frozen=True)
class ModalBoundary:
    prescribed: Array
    defect: Array


def _volume_jets(
    field: BoxField, columns: list[torch.Tensor], device: str, batch_size: int
) -> tuple[Array, Array, Array]:
    values, rates, laplacians = [], [], []
    for start in range(0, len(columns[0]), batch_size):
        spatial = [
            x[start : start + batch_size].to(device).detach().requires_grad_()
            for x in columns[:-1]
        ]
        time = columns[-1][start : start + batch_size].to(device).detach().requires_grad_()
        with torch.set_grad_enabled(True):
            physical = field(spatial, time)
            equation = Equation({"u": physical}, Coordinates(dict(zip("xyz", spatial)), time))
            rate, laplacian = equation.dt("u"), equation.laplacian("u")
        values.append(physical.detach().cpu().numpy().astype(np.float64))
        rates.append(rate.detach().cpu().numpy().astype(np.float64))
        laplacians.append(laplacian.detach().cpu().numpy().astype(np.float64))
    return (
        np.concatenate(values).reshape(-1),
        np.concatenate(rates).reshape(-1),
        np.concatenate(laplacians).reshape(-1),
    )


def _volume(
    field: BoxField,
    reference: BoxField | None,
    source: BoxField,
    initial: BoxField | None,
    mode: HeatEigenmode,
    grid: ModalGrid,
    device: str,
    batch_size: int,
) -> ModalVolume:
    columns = grid.columns()
    phi = mode.spatial(columns[:-1]).detach().numpy().astype(np.float64).reshape(-1)
    norm = grid.integrate(phi**2)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError("Eigenmode must have nonzero spatial norm")
    value, rate, laplacian = _volume_jets(field, columns, device, batch_size)
    forcing = source(columns[:-1], columns[-1]).detach().numpy().astype(np.float64).reshape(-1)
    exact = (
        reference(columns[:-1], columns[-1]).detach().cpu().numpy().astype(np.float64).reshape(-1)
        if reference is not None
        else None
    )
    initial_coefficient = None
    if initial is not None:
        spatial = [column[:: len(grid.instants)] for column in columns[:-1]]
        clock = torch.full_like(spatial[0], float(grid.instants[0]))
        values = initial(spatial, clock).detach().numpy().astype(np.float64).reshape(-1)
        initial_coefficient = float(
            _integrate(
                (values * phi[:: len(grid.instants)]).reshape(
                    tuple(len(axis.nodes) for axis in grid.axes)
                ),
                grid.axes,
            )
            / norm[0]
        )
    return ModalVolume(
        norm,
        grid.integrate(value * phi) / norm,
        grid.integrate(rate * phi) / norm,
        grid.integrate(forcing * phi) / norm,
        grid.integrate((rate - mode.conductivity * laplacian - forcing) * phi) / norm,
        grid.integrate((value - exact) * phi) / norm if exact is not None else None,
        grid.integrate((value - exact) ** 2) if exact is not None else None,
        initial_coefficient,
    )


def _boundary(
    field: BoxField,
    targets: tuple[tuple[BoxField, BoxField], ...],
    operators: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    mode: HeatEigenmode,
    grid: ModalGrid,
    norm: Array,
    device: str,
    batch_size: int,
) -> ModalBoundary:
    supplied = np.zeros(len(grid.instants), dtype=np.float64)
    boundary = np.zeros_like(supplied)
    for axis, axis_name in enumerate("xyz"[: len(grid.axes)]):
        for side in (0, 1):
            columns = grid.columns((axis, side))
            alpha, beta = operators[axis][side]
            phi = mode.spatial(columns[:-1])
            normal = mode.normal_derivative(columns[:-1], axis, side)
            eigen_defect = cast(torch.Tensor, alpha * phi + beta * normal)
            if float(eigen_defect.abs().max()) > 2e-5:
                raise ValueError(
                    "Chosen eigenmode does not satisfy the homogeneous boundary operator"
                )
            wall, derivative = _sample(field, columns, axis_name, device, batch_size)
            prescribed = (
                targets[axis][side](columns[:-1], columns[-1])
                .detach()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            defect = alpha * wall + beta * (-1 if side == 0 else 1) * derivative - prescribed
            weight = phi.detach().numpy().astype(np.float64).reshape(-1) / beta
            supplied += mode.conductivity * grid.integrate(prescribed * weight, axis) / norm
            boundary += mode.conductivity * grid.integrate(defect * weight, axis) / norm
    return ModalBoundary(supplied, boundary)


def _summarize(
    volume: ModalVolume, boundary: ModalBoundary, mode: HeatEigenmode, grid: ModalGrid
) -> ModalBalance:
    clock = grid.instants.numpy().astype(np.float64)
    decay = np.exp(-mode.decay_rate * (clock - clock[0]))
    drive = volume.residual + boundary.defect
    identity_defect = (
        volume.rate
        + mode.decay_rate * volume.solution
        - volume.source
        - boundary.prescribed
        - drive
    )
    rows = [
        {
            "time": float(t),
            "solution_coefficient": float(volume.solution[i]),
            "projected_residual": float(volume.residual[i]),
            "boundary_forcing": float(boundary.defect[i]),
            "ode_identity_defect": float(identity_defect[i]),
        }
        for i, t in enumerate(clock)
    ]
    metrics = {
        "ode_identity_rmse": float(np.sqrt(np.mean(identity_defect**2))),
        "max_boundary_forcing": float(np.max(np.abs(boundary.defect))),
        "max_projected_residual": float(np.max(np.abs(volume.residual))),
    }
    if volume.error is not None:
        relative = volume.error / decay
        reconstructed = relative[0] + cumulative_trapezoid(drive / decay, clock, initial=0)
        for i, row in enumerate(rows):
            row.update(
                amplitude_error=float(volume.error[i]),
                relative_amplitude_error=float(relative[i]),
                reconstructed_relative_amplitude_error=float(reconstructed[i]),
            )
        metrics.update(
            final_signed_relative_amplitude_error=float(relative[-1]),
            final_reconstructed_relative_amplitude_error=float(reconstructed[-1]),
            reconstruction_max_error=float(np.max(np.abs(relative - reconstructed))),
        )
        if volume.error_squared_integral is not None:
            total = np.sqrt(volume.error_squared_integral / volume.norm) / decay
            measure = float(np.prod([upper - lower for lower, upper in grid.bounds]))
            orthogonal = np.sqrt(
                np.maximum(volume.error_squared_integral - volume.error**2 * volume.norm, 0)
                / measure
            )
            for i, row in enumerate(rows):
                row.update(
                    mode_relative_l2_gauss=float(total[i]),
                    orthogonal_rmse_gauss=float(orthogonal[i]),
                )
            metrics.update(
                final_mode_relative_l2_gauss=float(total[-1]),
                final_orthogonal_rmse_gauss=float(orthogonal[-1]),
            )
    if volume.initial is not None:
        expected = decay * (
            volume.initial
            + cumulative_trapezoid((volume.source + boundary.prescribed) / decay, clock, initial=0)
        )
        defect = volume.solution - expected
        for i, row in enumerate(rows):
            row.update(
                coefficient_from_problem_data=float(expected[i]),
                coefficient_defect_from_problem_data=float(defect[i]),
            )
        metrics.update(
            teacher_free_final_coefficient_defect=float(defect[-1]),
            teacher_free_max_coefficient_defect=float(np.max(np.abs(defect))),
        )
    return ModalBalance(rows, metrics)


def modal_heat_balance(
    field: BoxField,
    reference: BoxField | None,
    source: BoxField,
    targets: tuple[tuple[BoxField, BoxField], ...],
    operators: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    mode: HeatEigenmode,
    bounds: tuple[tuple[float, float], ...],
    time_interval: tuple[float, float],
    device: str,
    space_points: int = 13,
    time_points: int = 81,
    batch_size: int = 512,
    initial: BoxField | None = None,
) -> ModalBalance:
    dimensions = len(bounds)
    if (
        dimensions != len(mode.frequencies)
        or len(targets) != dimensions
        or len(operators) != dimensions
    ):
        raise ValueError("Modal balance requires matching dimensions, operators and data")
    if reference is None and initial is None:
        raise ValueError("Supply initial data for reference-free modal diagnostics")
    if (
        time_points < 2
        or batch_size < 1
        or not np.isfinite(time_interval).all()
        or time_interval[1] <= time_interval[0]
    ):
        raise ValueError(
            "Increasing time interval, at least two times and positive batch required"
        )
    if any(
        not np.isfinite((alpha, beta)).all() or beta == 0
        for pair in operators
        for alpha, beta in pair
    ):
        raise ValueError(
            "Modal boundary correction requires finite Robin operators with nonzero beta"
        )
    grid = ModalGrid(
        tuple(
            SpatialQuadrature.build(interval, space_points, "gauss_legendre")
            for interval in bounds
        ),
        torch.linspace(*time_interval, time_points),
        bounds,
    )
    volume = _volume(field, reference, source, initial, mode, grid, device, batch_size)
    boundary = _boundary(field, targets, operators, mode, grid, volume.norm, device, batch_size)
    return _summarize(volume, boundary, mode, grid)
