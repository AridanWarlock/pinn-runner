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
class ModalBalance:
    rows: list[dict[str, float]]
    metrics: dict[str, float]


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


def modal_heat_balance(
    field: BoxField,
    reference: BoxField,
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
) -> ModalBalance:
    dimensions = len(bounds)
    if (
        dimensions != len(mode.frequencies)
        or len(targets) != dimensions
        or len(operators) != dimensions
    ):
        raise ValueError("Modal balance requires matching dimensions, operators and data")
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
    axes = tuple(
        SpatialQuadrature.build(interval, space_points, "gauss_legendre") for interval in bounds
    )
    instants = torch.linspace(*time_interval, time_points)
    shape = (space_points,) * dimensions + (time_points,)
    mesh = torch.meshgrid(*(axis.nodes for axis in axes), instants, indexing="ij")
    columns = [column.reshape(-1, 1) for column in mesh]
    phi = mode.spatial(columns[:-1]).detach().numpy().astype(np.float64).reshape(shape)
    norm = _integrate(phi**2, axes)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError("Eigenmode must have nonzero spatial norm")
    value, rate, laplacian = _volume_jets(field, columns, device, batch_size)
    exact, exact_rate = _sample(reference, columns, "t", "cpu", batch_size)
    forcing = source(columns[:-1], columns[-1]).detach().numpy().astype(np.float64).reshape(shape)
    error = value.reshape(shape) - exact.reshape(shape)
    error_rate = rate.reshape(shape) - exact_rate.reshape(shape)
    amplitude = _integrate(error * phi, axes) / norm
    amplitude_rate = _integrate(error_rate * phi, axes) / norm
    residual = rate.reshape(shape) - mode.conductivity * laplacian.reshape(shape) - forcing
    projected_residual = _integrate(residual * phi, axes) / norm
    boundary = np.zeros(time_points, dtype=np.float64)
    for axis, axis_name in enumerate("xyz"[:dimensions]):
        for side in (0, 1):
            nodes = [rule.nodes for rule in axes]
            nodes[axis] = torch.tensor([bounds[axis][side]], dtype=torch.float32)
            face_mesh = torch.meshgrid(*nodes, instants, indexing="ij")
            face_columns = [column.reshape(-1, 1) for column in face_mesh]
            alpha, beta = operators[axis][side]
            eigenvalue = mode.spatial(face_columns[:-1])
            eigennormal = mode.normal_derivative(face_columns[:-1], axis, side)
            eigen_defect = cast(torch.Tensor, alpha * eigenvalue + beta * eigennormal)
            if float(eigen_defect.abs().max()) > 2e-5:
                raise ValueError(
                    "Chosen eigenmode does not satisfy the homogeneous boundary operator"
                )
            wall, derivative = _sample(field, face_columns, axis_name, device, batch_size)
            prescribed = (
                targets[axis][side](face_columns[:-1], face_columns[-1])
                .detach()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            defect = alpha * wall + beta * (-1 if side == 0 else 1) * derivative - prescribed
            face_shape = (space_points,) * (dimensions - 1) + (time_points,)
            weighted = defect * eigenvalue.detach().numpy().astype(np.float64).reshape(-1) / beta
            boundary += (
                mode.conductivity
                * _integrate(
                    weighted.reshape(face_shape),
                    tuple(rule for i, rule in enumerate(axes) if i != axis),
                )
                / norm
            )
    clock = instants.numpy().astype(np.float64)
    decay = np.exp(-mode.decay_rate * (clock - time_interval[0]))
    relative_amplitude = amplitude / decay
    drive = projected_residual + boundary
    reconstructed = relative_amplitude[0] + cumulative_trapezoid(drive / decay, clock, initial=0)
    identity_defect = amplitude_rate + mode.decay_rate * amplitude - drive
    return ModalBalance(
        rows=[
            {
                "time": float(t),
                "amplitude_error": float(amplitude[i]),
                "relative_amplitude_error": float(relative_amplitude[i]),
                "projected_residual": float(projected_residual[i]),
                "boundary_forcing": float(boundary[i]),
                "ode_identity_defect": float(identity_defect[i]),
                "reconstructed_relative_amplitude_error": float(reconstructed[i]),
            }
            for i, t in enumerate(clock)
        ],
        metrics={
            "final_signed_relative_amplitude_error": float(relative_amplitude[-1]),
            "final_reconstructed_relative_amplitude_error": float(reconstructed[-1]),
            "reconstruction_max_error": float(np.max(np.abs(relative_amplitude - reconstructed))),
            "ode_identity_rmse": float(np.sqrt(np.mean(identity_defect**2))),
            "max_boundary_forcing": float(np.max(np.abs(boundary))),
            "max_projected_residual": float(np.max(np.abs(projected_residual))),
        },
    )
