"""Face conditions with explicit normals and ownership of intersections."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from pinnstorch.contracts import FloatArray, LossAccumulator, SamplerBatch, TensorOutputs
from pinnstorch.data.mesh.mesh import MeshBase

from .sampler_base import SamplerBase

BoundaryValue = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Constraint:
    field: str
    kind: str
    axis: int
    alpha: float
    beta: float
    conductivity: float | str


class BoundaryPlan:
    """Prepare one vectorized boundary dataset outside the training loop."""

    def __init__(
        self,
        mesh: MeshBase,
        boundary_values: Mapping[str, BoundaryValue] | None,
        coefficient_values: Mapping[str, BoundaryValue] | None = None,
    ) -> None:
        self.mesh = mesh
        self.boundary_values = boundary_values or {}
        self.coefficient_values = coefficient_values or {}
        self.spatial, self.time, _ = mesh.flatten_mesh(None)
        self.columns = [
            torch.as_tensor(self.spatial[:, i : i + 1], dtype=torch.float32)
            for i in range(self.spatial.shape[1])
        ]
        self.time_tensor = torch.as_tensor(self.time, dtype=torch.float32)
        self.parameter_indices = {
            name: mesh.spatial_dim + i for i, name in enumerate(mesh.parameter_names)
        }
        on_face = (self.spatial[:, : mesh.spatial_dim] == mesh.lb[: mesh.spatial_dim]) | (
            self.spatial[:, : mesh.spatial_dim] == mesh.ub[: mesh.spatial_dim]
        )
        self.face_count = on_face.sum(axis=1)
        self.targets: list[FloatArray] = []
        self.weights: list[FloatArray] = []
        self.constraints: list[Constraint] = []
        self.dirichlet_groups: dict[str, int] = {}
        self.conductivities: list[FloatArray | None] = []

    def add(self, specification: Mapping[str, object]) -> None:
        constraint, side, weight = self._read_constraint(specification)
        values = self._values(specification)
        mask = self._face_mask(constraint, side)
        before = len(self.constraints)
        self._merge(constraint, values, mask, weight)
        if len(self.constraints) != before:
            self.conductivities.append(self._conductivity_values(constraint))

    def _read_constraint(
        self, specification: Mapping[str, object]
    ) -> tuple[Constraint, str, float]:
        mesh = self.mesh
        unknown = set(specification) - {
            "kind",
            "field",
            "axis",
            "side",
            "weight",
            "value",
            "value_fn",
            "alpha",
            "beta",
            "conductivity",
        }
        if unknown:
            raise ValueError(f"Unknown boundary options: {sorted(unknown)}")
        kind = str(specification.get("kind", ""))
        field = str(specification.get("field", ""))
        axis = {"x": 0, "y": 1, "z": 2}.get(str(specification.get("axis")))
        side = specification.get("side")
        if kind not in ("dirichlet", "neumann", "robin", "heat_flux") or not field:
            raise ValueError(
                "Boundary kind must be dirichlet/neumann/robin/heat_flux and field is required"
            )
        if axis is None or axis >= mesh.spatial_dim or side not in ("lower", "upper", "both"):
            raise ValueError("Boundary axis/side must identify an existing physical face")
        if kind != "robin" and ({"alpha", "beta"} & set(specification)):
            raise ValueError("alpha/beta are only valid for Robin conditions")
        if kind != "heat_flux" and "conductivity" in specification:
            raise ValueError("conductivity is only valid for heat_flux conditions")
        weight = self._number(specification, "weight", 1.0)
        alpha = self._number(specification, "alpha", 1.0)
        beta = self._number(specification, "beta", 1.0)
        conductivity = self._read_conductivity(specification)
        if not np.isfinite([weight, alpha, beta]).all() or weight <= 0:
            raise ValueError("Boundary weight must be positive; weight/alpha/beta must be finite")
        if kind == "robin" and alpha == 0 and beta == 0:
            raise ValueError("Robin alpha and beta cannot both be zero")
        return Constraint(field, kind, axis, alpha, beta, conductivity), str(side), weight

    def _read_conductivity(self, specification: Mapping[str, object]) -> float | str:
        mesh = self.mesh
        conductivity = specification.get("conductivity", 1.0)
        if isinstance(conductivity, str):
            if (
                conductivity not in self.parameter_indices
                and conductivity not in self.coefficient_values
            ):
                raise ValueError(f"Unknown conductivity parameter/coefficient '{conductivity}'")
            if (
                conductivity in self.parameter_indices
                and mesh.lb[self.parameter_indices[conductivity]] <= 0
            ):
                raise ValueError("Conductivity parameter range must be positive")
        else:
            conductivity = self._number(specification, "conductivity", 1.0)
            if not np.isfinite(conductivity) or conductivity <= 0:
                raise ValueError("Conductivity must be finite and positive")
        return conductivity

    def _conductivity_values(self, constraint: Constraint) -> FloatArray | None:
        name = constraint.conductivity
        if (
            constraint.kind != "heat_flux"
            or not isinstance(name, str)
            or name not in self.coefficient_values
        ):
            return None
        tensor = self.coefficient_values[name](self.columns, self.time_tensor).detach()
        if (
            tensor.shape != self.time_tensor.shape
            or not torch.isfinite(tensor).all()
            or not torch.all(tensor.gt(0))
        ):
            raise ValueError(
                f"Heat flux coefficient '{name}' must be finite, positive and shaped (N, 1)"
            )
        return cast(FloatArray, tensor.cpu().numpy().reshape(-1))

    @staticmethod
    def _number(specification: Mapping[str, object], name: str, default: float) -> float:
        value = specification.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"Boundary {name} must be a finite number")
        return float(value)

    def _values(self, specification: Mapping[str, object]) -> FloatArray:
        boundary_values, columns, time_tensor, spatial = (
            self.boundary_values,
            self.columns,
            self.time_tensor,
            self.spatial,
        )
        if "value_fn" in specification and "value" in specification:
            raise ValueError("Specify either value or value_fn")
        if "value_fn" in specification:
            name = str(specification["value_fn"])
            if name not in (boundary_values or {}):
                raise ValueError(f"Boundary value function '{name}' not found")
            value = boundary_values[name](columns, time_tensor)
        else:
            value = torch.tensor(self._number(specification, "value", 0.0))
        value = torch.as_tensor(value, dtype=torch.float32).detach()
        if value.numel() == 1:
            value = value.expand(len(spatial), 1)
        if value.shape != (len(spatial), 1) or not torch.isfinite(value).all():
            raise ValueError("Boundary value must be finite with shape (N, 1), or a scalar")
        values = cast(FloatArray, value.cpu().numpy().reshape(-1))
        return values

    def _face_mask(self, constraint: Constraint, side: str) -> NDArray[np.bool_]:
        spatial, mesh, axis, kind = self.spatial, self.mesh, constraint.axis, constraint.kind
        mask = np.zeros(len(spatial), dtype=bool)
        if side in ("lower", "both"):
            mask |= spatial[:, axis] == mesh.lb[axis]
        if side in ("upper", "both"):
            mask |= spatial[:, axis] == mesh.ub[axis]
        if kind != "dirichlet":
            mask &= self.face_count == 1
        if not mask.any():
            raise ValueError(
                "Boundary has no eligible points; increase tangential axis point counts"
            )
        return mask

    def _merge(
        self, constraint: Constraint, values: FloatArray, mask: NDArray[np.bool_], weight: float
    ) -> None:
        kind, field = constraint.kind, constraint.field
        dirichlet_groups, targets, weights, constraints = (
            self.dirichlet_groups,
            self.targets,
            self.weights,
            self.constraints,
        )
        if kind == "dirichlet" and field in dirichlet_groups:
            group = dirichlet_groups[field]
            overlap = mask & (weights[group] > 0)
            if not np.allclose(targets[group][overlap], values[overlap], rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"Conflicting Dirichlet values for '{field}' at shared boundary points"
                )
            targets[group][mask] = values[mask]
            weights[group][mask] = np.maximum(weights[group][mask], weight)
        else:
            group = len(constraints)
            constraints.append(constraint)
            targets.append(np.where(mask, values, 0.0))
            weights.append(np.where(mask, weight, 0.0))
            if kind == "dirichlet":
                dirichlet_groups[field] = group


class BoundaryConditions(SamplerBase):
    """Deduplicate Dirichlet faces; derivative constraints use face interiors."""

    def __init__(
        self,
        mesh: MeshBase,
        conditions: Sequence[Mapping[str, object]],
        boundary_values: Mapping[str, BoundaryValue] | None = None,
        coefficient_values: Mapping[str, BoundaryValue] | None = None,
    ) -> None:
        super().__init__()
        if not conditions:
            raise ValueError("BoundaryConditions requires at least one condition")
        plan = BoundaryPlan(mesh, boundary_values, coefficient_values)
        for specification in conditions:
            plan.add(specification)
        self.constraints = plan.constraints
        self.parameter_indices = plan.parameter_indices
        self.lower = tuple(float(value) for value in mesh.lb[: mesh.spatial_dim])
        self._pack(plan)

    def _pack(self, plan: BoundaryPlan) -> None:
        union = np.any(np.column_stack(plan.weights) > 0, axis=1)
        self.spatial_domain_sampled = [column[union] for column in plan.columns]
        self.time_domain_sampled = plan.time_tensor[union]
        self.solution_names = []
        self.solution_sampled = []
        for i, (target, weight) in enumerate(zip(plan.targets, plan.weights)):
            arrays = [(f"target_{i}", target), (f"weight_{i}", weight)]
            conductivity = plan.conductivities[i]
            if conductivity is not None:
                arrays.append((f"conductivity_{i}", conductivity))
            for name, array in arrays:
                self.solution_names.append(name)
                self.solution_sampled.append(
                    torch.as_tensor(array[union, None], dtype=torch.float32)
                )

    def _loss_fn(
        self, inputs: SamplerBatch, loss: LossAccumulator
    ) -> tuple[LossAccumulator, TensorOutputs]:
        columns, time, targets = inputs
        outputs = self.functions["forward"](columns, time)
        for i, constraint in enumerate(self.constraints):
            if constraint.field not in outputs:
                raise ValueError(
                    f"Boundary field '{constraint.field}' is not returned by the model/output_fn"
                )
            residual = (
                self._operator(
                    constraint,
                    outputs[constraint.field],
                    columns,
                    targets.get(f"conductivity_{i}"),
                )
                - targets[f"target_{i}"]
            )
            weight = targets[f"weight_{i}"]
            count = weight.gt(0).sum().clamp_min(1)
            loss = loss + (weight * residual.square()).sum() / count
        return loss, outputs

    def _operator(
        self,
        constraint: Constraint,
        field: torch.Tensor,
        columns: list[torch.Tensor],
        prescribed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if constraint.kind == "dirichlet":
            return field
        derivative = self._normal_derivative(
            field, columns[constraint.axis], self.lower[constraint.axis]
        )
        if constraint.kind == "neumann":
            return derivative
        if constraint.kind == "robin":
            return cast(torch.Tensor, constraint.alpha * field + constraint.beta * derivative)
        if prescribed is not None:
            return cast(torch.Tensor, -prescribed * derivative)
        coefficient = constraint.conductivity
        conductivity = (
            columns[self.parameter_indices[coefficient]]
            if isinstance(coefficient, str)
            else coefficient
        )
        return cast(torch.Tensor, -conductivity * derivative)

    @staticmethod
    def _normal_derivative(
        field: torch.Tensor, coordinate: torch.Tensor, lower: float
    ) -> torch.Tensor:
        derivative = None
        if field.requires_grad:
            derivative = torch.autograd.grad(
                field,
                coordinate,
                torch.ones_like(field),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
        if derivative is None:
            derivative = torch.zeros_like(coordinate)
        normal = torch.where(coordinate.eq(lower), -1.0, 1.0)
        return cast(torch.Tensor, normal * derivative)
