"""Face conditions with explicit normals and ownership of intersections."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from pinnstorch.contracts import LossAccumulator, SamplerBatch, TensorOutputs
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


class BoundaryConditions(SamplerBase):
    """One dataset owns all configured faces and deduplicates Dirichlet points.

    Derivative constraints exclude intersections of physical faces. Weights
    at shared Dirichlet points use the maximum configured weight.
    """

    def __init__(
        self,
        mesh: MeshBase,
        conditions: Sequence[Mapping[str, object]],
        boundary_values: Mapping[str, BoundaryValue] | None = None,
    ) -> None:
        super().__init__()
        if not conditions:
            raise ValueError("BoundaryConditions requires at least one condition")
        spatial, time, _ = mesh.flatten_mesh(None)
        columns = [
            torch.as_tensor(spatial[:, i : i + 1], dtype=torch.float32)
            for i in range(spatial.shape[1])
        ]
        time_tensor = torch.as_tensor(time, dtype=torch.float32)
        names = tuple(getattr(mesh, "parameter_names", ()))
        self.parameter_indices = {name: mesh.spatial_dim + i for i, name in enumerate(names)}
        on_face = (spatial[:, : mesh.spatial_dim] == mesh.lb[: mesh.spatial_dim]) | (
            spatial[:, : mesh.spatial_dim] == mesh.ub[: mesh.spatial_dim]
        )
        targets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        constraints: list[Constraint] = []
        dirichlet_groups: dict[str, int] = {}
        for specification in conditions:
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
            weight = float(specification.get("weight", 1.0))
            alpha = float(specification.get("alpha", 1.0))
            beta = float(specification.get("beta", 1.0))
            conductivity = specification.get("conductivity", 1.0)
            if isinstance(conductivity, str):
                if conductivity not in self.parameter_indices:
                    raise ValueError(f"Unknown conductivity parameter '{conductivity}'")
                if mesh.lb[self.parameter_indices[conductivity]] <= 0:
                    raise ValueError("Conductivity parameter range must be positive")
            else:
                conductivity = float(conductivity)
                if not np.isfinite(conductivity) or conductivity <= 0:
                    raise ValueError("Conductivity must be finite and positive")
            if not np.isfinite([weight, alpha, beta]).all() or weight <= 0:
                raise ValueError(
                    "Boundary weight must be positive; weight/alpha/beta must be finite"
                )
            if kind == "robin" and alpha == 0 and beta == 0:
                raise ValueError("Robin alpha and beta cannot both be zero")
            if "value_fn" in specification and "value" in specification:
                raise ValueError("Specify either value or value_fn")
            if "value_fn" in specification:
                name = str(specification["value_fn"])
                if name not in (boundary_values or {}):
                    raise ValueError(f"Boundary value function '{name}' not found")
                value = boundary_values[name](columns, time_tensor)
            else:
                value = torch.tensor(float(specification.get("value", 0.0)))
            value = torch.as_tensor(value, dtype=torch.float32).detach()
            if value.numel() == 1:
                value = value.expand(len(spatial), 1)
            if value.shape != (len(spatial), 1) or not torch.isfinite(value).all():
                raise ValueError("Boundary value must be finite with shape (N, 1), or a scalar")
            values = value.cpu().numpy().reshape(-1)
            mask = np.zeros(len(spatial), dtype=bool)
            if side in ("lower", "both"):
                mask |= spatial[:, axis] == mesh.lb[axis]
            if side in ("upper", "both"):
                mask |= spatial[:, axis] == mesh.ub[axis]
            if kind != "dirichlet":
                mask &= on_face.sum(axis=1) == 1
            if not mask.any():
                raise ValueError(
                    "Boundary has no eligible points; increase tangential axis point counts"
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
                constraints.append(Constraint(field, kind, axis, alpha, beta, conductivity))
                targets.append(np.where(mask, values, 0.0))
                weights.append(np.where(mask, weight, 0.0))
                if kind == "dirichlet":
                    dirichlet_groups[field] = group
        self.constraints = constraints
        union = np.any(np.column_stack(weights) > 0, axis=1)
        self.spatial_domain_sampled = [column[union] for column in columns]
        self.time_domain_sampled = time_tensor[union]
        self.solution_names = []
        self.solution_sampled = []
        for i, (target, weight) in enumerate(zip(targets, weights)):
            for name, array in ((f"target_{i}", target), (f"weight_{i}", weight)):
                self.solution_names.append(name)
                self.solution_sampled.append(
                    torch.as_tensor(array[union, None], dtype=torch.float32)
                )
        self.lower = tuple(float(value) for value in mesh.lb[: mesh.spatial_dim])

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
            field = outputs[constraint.field]
            if constraint.kind == "dirichlet":
                residual = field - targets[f"target_{i}"]
            else:
                coordinate = columns[constraint.axis]
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
                normal = torch.where(coordinate == self.lower[constraint.axis], -1.0, 1.0)
                normal_derivative = normal * derivative
                if constraint.kind == "neumann":
                    residual = normal_derivative
                elif constraint.kind == "robin":
                    residual = constraint.alpha * field + constraint.beta * normal_derivative
                else:
                    conductivity = constraint.conductivity
                    if isinstance(conductivity, str):
                        conductivity = columns[self.parameter_indices[conductivity]]
                    residual = -conductivity * normal_derivative
                residual = residual - targets[f"target_{i}"]
            weight = targets[f"weight_{i}"]
            count = (weight > 0).sum().clamp_min(1)
            loss = loss + (weight * residual.square()).sum() / count
        return loss, outputs
