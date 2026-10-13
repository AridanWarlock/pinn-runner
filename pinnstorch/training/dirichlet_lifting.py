"""Boolean-sum interpolation of compatible Cartesian Dirichlet traces."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, product

import torch

from pinnstorch.contracts import TensorOutputs

TraceFunction = Callable[[list[torch.Tensor], torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FaceTrace:
    field: str
    axis: int
    side: int
    value: TraceFunction


@dataclass(frozen=True)
class ProjectionTerm:
    trace: FaceTrace
    projections: tuple[tuple[int, int], ...]
    sign: int


@dataclass(frozen=True)
class DirichletLifting:
    bounds: tuple[tuple[float, float], ...]
    fields: tuple[str, ...]
    terms: tuple[ProjectionTerm, ...]

    @classmethod
    def build(
        cls,
        bounds: tuple[tuple[float, float], ...],
        fields: tuple[str, ...],
        traces: tuple[FaceTrace, ...],
    ) -> "DirichletLifting":
        lookup = {(face.field, face.axis, face.side): face for face in traces}
        terms = []
        for field in fields:
            for degree in range(1, len(bounds) + 1):
                for axes in combinations(range(len(bounds)), degree):
                    for sides in product((0, 1), repeat=degree):
                        projections = tuple(zip(axes, sides))
                        terms.append(
                            ProjectionTerm(
                                lookup[field, axes[0], sides[0]], projections, (-1) ** (degree + 1)
                            )
                        )
        return cls(bounds, fields, tuple(terms))

    def project(self, spatial: list[torch.Tensor], axis: int, side: int) -> list[torch.Tensor]:
        projected = list(spatial)
        projected[axis] = torch.full_like(spatial[axis], self.bounds[axis][side])
        return projected

    def __call__(self, spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        normalized = [
            column.sub(left).div(right - left)
            for column, (left, right) in zip(spatial, self.bounds)
        ]
        result = {field: torch.zeros_like(time) for field in self.fields}
        for term in self.terms:
            projected = list(spatial)
            weight = torch.ones_like(time)
            for axis, side in term.projections:
                projected = self.project(projected, axis, side)
                weight = weight.mul(normalized[axis] if side else normalized[axis].neg().add(1))
            value = term.trace.value(projected, time)
            result[term.trace.field] = result[term.trace.field].add(
                weight.mul(value).mul(term.sign)
            )
        return result
