"""Compile the problem language into vectorized sampler callbacks."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from pinnstorch.api import Coordinates, Equation, Problem, TensorFields
from pinnstorch.contracts import ArrayOutputs, FloatArray

from .settings import Domain


def load_problem(path: Path) -> Problem:
    specification = importlib.util.spec_from_file_location("task_problem", path)
    if specification is None or specification.loader is None:
        raise ValueError(f"Cannot import problem from '{path}'")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    problem = getattr(module, "problem", None)
    if not isinstance(problem, Problem):
        raise ValueError("functions.py must export problem = Problem(...); see docs/API.md")
    return problem


@dataclass(frozen=True)
class InputLayout:
    spatial_names: tuple[str, ...]
    parameter_names: tuple[str, ...]

    @classmethod
    def from_domain(cls, domain: Domain) -> InputLayout:
        return cls(tuple(domain.space), tuple(domain.parameters))

    def coordinates(self, columns: list[torch.Tensor], time: torch.Tensor) -> Coordinates:
        expected = len(self.spatial_names) + len(self.parameter_names)
        if len(columns) != expected:
            raise ValueError(f"Expected {expected} input columns, received {len(columns)}")
        split = len(self.spatial_names)
        return Coordinates(
            dict(zip(self.spatial_names, columns[:split])),
            time,
            dict(zip(self.parameter_names, columns[split:])),
        )


class ProblemAdapter:
    """Convert public contexts once per batch; never copy PDE tensors to NumPy."""

    def __init__(self, problem: Problem, domain: Domain) -> None:
        self.problem = problem
        self.layout = InputLayout.from_domain(domain)
        self.initial_time = domain.time.interval[0]
        reserved = set(domain.input_names)
        if reserved & (set(problem.fields) | set(problem.outputs) | set(problem.equations)):
            raise ValueError("Field and equation names must differ from coordinates/parameters")

    @staticmethod
    def _column(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
            raise ValueError(
                f"'{name}' must return a Tensor column of shape {tuple(reference.shape)}"
            )
        if value.device != reference.device:
            raise ValueError(f"'{name}' must remain on the same device as its coordinates")
        return value

    def residuals(self, outputs: TensorFields, *columns: torch.Tensor) -> TensorFields:
        coordinates = self.layout.coordinates(list(columns[:-1]), columns[-1])
        equation = Equation(outputs, coordinates)
        residuals = {
            name: self._column(function(equation), coordinates.time, name)
            for name, function in self.problem.equations.items()
        }
        return {**outputs, **residuals}

    def transform(self, outputs: TensorFields, *columns: torch.Tensor) -> TensorFields:
        function = self.problem.transform
        if function is None:
            return outputs
        coordinates = self.layout.coordinates(list(columns[:-1]), columns[-1])
        derived = function(Equation(outputs, coordinates))
        self._validate_fields(derived, coordinates.time)
        return {**outputs, **derived}

    def _validate_fields(self, fields: Mapping[str, torch.Tensor], time: torch.Tensor) -> None:
        for name in self.problem.fields:
            if name not in fields:
                raise ValueError(f"Problem callback does not provide field '{name}'")
            self._column(fields[name], time, name)

    def initial(self, spatial: FloatArray) -> ArrayOutputs:
        tensor = torch.as_tensor(spatial, dtype=torch.float32)
        columns = [tensor[:, i : i + 1] for i in range(tensor.shape[1])]
        time = tensor.new_full((len(tensor), 1), self.initial_time)
        values = self.problem.initial(self.layout.coordinates(columns, time))
        self._validate_fields(values, time)
        return {
            name: cast(FloatArray, value.detach().cpu().numpy().astype(np.float64))
            for name, value in values.items()
        }

    def boundary_value(
        self, name: str, columns: list[torch.Tensor], time: torch.Tensor
    ) -> torch.Tensor:
        if name not in self.problem.boundary_values:
            raise ValueError(f"Unknown boundary value function '{name}'")
        result = self.problem.boundary_values[name](self.layout.coordinates(columns, time))
        return torch.as_tensor(result, device=time.device, dtype=time.dtype)
