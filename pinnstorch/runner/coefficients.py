"""Bind prescribed coefficients to named coordinates without per-point Python work."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch import nn

from pinnstorch.api import Coordinates, ValueFunction
from pinnstorch.coefficients import RegularGridCoefficient

from .settings import Domain, TableCoefficient


class CoefficientBank(nn.Module):
    def __init__(self, providers: Mapping[str, ValueFunction]) -> None:
        super().__init__()
        self.names = tuple(providers)
        self.tables = nn.ModuleDict(
            {
                f"field_{name}": provider
                for name, provider in providers.items()
                if isinstance(provider, RegularGridCoefficient)
            }
        )
        self.table_names = tuple(
            name
            for name, provider in providers.items()
            if isinstance(provider, RegularGridCoefficient)
        )
        if any(
            isinstance(provider, nn.Module) and not isinstance(provider, RegularGridCoefficient)
            for provider in providers.values()
        ):
            raise ValueError(
                "Coefficients must be pure callbacks or RegularGridCoefficient, not trainable modules"
            )
        self.formulas = {
            name: provider
            for name, provider in providers.items()
            if not isinstance(provider, RegularGridCoefficient)
        }

    @classmethod
    def build(
        cls,
        formulas: Mapping[str, ValueFunction],
        files: Mapping[str, TableCoefficient],
        directory: Path,
        domain: Domain,
    ) -> CoefficientBank:
        duplicate = set(formulas) & set(files)
        if duplicate:
            raise ValueError(f"Coefficients defined in both YAML and Problem: {sorted(duplicate)}")
        providers = dict(formulas)
        for name, specification in files.items():
            path = (directory / specification.file).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Coefficient file must be inside input-dir")
            providers[name] = RegularGridCoefficient.load(path)
        bank = cls(providers)
        bank.validate_domain(domain)
        return bank

    def validate_domain(self, domain: Domain) -> None:
        for name in self.table_names:
            table = self.table(name)
            for axis, (lower, upper) in table.bounds().items():
                if axis not in domain.space:
                    raise ValueError(
                        f"Coefficient '{name}' axis '{axis}' is absent from domain.space"
                    )
                requested = domain.space[axis].interval
                if requested[0] < lower or requested[1] > upper:
                    raise ValueError(f"Coefficient '{name}' does not cover domain.space.{axis}")

    def signature(self, source: bytes | None) -> dict[str, str]:
        result = {name: "grid:" + self.table(name).signature for name in self.table_names}
        if self.formulas:
            if source is None:
                raise ValueError(
                    "Formula coefficients require functions.py source for checkpoint identity"
                )
            digest = hashlib.sha256(source).hexdigest()
            result.update({name: "functions.py:" + digest for name in self.formulas})
        return result

    def context(self, coordinates: Coordinates) -> Coordinates:
        view = CoefficientView(self, coordinates)
        return replace(coordinates, coefficients=view)

    def table(self, name: str) -> RegularGridCoefficient:
        return cast(RegularGridCoefficient, self.tables[f"field_{name}"])

    def evaluate(self, name: str, coordinates: Coordinates) -> torch.Tensor:
        value: torch.Tensor | float
        if name in self.table_names:
            value = self.table(name).forward(coordinates)
        elif name in self.formulas:
            value = self.formulas[name](coordinates)
        else:
            raise ValueError(f"Unknown coefficient '{name}'")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full_like(coordinates.t, value)
        if not isinstance(value, torch.Tensor) or value.shape != coordinates.t.shape:
            raise ValueError(f"Coefficient '{name}' must return a scalar or Tensor column (N, 1)")
        if value.device != coordinates.t.device:
            raise ValueError(f"Coefficient '{name}' must remain on the coordinate device")
        return value


class CoefficientView(Mapping[str, torch.Tensor]):
    def __init__(self, bank: CoefficientBank, coordinates: Coordinates) -> None:
        self.bank, self.coordinates = bank, coordinates
        self.cache: dict[str, torch.Tensor] = {}
        self.resolving: set[str] = set()

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self.bank.names:
            raise KeyError(name)
        if name in self.resolving:
            raise ValueError(f"Cyclic coefficient dependency involving '{name}'")
        if name not in self.cache:
            self.resolving.add(name)
            try:
                self.cache[name] = self.bank.evaluate(
                    name, replace(self.coordinates, coefficients=self)
                )
            finally:
                self.resolving.remove(name)
        return self.cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.bank.names)

    def __len__(self) -> int:
        return len(self.bank.names)
