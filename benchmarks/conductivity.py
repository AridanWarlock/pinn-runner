"""Explicit physical identity and self-contained affine-conductivity sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

REPRESENTATIONS = ("formula", "table")
CASES = {"heat_conductivity_beta0": 0, "heat_conductivity_beta1": 1}
SOURCE = Path(__file__).parent / "problems/heat_variable_conductivity/functions.py"


def materialized_source(case: str, representation: str) -> str:
    if case not in CASES or representation not in REPRESENTATIONS:
        raise ValueError("Unknown conductivity case/representation")
    source = SOURCE.read_text(encoding="utf-8")
    if source.count("BETA = 1") != 1 or source.count('REPRESENTATION = "formula"') != 1:
        raise ValueError("Conductivity template constants changed")
    return source.replace("BETA = 1", f"BETA = {CASES[case]}").replace(
        'REPRESENTATION = "formula"', f'REPRESENTATION = "{representation}"'
    )


def physics(case: str) -> dict[str, Any]:
    if case not in CASES:
        raise ValueError("Unknown conductivity physics")
    return dict(
        equation="dt(u)-div(kappa*grad(u))=manufactured_source",
        dimensions=2,
        volumetric_heat_capacity=1,
        beta=CASES[case],
        kappa0=0.125,
        space=[[0, 1], [0, 1]],
        time=[0, 1],
        initial="product(3*x_i**2-2*x_i**3)",
        boundary="zero_normal_heat_flux",
        template_sha256=hashlib.sha256(SOURCE.read_text(encoding="utf-8").encode()).hexdigest(),
    )


def source_sha256(case: str, representation: str) -> str:
    return hashlib.sha256(materialized_source(case, representation).encode()).hexdigest()
