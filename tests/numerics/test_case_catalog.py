"""Catalog migration preserves published configurations and analytical fields."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.cases import CASES
from benchmarks.catalog import CATALOG, CONVERGENCE_CASES
from research_tools.schema import face_identity


def test_all_published_case_configs_and_references_are_preserved() -> None:
    path = Path(__file__).parent / "fixtures/case-catalog-2026-10-17.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert set(CASES) == {r["name"] for r in rows} | {
        "heat_constant_surrogate",
        "heat_conductivity_beta0",
        "heat_conductivity_beta1",
        "burgers_unforced_nu010",
        "burgers_unforced_nu001",
    }
    for row in rows:
        case = CASES[row["name"]]
        assert type(case).__name__ == row["class"]
        assert case.dimensions == int(row["dimensions"])
        config = case.config("cpu", 2, 19)
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == row["config_sha256"], case.name
        spatial = {
            axis: np.array([0.2, 0.5, 0.8]).reshape(-1, 1) + i * 0.1
            for i, axis in enumerate(("x", "y", "z")[: case.dimensions])
        }
        actual = case.reference(spatial, np.array([0.0, 0.25, 1.0]).reshape(-1, 1)).reshape(-1)
        np.testing.assert_allclose(
            actual, [float(row[f"u{i}"]) for i in range(3)], rtol=1e-14, atol=1e-14
        )


def test_light_catalog_faces_and_shapes_match_problem_implementations() -> None:
    assert set(CASES) == set(CATALOG)
    assert set(CONVERGENCE_CASES) == {
        name for name in CASES if name.startswith("heat") and name != "heat_constant_surrogate"
    }
    for name, case in CASES.items():
        spec = CATALOG[name]
        config = case.config("cpu", 2, 19)
        assert config["network"] == dict(hidden_layers=list(spec.layers))
        actual = []
        for boundary in config["boundaries"]:
            for side in [boundary["side"]] if "side" in boundary else ["lower", "upper"]:
                actual.append({k: boundary[k] for k in ("field", "kind", "axis")} | {"side": side})
        assert {face_identity(f) for f in actual} == {face_identity(f) for f in spec.faces()}, name
