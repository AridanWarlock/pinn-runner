"""A coefficient comparison fixes physics while allowing different physical buffers."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.catalog import CATALOG
from benchmarks.conductivity import materialized_source, physics, source_sha256
from research_tools.comparison import compare
from research_tools.evidence import collect
from research_tools.planning import plan
from research_tools.schema import load_manifest, read_object

ROOT = Path(__file__).resolve().parents[2]


def manifest() -> dict[str, Any]:
    value = read_object(ROOT / "docs/experiments/2026-10-17-queue-probe-protocol.json")
    case = "heat_conductivity_beta1"
    arms = [
        dict(
            backend="eager",
            lifting="recursive",
            backward="all",
            background="general",
            spatial_encoding="neumann_polynomial",
            coefficient_representation=r,
        )
        for r in ("formula", "table")
    ]
    value.update(
        physics={case: physics(case)},
        case_limits={
            case: dict(
                relative_l2=0.002,
                final_relative_l2=0.003,
                pde_rmse=0.003,
                worst_time_relative_l2=0.005,
                physical_flux_max=2e-5,
            )
        },
        case_faces={case: CATALOG[case].faces()},
        expected_trials=[dict(case=case, seed=19, **a) for a in arms],
        comparison=dict(baseline=arms[0], candidate=arms[1], quality_factor=1.5),
    )
    return value


def test_coefficient_preflight_is_absolute_quality_not_speed() -> None:
    result = plan(manifest())
    assert result["comparison_purpose"] == "absolute_accuracy"
    assert result["trials"] == 2
    assert physics("heat_conductivity_beta0") != physics("heat_conductivity_beta1")
    assert source_sha256("heat_conductivity_beta1", "formula") != source_sha256(
        "heat_conductivity_beta1", "table"
    )
    assert "BETA = 0" in materialized_source("heat_conductivity_beta0", "table")


@pytest.mark.parametrize(
    "damage", ["beta", "source", "representation", "backend", "missing_flux", "density"]
)
def test_coefficient_preflight_rejects_mixed_science(damage: str) -> None:
    value = manifest()
    case = next(iter(value["physics"]))
    if damage == "beta":
        value["physics"][case]["beta"] = 0
    elif damage == "source":
        value["physics"][case]["template_sha256"] = "0" * 64
    elif damage == "representation":
        value["comparison"]["candidate"]["coefficient_representation"] = "formula"
    elif damage == "backend":
        value["comparison"]["candidate"]["backend"] = "neumann_analytic"
        value["expected_trials"][1]["backend"] = "neumann_analytic"
    elif damage == "missing_flux":
        del value["case_limits"][case]["physical_flux_max"]
    else:
        value["density"] = {}
    with pytest.raises(ValueError):
        plan(value)


def test_old_protocol_still_has_strict_full_state_pairing() -> None:
    value = load_manifest(ROOT / "docs/experiments/2026-10-17-queue-probe-protocol.json")
    assert plan(value)["comparison_purpose"] == "paired_cost"


def test_quality_comparison_rejects_tampered_physics_and_requires_absolute_accuracy(
    tmp_path: Path,
) -> None:
    import json

    value = manifest()
    case = next(iter(value["physics"]))
    parameters = value["report_parameters"]
    limits = value["case_limits"][case]
    trials = []
    for i, arm in enumerate(value["expected_trials"]):
        trials.append(
            dict(
                **arm,
                physics=value["physics"][case],
                problem_source_sha256=source_sha256(case, arm["coefficient_representation"]),
                network=value["network"],
                feature_derivatives="autograd",
                control_feature_derivatives="autograd",
                control_prediction_execution="eager",
                control_prediction_background="general",
                initial_state_sha256=str(i + 1) * 64,
                initial_parameters_sha256="a" * 64,
                fixed_objective_sha256="b" * 64,
                training_seconds_with_snapshots=100 + i * 100,
                warm_prediction_seconds=0.01,
                prediction_points=10,
                prediction_batch_size=10,
                prediction_warmup_passes=3,
                prediction_measurement_passes=7,
                adam_steps=parameters["adam_steps"],
                lbfgs_tolerance_change=parameters["lbfgs_tolerance_change"],
                snapshots=[
                    dict(
                        lb_steps=step,
                        seconds=j + 1,
                        closures=j,
                        metrics={k: 0.0 for k in limits} | {"initial_rmse": 0.0},
                        boundary_operators=[
                            f | {"max_error": 0.0} for f in value["case_faces"][case]
                        ],
                    )
                    for j, step in enumerate(parameters["lbfgs_snapshots"])
                ],
            )
        )
    path = tmp_path / value["reports"][0]
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                contract=3,
                status="completed",
                case=case,
                limits=limits,
                environment=dict(
                    revision=value["revision"],
                    working_tree_dirty=False,
                    **value["environment_requirements"],
                ),
                **parameters,
                trials=trials,
            )
        )
    )
    protocol = tmp_path / "manifest.json"
    protocol.write_text(json.dumps(value))
    evidence = collect(tmp_path, value, protocol)
    assert compare(evidence)["comparison_passed"]  # Even a 2x timing ratio is diagnostic only.
    for key, replacement in (
        ("physics", physics("heat_conductivity_beta0")),
        ("problem_source_sha256", "0" * 64),
        ("initial_parameters_sha256", "0" * 64),
    ):
        damaged = deepcopy(evidence)
        damaged["trials"][1][key] = replacement
        with pytest.raises(ValueError):
            compare(damaged)
    for metric in limits:
        damaged = deepcopy(evidence)
        damaged["trials"][0]["snapshots"][-1]["metrics"][metric] = limits[metric] * 2
        assert not compare(damaged)["comparison_passed"], metric
