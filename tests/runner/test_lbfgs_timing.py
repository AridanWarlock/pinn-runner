"""Protocol integrity and raw counter validation; numerical tests are separate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from research_tools import lbfgs_timing
from research_tools.comparison import compare
from research_tools.evidence import read_bundle, write_bundle
from research_tools.planning import plan
from research_tools.schema import validate_manifest
from tests.runner.test_frozen_timing import protocol as timing_protocol


def protocol() -> dict[str, Any]:
    old = timing_protocol()
    sources = [
        old["sources"][0] | {"case": "heat_neumann_polynomial3d", "seed": seed}
        for seed in (19, 4321)
    ]
    return dict(
        contract=5,
        kind="lbfgs-timing",
        registration="prospective",
        revision=old["revision"],
        sources=sources,
        points=1024,
        repeats=3,
        steps=10,
        environment_requirements=old["environment_requirements"],
        comparison=copy.deepcopy(lbfgs_timing.GATES),
    )


def evidence() -> dict[str, Any]:
    manifest = protocol()
    trials = []
    for i, r, rule in lbfgs_timing.expected(manifest):
        seconds = 0.01 if rule == "autograd" else 0.008
        row = dict(
            seconds=seconds,
            closure_seconds=seconds * 0.9,
            outside_closure_seconds=seconds * 0.1,
            closure_calls=2,
            optimizer_evaluations=2,
            optimizer_iterations=1,
        )
        attributed = dict(
            steps=10,
            seconds=seconds * 10,
            closure_seconds=seconds * 9,
            closure_calls=20,
            rows=[copy.deepcopy(row) for _ in range(10)],
        )
        control = dict(
            loss_absolute_difference=0,
            gradient_relative_l2_difference=0,
            gradient_max_absolute_difference=0,
        )
        trials.append(
            dict(
                source_index=i,
                repeat=r,
                rule=rule,
                source=manifest["sources"][i],
                revision=manifest["revision"],
                environment=manifest["environment_requirements"],
                source_unchanged=True,
                restored=True,
                timing_paths_match=True,
                initial_state_sha256="a" * 64,
                restored_state_sha256="a" * 64,
                final_state_sha256=dict(total="d" * 64, attributed="d" * 64),
                initial_field_sha256="b" * 64,
                points_sha256="c" * 64,
                total=dict(steps=10, seconds=seconds * 10, closure_calls=20),
                attributed=attributed,
                initial_control=control,
                final_control=control,
                field=[0.1] * 257,
                pde_rmse=0.001,
                relative_l2=0.0001,
            )
        )
    return dict(
        contract=1,
        kind="lbfgs-timing-evidence",
        manifest=manifest,
        trials=trials,
        artifact_sha256={},
    )


def test_plan_compare_bundle_and_detailed_summary(tmp_path: Path) -> None:
    data = evidence()
    assert validate_manifest(data["manifest"], tmp_path) == data["manifest"]
    assert plan(data["manifest"])["measured_outer_steps"] == 240
    result = compare(data)
    assert result["comparison_passed"]
    assert result["trials"] == 12
    assert result["median_total_ratio"] == pytest.approx(0.8)
    write_bundle(tmp_path / "bundle", data)
    assert compare(read_bundle(tmp_path / "bundle")) == result


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "counter",
        "overlap",
        "summary",
        "pair",
        "restore",
        "trajectory",
        "nan",
    ],
)
def test_reject_corrupt_evidence(mutation: str) -> None:
    data = evidence()
    t = data["trials"][0]
    if mutation == "missing":
        data["trials"].pop()
    elif mutation == "counter":
        t["attributed"]["rows"][0]["optimizer_evaluations"] = 3
    elif mutation == "overlap":
        t["attributed"]["rows"][0]["closure_seconds"] = 1
    elif mutation == "summary":
        t["attributed"]["seconds"] = 99
    elif mutation == "pair":
        t["points_sha256"] = "d" * 64
    elif mutation == "restore":
        t["restored"] = False
    elif mutation == "trajectory":
        t["timing_paths_match"] = False
    else:
        t["field"][0] = float("nan")
    with pytest.raises(ValueError):
        compare(data)


def test_quality_failure_is_collected_as_negative_result() -> None:
    data = evidence()
    data["trials"][0]["pde_rmse"] = 0.1
    result = compare(data)
    assert not result["numerical_passed"]
    assert not result["comparison_passed"]


def test_runner_and_collection_keep_exact_child_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_tools.schema import digest

    data = evidence()
    root = Path(r"C:\Projects\pinn-runner")
    lookup = {(t["source_index"], t["repeat"], t["rule"]): t for t in data["trials"]}
    commands = []

    def child(command: list[str], **kwargs: Any) -> None:
        commands.append(command)
        assert kwargs["cwd"] == root and command[1] == "-B"
        i = int(command[command.index("--source-index") + 1])
        r = int(command[command.index("--repeat") + 1])
        rule = command[command.index("--rule") + 1]
        target = Path(command[command.index("--output") + 1])
        target.write_text(json.dumps(lookup[i, r, rule]))

    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: root))
    monkeypatch.setattr(lbfgs_timing.subprocess, "run", child)
    output = tmp_path / "r"
    lbfgs_timing.run(root, tmp_path / "checkout", output, data["manifest"])
    assert len(commands) == 12
    result = lbfgs_timing.collect(output, data["manifest"])
    assert len(result["artifact_sha256"]) == 24
    # Even a rehashed child must agree with the parent scientific summary.
    report = output / "s0-r0-autograd.json"
    changed = json.loads(report.read_text())
    changed["relative_l2"] = 0.1
    report.write_text(json.dumps(changed))
    result["artifact_sha256"][report.name] = digest(report)
    (output / "lbfgs.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="summary differs"):
        lbfgs_timing.collect(output, data["manifest"])


@pytest.mark.parametrize("key,value", [("steps", True), ("steps", 11), ("repeats", 4)])
def test_reject_unregistered_budget(key: str, value: object) -> None:
    manifest = protocol()
    manifest[key] = value
    with pytest.raises(ValueError):
        lbfgs_timing.validate(manifest)
