"""Offline preflight and no-contact rejection using a registered real protocol."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from research_tools.planning import plan
from research_tools.schema import read_object

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/experiments/2026-10-17-queue-probe-protocol.json"


@pytest.fixture
def manifest() -> dict[str, Any]:
    return read_object(PROTOCOL)


def rename_case(value: dict[str, Any], name: str) -> None:
    old = next(iter(value["case_limits"]))
    value["case_limits"] = {name: value["case_limits"][old]}
    value["case_faces"] = {name: value["case_faces"][old]}
    for trial in value["expected_trials"]:
        trial["case"] = name


def test_plan_accounts_for_calls_not_unknown_closures(manifest: dict[str, Any]) -> None:
    result = plan(manifest)
    assert result["trials"] == 2
    assert result["adam_steps_total"] == 2
    assert result["lbfgs_calls_total"] == 2
    assert result["saved_states_total"] == 4
    assert "seconds" not in result
    assert result["quality_confirmation_possible"]
    assert result["catalog_scope"] == "current_tool_version"


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_case",
        "no_modal_metric",
        "missing_face",
        "architecture",
        "precision",
        "metric",
        "schedule",
        "constraints",
    ],
)
def test_preflight_rejects_unexecutable_protocol(manifest: dict[str, Any], damage: str) -> None:
    case = next(iter(manifest["case_limits"]))
    if damage == "unknown_case":
        rename_case(manifest, "unknown")
    elif damage == "no_modal_metric":
        rename_case(manifest, "heat_neumann_polynomial2d")
    elif damage == "missing_face":
        manifest["case_faces"][case].pop()
    elif damage == "architecture":
        manifest["network"]["hidden_layers"] = [64, 64]
    elif damage == "precision":
        manifest["environment_requirements"]["default_dtype"] = "torch.float64"
    elif damage == "metric":
        manifest["case_limits"][case]["unknown_error"] = 1
    elif damage == "schedule":
        manifest["report_parameters"]["lbfgs_snapshots"] = [1]
    else:
        manifest["report_parameters"]["constraints"] = "soft"
    with pytest.raises(ValueError):
        plan(manifest)


def test_nested_output_directories_are_rejected(manifest: dict[str, Any]) -> None:
    case = next(iter(manifest["case_limits"]))
    second = "heat_neumann_homogeneous3d"
    manifest["case_limits"][second] = copy.deepcopy(manifest["case_limits"][case])
    manifest["case_faces"][second] = manifest["case_faces"][case] + [
        dict(field="u", kind="neumann", axis="z", side=side) for side in ("lower", "upper")
    ]
    manifest["expected_trials"] += [
        t | {"case": second} for t in list(manifest["expected_trials"])
    ]
    manifest["reports"] = ["a/report.json", "a/b/report.json"]
    with pytest.raises(ValueError, match="nested"):
        plan(manifest)


def test_default_plan_bounds_seed_output(manifest: dict[str, Any]) -> None:
    original = manifest["expected_trials"]
    manifest["expected_trials"] = [t | {"seed": seed} for seed in range(20) for t in original]
    result = plan(manifest)
    assert result["trials"] == 40
    assert result["cases"][0]["seeds"] == [0, 1, 2, 3, 4]
    assert result["cases"][0]["seeds_truncated"]
    assert len(plan(manifest, details=True)["cases"][0]["seeds"]) == 20


def test_cli_plan_works_without_site_packages_or_torch(tmp_path: Path) -> None:
    command = (
        "import sys; from scripts.research import main; result=main(['plan','--manifest',"
        + repr(str(PROTOCOL))
        + "]); assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; raise SystemExit(result)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-c", command], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["trials"] == 2
    assert list(tmp_path.iterdir()) == []


def test_invalid_submit_does_not_contact_windows(
    manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_tools import control

    rename_case(manifest, "unknown")
    monkeypatch.setattr(
        control.subprocess, "check_output", lambda *a, **kw: json.dumps(manifest).encode()
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No network for invalid protocol")

    monkeypatch.setattr(control, "call", forbidden)
    with pytest.raises(ValueError, match="registered"):
        control.submit(ROOT, "a" * 40, "docs/protocol.json", "bad-plan", 60, False)
