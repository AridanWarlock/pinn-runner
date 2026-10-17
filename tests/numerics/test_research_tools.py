"""Evidence integrity, incomplete runs and scientific distinctions without Torch."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from research_tools.comparison import compare
from research_tools.evidence import collect, status, write_bundle
from research_tools.schema import load_manifest
from scripts.research import main


@pytest.fixture
def study(tmp_path: Path) -> tuple[Path, Path]:
    control = dict(
        backend="eager",
        lifting="recursive",
        backward="parameters",
        background="auto",
        spatial_encoding="neumann_polynomial",
    )
    candidate = control | {"backend": "neumann_analytic"}
    limits = dict(relative_l2=0.01, final_relative_l2=0.02, pde_rmse=0.03)
    parameters = dict(
        adam_steps=2,
        lbfgs_snapshots=[0, 1, 2],
        constraints="hard_initial_robin",
        points_override=8,
        lbfgs_tolerance_change=1e-12,
        execution_lifecycle="isolated_trials",
    )
    manifest = dict(
        contract=1,
        registration="prospective",
        revision="a" * 40,
        reports=["case/report.json"],
        expected_trials=[dict(case="heat", seed=19, **arm) for arm in (control, candidate)],
        report_parameters=parameters,
        case_limits={"heat": limits},
        physical_limits=dict(initial_rmse=1e-6, boundary_max=1e-5),
        comparison=dict(
            baseline=control,
            candidate=candidate,
            quality_factor=1.5,
            median_training_ratio_max=0.95,
            worst_training_ratio_max=1.1,
        ),
    )
    trials: list[dict[str, Any]] = []
    for arm in (control, candidate):
        snapshots = [
            dict(
                lb_steps=i,
                seconds=i + 1,
                closures=i,
                metrics=dict(
                    relative_l2=0.001, final_relative_l2=0.002, pde_rmse=0.003, initial_rmse=0
                ),
                boundary_operators=[dict(max_error=1e-8)],
                passed=False,
            )
            for i in range(3)
        ]
        trials.append(
            dict(
                case="heat",
                seed=19,
                **arm,
                feature_derivatives="analytic" if arm == candidate else "autograd",
                control_feature_derivatives="autograd",
                control_prediction_execution="eager",
                control_prediction_background="general",
                initial_state_sha256="b" * 64,
                fixed_objective_sha256="c" * 64,
                training_seconds_with_snapshots=8 if arm == candidate else 10,
                warm_prediction_seconds=0.01,
                prediction_points=10,
                prediction_batch_size=10,
                prediction_warmup_passes=3,
                prediction_measurement_passes=7,
                adam_steps=2,
                lbfgs_tolerance_change=1e-12,
                snapshots=snapshots,
            )
        )
    report = dict(
        contract=3,
        status="completed",
        case="heat",
        environment=dict(revision="a" * 40, working_tree_dirty=False),
        limits=limits,
        **parameters,
        trials=trials,
    )
    (tmp_path / "case").mkdir()
    (tmp_path / "case/report.json").write_text(json.dumps(report))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return tmp_path, manifest_path


def evidence(study: tuple[Path, Path]) -> dict[str, Any]:
    root, manifest_path = study
    return collect(root, load_manifest(manifest_path), manifest_path)


def edit_json(path: Path, change: Any) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_complete_evidence_recomputes_crossings_and_writes_no_raw_arrays(
    study: tuple[Path, Path],
) -> None:
    value = evidence(study)
    result = compare(value)
    assert result["comparison_passed"]
    assert result["cases"][0]["median_training_ratio"] == 0.8
    assert result["cases"][0]["candidate_confirmed"] == 1
    assert value["trials"][0]["crossing"]["left_censored"]
    output = study[0] / "bundle"
    write_bundle(output, value)
    assert set(p.name for p in output.iterdir()) == {"evidence.json", "final.csv", "snapshots.csv"}
    assert len((output / "snapshots.csv").read_text().splitlines()) == 7
    with pytest.raises(FileExistsError):
        write_bundle(output, value)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="running"),
        lambda r: r.update(adam_steps=99),
        lambda r: r["environment"].update(revision="d" * 40),
        lambda r: r["environment"].update(working_tree_dirty=True),
        lambda r: r["trials"].pop(),
        lambda r: r["trials"].append(copy.deepcopy(r["trials"][0])),
        lambda r: r["trials"][1]["snapshots"].pop(),
        lambda r: r["trials"][1].update(control_feature_derivatives="analytic"),
        lambda r: r["trials"][1].update(feature_derivatives="autograd"),
        lambda r: r["trials"][1]["snapshots"][0]["metrics"].update(pde_rmse=float("nan")),
        lambda r: r["trials"][1]["snapshots"][0].update(boundary_operators=[]),
    ],
)
def test_invalid_source_never_becomes_success(study: tuple[Path, Path], change: Any) -> None:
    edit_json(study[0] / "case/report.json", change)
    with pytest.raises(ValueError):
        evidence(study)


@pytest.mark.parametrize(
    "key", ["initial_state_sha256", "fixed_objective_sha256", "prediction_points"]
)
def test_incomparable_pairs_fail_before_writing(
    study: tuple[Path, Path], key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_json(
        study[0] / "case/report.json",
        lambda r: r["trials"][1].update({key: 99 if key == "prediction_points" else "d" * 64}),
    )
    root, manifest = study
    assert (
        main(
            [
                "collect",
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--output",
                str(root / "bad"),
            ]
        )
        == 2
    )
    assert not (root / "bad").exists()
    assert "incomparable" in capsys.readouterr().err


def test_preservation_is_not_absolute_accuracy(study: tuple[Path, Path]) -> None:
    edit_json(
        study[0] / "case/report.json",
        lambda r: [s["metrics"].update(pde_rmse=0.1) for t in r["trials"] for s in t["snapshots"]],
    )
    result = compare(evidence(study))
    assert result["comparison_passed"]
    assert result["cases"][0]["candidate_confirmed"] == 0
    assert result["cases"][0]["candidate_final_target_passed"] == 0


def test_physical_and_timing_failures_are_reported(study: tuple[Path, Path]) -> None:
    value = evidence(study)
    value["trials"][1]["training_seconds_with_snapshots"] = 20
    value["trials"][1]["snapshots"][0]["boundary_max"] = 0.1
    result = compare(value)
    assert not result["comparison_passed"]
    assert not result["cases"][0]["physical_passed"]
    assert not result["cases"][0]["timing_passed"]


@pytest.mark.parametrize(
    "relative", ["../report.json", "/tmp/report.json", "C:\\report.json", "case/../../report.json"]
)
def test_manifest_rejects_path_escape(study: tuple[Path, Path], relative: str) -> None:
    edit_json(study[1], lambda m: m.update(reports=[relative]))
    with pytest.raises(ValueError):
        load_manifest(study[1])


def test_manifest_rejects_duplicate_or_missing_pair(study: tuple[Path, Path]) -> None:
    edit_json(study[1], lambda m: m["expected_trials"].pop())
    with pytest.raises(ValueError, match="exactly one"):
        load_manifest(study[1])


def test_status_and_cli_do_not_import_training_runtime(study: tuple[Path, Path]) -> None:
    root = Path(__file__).resolve().parents[2]
    command = "import sys; from scripts.research import main; assert 'torch' not in sys.modules; raise SystemExit(main(['--help']))"
    process = subprocess.run(
        [sys.executable, "-c", command], cwd=root, capture_output=True, text=True
    )
    assert process.returncode == 0
    with pytest.raises(ValueError, match="unknown"):
        status(study[0])


def test_task_brief_excludes_history(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_tools import board

    response = dict(
        schemaVersion=1,
        kind="task-view",
        task=dict(id="TASK-21", documentation=["docs/req.md"], implementationNotes="x" * 100000),
    )
    monkeypatch.setattr(
        board.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 0, stdout=json.dumps(response)),
    )
    result = board.brief(Path("."), "TASK-21")
    assert result["documentation"] == ["docs/req.md"]
    assert "implementationNotes" not in result


def test_remote_status_uses_guarded_existing_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_tools import remote_status as module

    calls: list[str] = []

    def read(script: str) -> str:
        calls.append(script)
        return json.dumps(dict(state="completed", commit="a" * 40))

    monkeypatch.setattr(module, "remote", read)
    assert module.remote_status("numerical-reports/a008")["state"] == "completed"
    assert "C:\\Projects\\pinn-runner\\numerical-reports\\a008\\status.json" in calls[0]
    for path in (
        "../",
        "numerical-reports/a';exit",
        "C:\\Windows",
        "numerical-reports/../outside",
    ):
        with pytest.raises(ValueError):
            module.remote_status(path)


def test_cli_distinguishes_failed_hypothesis_from_invalid_data(
    study: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, manifest = study
    edit_json(
        root / "case/report.json",
        lambda r: r["trials"][1].update(training_seconds_with_snapshots=20),
    )
    output = root / "failed-hypothesis"
    assert (
        main(
            ["collect", "--root", str(root), "--manifest", str(manifest), "--output", str(output)]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["compare", "--bundle", str(output)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert not summary["comparison_passed"]
    assert "details" not in summary["cases"][0]
    edit_json(
        output / "evidence.json",
        lambda e: e["trials"][1]["snapshots"][0]["metrics"].update(pde_rmse=float("nan")),
    )
    assert main(["compare", "--bundle", str(output)]) == 2
    assert "finite" in capsys.readouterr().err


def test_cached_crossing_cannot_fake_two_successes(study: tuple[Path, Path]) -> None:
    value = evidence(study)
    candidate = value["trials"][1]
    candidate["crossing"] = {"status": "observed"}
    for snap in candidate["snapshots"]:
        snap["metrics"]["pde_rmse"] = 0.1
    result = compare(value)
    assert result["cases"][0]["candidate_confirmed"] == 0
    assert not result["cases"][0]["confirmed_crossing_preserved"]


def test_symbolic_link_cannot_escape_study(study: tuple[Path, Path], tmp_path: Path) -> None:
    root, manifest = study
    target = tmp_path / "external-report.json"
    target.write_text("{}")
    # A distinct study root is needed because the fixture root is tmp_path itself.
    inner = root / "inner"
    inner.mkdir()
    (inner / "case").symlink_to(root / "case", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        collect(inner, load_manifest(manifest), manifest)


@pytest.mark.parametrize("damage", ["missing_face", "duplicate_face", "dtype", "architecture"])
def test_contract_two_checks_geometry_precision_and_architecture(
    study: tuple[Path, Path], damage: str
) -> None:
    root, manifest_path = study
    manifest = json.loads(manifest_path.read_text())
    faces = [dict(field="u", kind="neumann", axis="x", side=s) for s in ("lower", "upper")]
    manifest.update(
        contract=2,
        case_faces={"heat": faces},
        environment_requirements=dict(
            torch="2.1.0",
            cuda_runtime="11.8",
            default_dtype="torch.float32",
            float32_matmul_precision="highest",
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
        ),
        network=dict(hidden_layers=[32, 32, 32]),
    )
    manifest_path.write_text(json.dumps(manifest))
    report_path = root / "case/report.json"
    report = json.loads(report_path.read_text())
    report["environment"].update(manifest["environment_requirements"])
    for trial in report["trials"]:
        trial["network"] = manifest["network"]
        for snap in trial["snapshots"]:
            snap["boundary_operators"] = [f | {"max_error": 1e-8} for f in faces]
    report_path.write_text(json.dumps(report))
    assert compare(evidence(study))["comparison_passed"]
    if damage == "dtype":
        report["environment"]["default_dtype"] = "torch.float64"
    elif damage == "architecture":
        report["trials"][0]["network"] = {"hidden_layers": [1]}
    elif damage == "missing_face":
        report["trials"][0]["snapshots"][0]["boundary_operators"].pop()
    else:
        report["trials"][0]["snapshots"][0]["boundary_operators"] *= 2
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        evidence(study)
