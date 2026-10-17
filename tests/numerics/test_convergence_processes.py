"""Fresh child processes and partial-report validation without neural training."""

import json
import subprocess
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.convergence import ConvergenceArm, ConvergencePlan
from benchmarks.convergence_processes import read_trial, run_isolated
from scripts import benchmark_time_to_quality as cli


@pytest.fixture
def child_script(tmp_path: Path) -> Path:
    script = tmp_path / "scripts" / "child.py"
    script.parent.mkdir()
    script.write_text("""import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path);p.add_argument('--case');p.add_argument('--seeds',type=int)
p.add_argument('--steps',type=int);p.add_argument('--snapshots',nargs='+',type=int)
p.add_argument('--constraints');p.add_argument('--hard-execution');p.add_argument('--hard-lifting');p.add_argument('--backward')
p.add_argument('--relative-l2',type=float);p.add_argument('--final-relative-l2',type=float);p.add_argument('--pde-rmse',type=float)
p.add_argument('--points',type=int);p.add_argument('--hard-background',default='general')
p.add_argument('--lbfgs-tolerance-change',type=float,default=1e-9)
p.add_argument('--spatial-encoding',default='identity')
a,_=p.parse_known_args()
a.output.mkdir(parents=True)
trial=dict(case=a.case,seed=a.seeds,backend=a.hard_execution,lifting=a.hard_lifting,backward=a.backward,background=a.hard_background,pid=os.getpid(),spatial_encoding=a.spatial_encoding,feature_derivatives='analytic' if a.hard_execution=='neumann_analytic' else 'autograd')
r=dict(contract=3,status='completed',execution_lifecycle='single_trial',case=a.case,environment=dict(revision='fixture',device='cpu'),limits=dict(relative_l2=a.relative_l2,final_relative_l2=a.final_relative_l2,pde_rmse=a.pde_rmse),adam_steps=a.steps,lbfgs_snapshots=a.snapshots,constraints=a.constraints,points_override=a.points,lbfgs_tolerance_change=a.lbfgs_tolerance_change,trials=[trial])
(a.output/'report.json').write_text(json.dumps(r))
""")
    return script


def plan() -> ConvergencePlan:
    return ConvergencePlan(
        CASES["heat_neumann2d"],
        2,
        (0, 1, 2),
        {"relative_l2": 0.1, "final_relative_l2": 0.2, "pde_rmse": 0.3},
        "hard_initial_robin",
        7,
        "cpu",
    )


def test_coordinator_starts_fresh_processes_and_reverses_order(
    tmp_path: Path, child_script: Path
) -> None:
    arms = [ConvergenceArm("robin_grouped", "recursive", mode) for mode in ("all", "parameters")]
    output = tmp_path / "study"
    report = run_isolated(plan(), arms, [19, 73], output, child_script)
    assert report["status"] == "completed"
    assert report["execution_lifecycle"] == "isolated_trials"
    assert [(t["seed"], t["backward"]) for t in report["trials"]] == [
        (19, "all"),
        (19, "parameters"),
        (73, "parameters"),
        (73, "all"),
    ]
    assert len({t["pid"] for t in report["trials"]}) == 4
    assert all(t["process_seconds"] > 0 for t in report["trials"])
    assert json.loads((output / "report.json").read_text()) == report
    assert not (output / "report.json.tmp").exists()
    with pytest.raises(FileExistsError):
        run_isolated(plan(), arms, [19], output, child_script)


def test_coordinator_preserves_partial_evidence_after_child_failure(
    tmp_path: Path, child_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.convergence_processes as module

    original = module.subprocess.run
    count = 0

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        nonlocal count
        count += 1
        if count == 2:
            raise subprocess.CalledProcessError(7, args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "partial"
    with pytest.raises(subprocess.CalledProcessError):
        run_isolated(plan(), [ConvergenceArm()], [19, 73], output, child_script)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "running"
    assert len(report["trials"]) == 1


def test_child_report_cannot_silently_change_protocol(tmp_path: Path, child_script: Path) -> None:
    output = tmp_path / "invalid"
    arm = ConvergenceArm()
    run_isolated(plan(), [arm], [19], output, child_script)
    child = output / "processes" / f"{arm.name}-19"
    report_path = child / "report.json"
    original = json.loads(report_path.read_text())
    for change in (
        {"adam_steps": 999},
        {"status": "running"},
        {"environment": {}},
        {"trials": []},
    ):
        report_path.write_text(json.dumps({**original, **change}))
        with pytest.raises(ValueError):
            read_trial(child, plan(), arm, 19)
    modified = json.loads(json.dumps(original))
    modified["trials"][0]["seed"] = 73
    report_path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="different seed"):
        read_trial(child, plan(), arm, 19)
    modified = json.loads(json.dumps(original))
    modified["trials"][0]["spatial_encoding"] = "neumann_polynomial"
    report_path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="different spatial"):
        read_trial(child, plan(), arm, 19)


def test_default_cli_coordinator_never_queries_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[Path] = []
    monkeypatch.setattr(
        cli,
        "arguments",
        lambda: (
            Namespace(shared_process=False, seeds=[19], output=tmp_path),
            plan(),
            [ConvergenceArm()],
        ),
    )
    monkeypatch.setattr(cli, "run_isolated", lambda p, a, s, output, script: called.append(output))

    def forbidden() -> bool:
        raise AssertionError("Coordinator must not query or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    cli.main()
    assert called == [tmp_path]


def test_coordinator_rejects_environment_change_before_appending(
    tmp_path: Path, child_script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.convergence_processes as module

    original = module.read_trial
    count = 0

    def read(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        report = original(*args, **kwargs)
        if count == 2:
            report["environment"]["revision"] = "unexpected-source"
        return report

    monkeypatch.setattr(module, "read_trial", read)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="environment changed"):
        run_isolated(plan(), [ConvergenceArm()], [19, 73], output, child_script)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "running"
    assert len(report["trials"]) == 1
    assert report["environment"]["revision"] == "fixture"


def test_encoding_cli_deduplicates_projection_free_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_time_to_quality.py",
            "--output",
            str(tmp_path / "out"),
            "--case",
            "heat_neumann_polynomial1d",
            "--seeds",
            "19",
            "--steps",
            "3",
            "--snapshots",
            "0",
            "1",
            "2",
            "--points",
            "7",
            "--constraints",
            "hard_initial_robin",
            "--hard-execution",
            "eager",
            "robin_grouped",
            "--spatial-encoding",
            "identity",
            "neumann_cosine",
            "neumann_polynomial",
        ],
    )
    _, _, arms = cli.arguments()
    assert len(arms) == 4
    assert {a.execution for a in arms if a.encoding != "identity"} == {"eager"}
    assert len({a.name for a in arms}) == 4


def test_mode_quality_limit_is_forwarded_to_fresh_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from benchmarks.convergence_processes import trial_command

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_time_to_quality.py",
            "--output",
            str(tmp_path / "study"),
            "--case",
            "heat_neumann_homogeneous1d",
            "--seeds",
            "19",
            "--constraints",
            "hard_initial_robin",
            "--mode-relative-l2",
            "0.02",
            "--lbfgs-tolerance-change",
            "1e-12",
            "--spatial-encoding",
            "neumann_polynomial",
        ],
    )
    _, selected, arms = cli.arguments()
    assert selected.limits["mode_relative_l2"] == 0.02
    assert selected.tolerance_change == 1e-12
    command = trial_command(Path("benchmark.py"), selected, arms[0], 19, tmp_path / "child")
    assert command[command.index("--mode-relative-l2") + 1] == "0.02"
    assert command[command.index("--lbfgs-tolerance-change") + 1] == "1e-12"
    assert command[command.index("--spatial-encoding") + 1] == "neumann_polynomial"


def test_tolerance_is_preserved_in_config_worker_and_report(
    tmp_path: Path, child_script: Path
) -> None:
    selected = replace(plan(), tolerance_change=1e-12)
    arm = ConvergenceArm("robin_grouped", "recursive", "parameters")
    assert selected.config(19, arm)["training"]["refinement"]["tolerance_change"] == 1e-12
    output = tmp_path / "small-tolerance"
    report = run_isolated(selected, [arm], [19], output, child_script)
    assert report["lbfgs_tolerance_change"] == 1e-12
    child = output / "processes" / f"{arm.name}-19"
    with pytest.raises(ValueError, match="tolerance_change"):
        read_trial(child, plan(), arm, 19)


@pytest.mark.parametrize("value", [0.0, -1e-9, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected_before_cuda(value: float) -> None:
    with pytest.raises(ValueError, match="tolerance_change"):
        replace(plan(), tolerance_change=value)


def test_coordinator_rejects_analytic_flag_without_actual_execution(
    tmp_path: Path, child_script: Path
) -> None:
    arm = ConvergenceArm(
        "neumann_analytic", "recursive", "parameters", "auto", "neumann_polynomial"
    )
    output = tmp_path / "analytic"
    run_isolated(plan(), [arm], [19], output, child_script)
    child = output / "processes" / f"{arm.name}-19"
    path = child / "report.json"
    report = json.loads(path.read_text())
    report["trials"][0]["feature_derivatives"] = "autograd"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="analytic feature derivatives"):
        read_trial(child, plan(), arm, 19)


def test_analytic_cli_preserves_two_explicit_executors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--output",
            str(tmp_path),
            "--case",
            "heat_neumann_homogeneous2d",
            "--constraints",
            "hard_initial_robin",
            "--hard-execution",
            "eager",
            "neumann_analytic",
            "--spatial-encoding",
            "neumann_polynomial",
        ],
    )
    _, _, arms = cli.arguments()
    assert {a.execution for a in arms} == {"eager", "neumann_analytic"}
    assert len({a.name for a in arms}) == 2


def test_density_budget_and_per_arm_points_reach_worker(tmp_path: Path) -> None:
    from benchmarks.convergence_processes import trial_command

    density = dict(
        training_seconds=60, snapshot_seconds=5, sampling="nested_uniform", prefix_points=8
    )
    selected = replace(plan(), points=None, density=density)
    arm = ConvergenceArm(
        "neumann_analytic", "recursive", "parameters", "auto", "neumann_polynomial", 32
    )
    command = trial_command(Path("benchmark.py"), selected, arm, 19, tmp_path)
    assert command[command.index("--points") + 1] == "32"
    assert json.loads(command[command.index("--density") + 1]) == density
    assert selected.config(19, arm)["training"]["collocation"]["points"] == 32


def test_pinned_coordinator_keeps_mandatory_working_directory(
    tmp_path: Path,
    child_script: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.convergence_processes as module

    original = module.subprocess.run
    observed: list[Path] = []

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        observed.append(kwargs["cwd"])
        return original(*args, **kwargs)

    monkeypatch.setenv("PINN_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(module.subprocess, "run", run)
    run_isolated(plan(), [ConvergenceArm()], [19], tmp_path / "root-cwd", child_script)
    assert observed == [tmp_path]
