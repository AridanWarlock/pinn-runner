"""Fresh child processes and partial-report validation without neural training."""

import json
import subprocess
from argparse import Namespace
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
p.add_argument('--points',type=int)
a,_=p.parse_known_args()
a.output.mkdir(parents=True)
trial=dict(case=a.case,seed=a.seeds,backend=a.hard_execution,lifting=a.hard_lifting,backward=a.backward,pid=os.getpid())
r=dict(contract=3,status='completed',execution_lifecycle='single_trial',case=a.case,environment=dict(revision='fixture',device='cpu'),limits=dict(relative_l2=a.relative_l2,final_relative_l2=a.final_relative_l2,pde_rmse=a.pde_rmse),adam_steps=a.steps,lbfgs_snapshots=a.snapshots,constraints=a.constraints,points_override=a.points,trials=[trial])
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
