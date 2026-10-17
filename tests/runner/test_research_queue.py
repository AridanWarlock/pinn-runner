"""Queue lifecycle with fake Git/Windows processes; no network and no CUDA."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from research_tools import queue


def protocol() -> dict[str, Any]:
    baseline = dict(
        backend="eager",
        lifting="recursive",
        backward="parameters",
        background="auto",
        spatial_encoding="neumann_polynomial",
    )
    candidate = baseline | {"backend": "neumann_analytic"}
    return dict(
        contract=2,
        registration="prospective",
        revision="a" * 40,
        reports=["c0/report.json"],
        expected_trials=[dict(case="heat", seed=19, **arm) for arm in (baseline, candidate)],
        report_parameters=dict(
            adam_steps=1,
            lbfgs_snapshots=[0, 1],
            constraints="hard_initial_robin",
            points_override=4,
            lbfgs_tolerance_change=1e-12,
            execution_lifecycle="isolated_trials",
        ),
        case_limits={"heat": dict(relative_l2=0.01, final_relative_l2=0.02, pde_rmse=0.03)},
        physical_limits=dict(initial_rmse=1e-6, boundary_max=1e-5),
        comparison=dict(
            baseline=baseline,
            candidate=candidate,
            quality_factor=1.5,
            median_training_ratio_max=0.95,
            worst_training_ratio_max=1.1,
        ),
        case_faces={
            "heat": [dict(field="u", kind="neumann", axis="x", side=s) for s in ("lower", "upper")]
        },
        network=dict(hidden_layers=[32, 32, 32], spatial_encoding="neumann_polynomial"),
        environment_requirements=dict(
            torch="2.1.0+cu118",
            cuda_runtime="11.8",
            default_dtype="torch.float32",
            float32_matmul_precision="highest",
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
        ),
    )


@pytest.fixture
def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    source = json.dumps(protocol()).encode()
    request = dict(
        job_id="probe",
        protocol_commit="b" * 40,
        protocol_path="docs/protocol.json",
        protocol_sha256=hashlib.sha256(source).hexdigest(),
        timeout_seconds=1,
        probe=True,
    )

    def git(root: Path, *args: str) -> bytes:
        return source if args[0] == "show" else b"refs/remotes/origin/topic\n"

    def archive(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        with zipfile.ZipFile(command[-1], "w") as z:
            z.writestr("scripts/research_worker.py", "# pinned worker\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(queue, "git", git)
    monkeypatch.setattr(queue.subprocess, "run", archive)
    queue.prepare(tmp_path, request)
    return tmp_path, request


def test_prepare_is_idempotent_but_conflicting_id_is_rejected(
    job: tuple[Path, dict[str, Any]],
) -> None:
    root, request = job
    assert queue.prepare(root, request)["state"] == "queued"
    with pytest.raises(ValueError, match="different request"):
        queue.prepare(root, request | {"timeout_seconds": 5})
    with pytest.raises(ValueError, match="differs"):
        queue.prepare(root, request | {"protocol_sha256": "d" * 64})


def test_only_published_revisions_and_exact_snapshots_are_allowed(
    job: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, request = job
    directory = root / ".gpu-jobs/probe"
    (directory / "checkout/scripts/research_worker.py").write_text("# changed")
    with pytest.raises(ValueError, match="modified"):
        queue.verify_snapshot(directory)
    monkeypatch.setattr(queue, "git", lambda *args: b"")
    with pytest.raises(ValueError, match="reachable"):
        queue.published(root, request["protocol_commit"])


def test_protocol_cannot_be_changed_after_registration(job: tuple[Path, dict[str, Any]]) -> None:
    directory = job[0] / ".gpu-jobs/probe"
    (directory / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        queue.registered_manifest(directory)


def test_uncertain_launch_never_repeats_process_creation(
    job: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = job
    calls = []

    def broken(*args: Any, **kwargs: Any) -> str:
        calls.append(args)
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(queue.subprocess, "Popen", broken)
    with pytest.raises(subprocess.CalledProcessError):
        queue.launch(root, root / ".gpu-jobs/probe")
    assert queue.launch(root, root / ".gpu-jobs/probe")["state"] == "unknown"
    assert len(calls) == 1


def test_reused_worker_pid_is_unknown(
    job: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = job[0] / ".gpu-jobs/probe"
    queue.write_json(directory / "worker.claim", dict(pid=123, token="old"))
    monkeypatch.setattr(queue, "process_token", lambda pid: "new")
    assert queue.job_status(directory)["state"] == "unknown"


@pytest.mark.parametrize("outcome", ["completed", "timed_out", "cancelled", "attach_failure"])
def test_worker_owns_gate_tree_and_final_status(
    job: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    root, _ = job
    directory = root / ".gpu-jobs/probe"
    events = []

    class Process:
        returncode: int | None = 0 if outcome == "completed" else None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs["cwd"] == root
            assert not (directory / "child.ready").exists()
            events.append("created")

        def poll(self) -> int | None:
            if outcome == "cancelled":
                (directory / "cancel.request").touch()
            return self.returncode

        def wait(self, timeout: int) -> int:
            events.append("waited")
            return 1

        def kill(self) -> None:
            events.append("killed")

    class Tree:
        def __init__(self, process: Process) -> None:
            assert not (directory / "child.ready").exists()
            if outcome == "attach_failure":
                raise OSError("attach failed")
            events.append("attached")
            self.process = process

        def terminate(self) -> None:
            events.append("terminated")
            self.process.returncode = 1

        def close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(queue, "process_token", lambda pid: "same")
    monkeypatch.setattr(queue, "acquire_lock", lambda *args: io.BytesIO())
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setattr(queue, "ProcessTree", Tree)
    monkeypatch.setattr(queue.time, "monotonic", iter([0.0, 2.0]).__next__)
    queue.execute(root, directory)
    state = queue.read_object(directory / "status.json")["state"]
    assert state == ("failed" if outcome == "attach_failure" else outcome)
    if outcome == "attach_failure":
        assert "killed" in events
    else:
        assert events[:2] == ["created", "attached"]
        assert (directory / "child.ready").exists()
        assert "closed" in events
    if outcome in ("timed_out", "cancelled"):
        assert "terminated" in events
    # Duplicate execution cannot run or clear a completed claim.
    before = list(events)
    queue.execute(root, directory)
    assert events == before


def test_cancel_while_queued_does_not_start_process(
    job: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = job
    directory = root / ".gpu-jobs/probe"
    queue.cancel(directory)
    monkeypatch.setattr(queue, "process_token", lambda pid: "same")

    def lock(path: Path, where: Path, cancelled: Any) -> Any:
        assert path == root / ".gpu-jobs/gpu.lock" and cancelled()
        raise InterruptedError

    monkeypatch.setattr(queue, "acquire_lock", lock)
    queue.execute(root, directory)
    assert queue.job_status(directory)["state"] == "cancelled"


def test_remote_transport_validates_ids_and_propagates_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_tools import control

    def disconnected(script: str) -> str:
        raise RuntimeError("SSH disconnected")

    monkeypatch.setattr(control, "remote", disconnected)
    with pytest.raises(RuntimeError, match="disconnected"):
        control.call("status", "probe")
    with pytest.raises(ValueError):
        control.call("status", "../outside")


def test_launch_is_detached_and_directory_guarded(
    job: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = job
    commands = []

    class Process:
        pid = 123

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            commands.append((command, kwargs))

    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setattr(queue, "process_token", lambda pid: "same")
    state = queue.launch(root, root / ".gpu-jobs/probe")
    assert state["launcher_pid"] == 123
    command, options = commands[0]
    assert command[1] == "-B"
    assert options["cwd"] == root and options["creationflags"] == 0x01000208
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    queue.launch(root, root / ".gpu-jobs/probe")
    assert len(commands) == 1
