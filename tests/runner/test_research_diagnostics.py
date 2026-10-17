"""Bounded diagnostics and unambiguous versus uncertain process creation failures."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from research_tools import diagnostics, queue


def test_tail_reads_only_the_budget_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.log"
    path.write_bytes(b"placeholder")
    data = b"A" * 2_000_000 + b"\xff\nlast line\n"

    class Bounded(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert 0 <= size <= 64
            return super().read(size)

    original = Path.open

    def opened(self: Path, *args: Any, **kwargs: Any) -> Any:
        return Bounded(data) if self == path else original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    value = diagnostics.tail(tmp_path, "run", 2, 64)
    assert value["bytes_read"] == 64 and value["file_bytes"] == len(data)
    assert value["truncated"] and "last line" in value["text"]
    assert len(value["text"].splitlines()) == 2


@pytest.mark.parametrize(
    "args",
    [
        ("../outside", 40, 4096),
        ("run", 0, 4096),
        ("run", 101, 4096),
        ("run", 1, -1),
        ("run", 1, 16385),
        ("run", True, 64),
    ],
)
def test_invalid_log_options_are_rejected(args: tuple[Any, Any, Any]) -> None:
    with pytest.raises(ValueError):
        diagnostics.validate_tail(*args)


def test_local_diagnosis_never_checks_a_mac_pid_as_windows_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue.write_json(tmp_path / "status.json", {"state": "running"})
    queue.write_json(tmp_path / "worker.claim", {"pid": 123, "token": "old"})

    def forbidden(pid: int) -> str:
        raise AssertionError("Do not check Windows PIDs on this host")

    monkeypatch.setattr(queue, "process_token", forbidden)
    value = diagnostics.diagnose(tmp_path)
    assert value["status"]["state"] == "unknown"
    assert value["status"]["recorded_state"] == "running"
    assert "log" not in value


def test_corrupt_metadata_and_missing_log_are_diagnosable(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text("{")
    value = diagnostics.diagnose(tmp_path, live=True, log="run")
    assert value["status"]["reason_code"] == "metadata_unreadable"
    assert "error" in value["log"]
    queue.write_json(tmp_path / "status.json", {"state": "running"})
    (tmp_path / "worker.claim").write_text("{")
    assert queue.job_status(tmp_path)["state"] == "unknown"


def test_log_symlink_cannot_read_outside_job(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_text("unrelated data")
    (root / "run.log").symlink_to(outside)
    value = diagnostics.diagnose(root, log="run")
    assert "escapes" in value["log"]["error"]


def test_definitive_creation_failure_is_persisted_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue.write_json(tmp_path / "status.json", {"state": "queued", "commit": "a" * 40})
    monkeypatch.setattr(queue, "registered_manifest", lambda d: {"revision": "a" * 40})
    monkeypatch.setattr(queue, "job_environment", lambda *a: {})
    calls: list[int] = []

    def refused(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        raise PermissionError("Process creation denied")

    monkeypatch.setattr(queue.subprocess, "Popen", refused)
    result = queue.launch(tmp_path, tmp_path)
    assert result["state"] == "failed" and result["reason_code"] == "launch_rejected"
    assert queue.launch(tmp_path, tmp_path)["state"] == "failed"
    assert len(calls) == 1


def test_metadata_failure_after_creation_remains_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue.write_json(tmp_path / "status.json", {"state": "queued"})
    monkeypatch.setattr(queue, "registered_manifest", lambda d: {"revision": "a" * 40})
    monkeypatch.setattr(queue, "job_environment", lambda *a: {})
    monkeypatch.setattr(queue, "process_token", lambda pid: "same")
    calls: list[int] = []

    class Process:
        pid = 123

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(1)

    monkeypatch.setattr(queue.subprocess, "Popen", Process)

    def failed_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("Disk unavailable")

    monkeypatch.setattr(queue, "write_json", failed_write)
    with pytest.raises(OSError):
        queue.launch(tmp_path, tmp_path)
    assert queue.launch(tmp_path, tmp_path)["state"] == "unknown"
    assert len(calls) == 1


def test_remote_diagnostics_validates_options_before_ssh(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_tools import control

    scripts: list[str] = []

    def remote(script: str) -> str:
        scripts.append(script)
        return json.dumps(
            {"contract": 1, "kind": "research-diagnosis", "status": {"state": "completed"}}
        )

    monkeypatch.setattr(control, "remote", remote)
    assert (
        control.call("diagnose", "probe", log="run", lines=3, max_bytes=128)["kind"]
        == "research-diagnosis"
    )
    assert "--lines 3 --max-bytes 128 --log run" in scripts[0]
    with pytest.raises(ValueError):
        control.call("diagnose", "probe", log="run;exit")
    assert len(scripts) == 1


def test_diagnostic_metadata_output_is_bounded(tmp_path: Path) -> None:
    queue.write_json(
        tmp_path / "status.json",
        {"state": "failed", "error": "x" * 10_000, "exit_code": {"nested": "y" * 10_000}},
    )
    result = diagnostics.diagnose(tmp_path, live=True)
    assert len(result["status"]["error"]) == 500
    assert "exit_code" not in result["status"]
    (tmp_path / "status.json").write_bytes(b"x" * 70_000)
    result = diagnostics.diagnose(tmp_path, live=True)
    assert result["status"]["reason_code"] == "metadata_unreadable"
    assert "64 KiB" in result["status"]["error"]
