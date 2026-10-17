"""Pinned research jobs reusing the existing Windows GPU lock and process tree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

from scripts.gpu_worker import (
    ProcessTree,
    acquire_lock,
    extract_snapshot,
    job_environment,
    timestamp,
    valid_id,
    write_json,
)

from .schema import Json, inside, object_value, read_object, validate_manifest

TERMINAL = {"completed", "failed", "timed_out", "cancelled"}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def published(root: Path, revision: str) -> None:
    refs = git(
        root, "for-each-ref", "--format=%(refname)", "--contains", revision, "refs/remotes/origin"
    )
    if not refs.strip():
        raise ValueError("Revision is not reachable from fetched origin refs")


def validate_request(request: Json) -> None:
    if set(request) != {
        "job_id",
        "protocol_commit",
        "protocol_path",
        "protocol_sha256",
        "timeout_seconds",
        "probe",
    }:
        raise ValueError("Unknown research request fields")
    if len(valid_id(request["job_id"])) > 16:
        raise ValueError("Research job IDs are limited to 16 characters for Windows paths")
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", request["protocol_commit"]):
        raise ValueError("Full protocol commit SHA required")
    if not re.fullmatch(r"[0-9a-f]{64}", request["protocol_sha256"]):
        raise ValueError("Protocol byte hash required")
    inside(Path.cwd(), request["protocol_path"])
    if type(request["timeout_seconds"]) is not int or request["timeout_seconds"] < 1:
        raise ValueError("Positive timeout required")
    if type(request["probe"]) is not bool:
        raise ValueError("probe must be boolean")


def prepare(root: Path, request: Json) -> Json:
    validate_request(request)
    published(root, request["protocol_commit"])
    source = git(root, "show", f"{request['protocol_commit']}:{request['protocol_path']}")
    if hashlib.sha256(source).hexdigest() != request["protocol_sha256"]:
        raise ValueError("Published protocol differs from request")
    manifest = validate_manifest(object_value(json.loads(source), "manifest"), root)
    if manifest["contract"] != 2 or manifest["registration"] != "prospective":
        raise ValueError("Only prospective protocols may be submitted")
    published(root, manifest["revision"])
    directory = inside(root, ".gpu-jobs/" + request["job_id"])
    if directory.exists():
        if read_object(directory / "request.json") != request:
            raise ValueError("Job ID already belongs to a different request")
        return job_status(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", request)
    write_json(
        directory / "status.json",
        {
            "state": "queued",
            "job_id": request["job_id"],
            "commit": manifest["revision"],
            "updated_at": timestamp(),
            "probe": request["probe"],
        },
    )
    try:
        (directory / "manifest.json").write_bytes(source)
        archive = directory / "source.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", manifest["revision"], "--output", str(archive)],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        extract_snapshot(archive, directory / "checkout")
        if not (directory / "checkout/scripts/research_worker.py").is_file():
            raise ValueError("Pinned source does not support research jobs")
        verify_snapshot(directory)
    except Exception as error:
        write_json(
            directory / "status.json",
            {"state": "failed", "error": str(error), "updated_at": timestamp()},
        )
        raise
    return read_object(directory / "status.json")


def registered_manifest(directory: Path) -> Json:
    request = read_object(directory / "request.json")
    source = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(source).hexdigest() != request["protocol_sha256"]:
        raise ValueError("Registered protocol was modified")
    return validate_manifest(object_value(json.loads(source), "manifest"), directory)


def verify_snapshot(directory: Path) -> None:
    with zipfile.ZipFile(directory / "source.zip") as archive:
        expected: set[Path] = set()
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = inside(directory / "checkout", item.filename)
            expected.add(path)
            if path.is_symlink() or path.read_bytes() != archive.read(item):
                raise ValueError("Pinned source snapshot was modified")
        actual = {p.resolve() for p in (directory / "checkout").rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError("Unexpected files in pinned source snapshot")


def process_token(pid: int) -> str | None:
    """Windows creation FILETIME prevents accepting a reused process ID."""
    import ctypes
    from ctypes import wintypes

    api = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    api.GetProcessTimes.restype = wintypes.BOOL
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.GetExitCodeProcess.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        code = wintypes.DWORD()
        if (
            not api.GetProcessTimes(handle, *(ctypes.byref(t) for t in times))
            or not api.GetExitCodeProcess(handle, ctypes.byref(code))
            or code.value != 259
        ):
            return None
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
    finally:
        api.CloseHandle(handle)


def job_status(directory: Path) -> Json:
    status = read_object(directory / "status.json")
    if status["state"] in TERMINAL:
        return status
    owner = directory / "worker.claim"
    if owner.exists():
        claim = read_object(owner)
        if not claim["token"] or process_token(claim["pid"]) != claim["token"]:
            return status | {
                "state": "unknown",
                "reason": "Worker identity is no longer verifiable; no automatic restart",
            }
    elif (directory / "launch.claim").exists():
        launcher = directory / "launcher.json"
        launcher_info = read_object(launcher) if launcher.exists() else {}
        if (
            not launcher_info.get("token")
            or process_token(launcher_info["pid"]) != launcher_info["token"]
        ):
            return status | {
                "state": "unknown",
                "reason": "Launch outcome is uncertain; no automatic restart",
            }
    return status


def launch(root: Path, directory: Path) -> Json:
    if (
        read_object(directory / "status.json")["state"] in TERMINAL
        or (directory / "worker.slot").exists()
    ):
        return job_status(directory)
    try:
        with (directory / "launch.claim").open("x", encoding="utf-8") as stream:
            stream.write(timestamp())
    except FileExistsError:
        return job_status(directory)
    manifest = registered_manifest(directory)
    command = [
        str(root / ".venv/Scripts/python.exe"),
        "-B",
        str(directory / "checkout/scripts/research_worker.py"),
        "execute",
        "--job-id",
        directory.name,
    ]
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB.
    # Fail explicitly if the SSH host's Job Object does not permit breakaway.
    with (directory / "launcher.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=job_environment(directory, manifest["revision"]),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=0x01000208,
            close_fds=True,
        )
    pid = process.pid
    write_json(directory / "launcher.json", {"pid": pid, "token": process_token(pid)})
    return job_status(directory) | {"launcher_pid": pid}


def cancel(directory: Path) -> Json:
    state = job_status(directory)
    if state["state"] not in TERMINAL:
        (directory / "cancel.request").touch(exist_ok=True)
    return state | {"cancel_requested": (directory / "cancel.request").exists()}


def execute(root: Path, directory: Path) -> None:
    request = read_object(directory / "request.json")
    validate_request(request)
    try:
        with (directory / "worker.slot").open("x", encoding="utf-8"):
            pass
        write_json(
            directory / "worker.claim", {"pid": os.getpid(), "token": process_token(os.getpid())}
        )
    except FileExistsError:
        return
    state = read_object(directory / "status.json")
    lock = None
    tree = None
    process = None
    try:

        def cancelled() -> bool:
            return (directory / "cancel.request").exists()

        lock = acquire_lock(root / ".gpu-jobs/gpu.lock", directory, cancelled)
        verify_snapshot(directory)
        state.update(state="running", started_at=timestamp(), updated_at=timestamp())
        write_json(directory / "status.json", state)
        manifest = registered_manifest(directory)
        script = directory / "checkout/scripts/research_run.py"
        command = [
            str(root / ".venv/Scripts/python.exe"),
            str(script),
            "--manifest",
            str(directory / "manifest.json"),
            "--output",
            str(directory / "r"),
        ]
        if request["probe"]:
            command = [str(root / ".venv/Scripts/python.exe"), "-c", "import time;time.sleep(3)"]
        gate = directory / "child.ready"
        # No script can spawn children before assignment to the Windows Job Object.
        bootstrap = "import pathlib,sys,time,runpy; p=pathlib.Path(sys.argv[1]);\nwhile not p.exists(): time.sleep(.05)\nsys.argv=sys.argv[2:];\nif sys.argv[0]=='-c': exec(sys.argv[1])\nelse: runpy.run_path(sys.argv[0],run_name='__main__')"
        command = [command[0], "-c", bootstrap, str(gate), *command[1:]]
        with (directory / "run.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=job_environment(directory, manifest["revision"]),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            tree = ProcessTree(process)
            gate.touch(exist_ok=False)
            started = time.monotonic()
            while process.poll() is None:
                if cancelled() or time.monotonic() - started > request["timeout_seconds"]:
                    tree.terminate()
                    process.wait(timeout=30)
                    state["state"] = "cancelled" if cancelled() else "timed_out"
                    break
                progress = (
                    read_object(directory / "progress.json")
                    if (directory / "progress.json").exists()
                    else {}
                )
                write_json(
                    directory / "status.json",
                    state | {"updated_at": timestamp(), "progress": progress},
                )
                time.sleep(0.2)
            else:
                state.update(
                    state="completed" if process.returncode == 0 else "failed",
                    exit_code=process.returncode,
                )
    except InterruptedError:
        state["state"] = "cancelled"
    except Exception as error:
        if tree is not None:
            tree.terminate()
        elif process is not None:
            process.kill()
        if process is not None:
            process.wait(timeout=30)
        state.update(state="failed", error=str(error))
    finally:
        progress = (
            read_object(directory / "progress.json")
            if (directory / "progress.json").exists()
            else {}
        )
        write_json(
            directory / "status.json",
            state | {"updated_at": timestamp(), "finished_at": timestamp(), "progress": progress},
        )
        if tree is not None:
            tree.close()
        if lock is not None:
            lock.close()
