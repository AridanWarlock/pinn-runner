"""Windows-only, detached CUDA worker with a pinned source snapshot and queue lock."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO, Callable, Protocol, cast

ROOT = Path(r"C:\Projects\pinn-runner")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
        raise ValueError("Invalid job ID")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for _ in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    raise RuntimeError("Cannot atomically update job status")


def extract_snapshot(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name = item.filename.replace("\\", "/")
            if ".." in name.split("/") or re.match(r"^[A-Za-z]:", name):
                raise ValueError("Unsafe snapshot entry")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(
                item.external_attr >> 16
            ):
                raise ValueError("Unsafe snapshot entry")
        source.extractall(destination)


def validate_request(request: dict[str, Any]) -> None:
    valid_id(request["job_id"])
    if not re.fullmatch(r"[0-9a-f]{40}", request["commit"]):
        raise ValueError("A full commit SHA is required")
    if request["case"] not in ("heat2d", "heat3d", "advection", "burgers"):
        raise ValueError("Unknown benchmark case")
    for key in ("steps", "timeout_seconds"):
        if type(request[key]) is not int or request[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not request["seeds"] or any(type(seed) is not int or seed < 0 for seed in request["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(request["seeds"])) != len(request["seeds"]):
        raise ValueError("Seeds must be unique")


def prepare(request: dict[str, Any]) -> dict[str, Any]:
    validate_request(request)
    directory = ROOT / ".gpu-jobs" / request["job_id"]
    directory.mkdir(parents=True, exist_ok=False)
    request = {**request, "created_at": timestamp()}
    write_json(directory / "request.json", request)
    try:
        archive = directory / "source.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", request["commit"], "--output", str(archive)],
            cwd=ROOT,
            check=True,
        )
        extract_snapshot(archive, directory / "checkout")
        if not (directory / "checkout/scripts/gpu_worker.py").is_file():
            raise ValueError("Pinned revision does not support GPU jobs")
        status = {
            "job_id": request["job_id"],
            "state": "queued",
            "commit": request["commit"],
            "updated_at": timestamp(),
        }
    except Exception as error:
        status = {
            "job_id": request["job_id"],
            "state": "failed",
            "error": str(error),
            "updated_at": timestamp(),
        }
        write_json(directory / "status.json", status)
        raise
    write_json(directory / "status.json", status)
    return status


class FileLockBackend(Protocol):
    LK_NBLCK: int

    def locking(self, descriptor: int, mode: int, count: int) -> None: ...


def acquire_lock(
    path: Path, directory: Path, cancelled: Callable[[], bool] | None = None
) -> BinaryIO:
    msvcrt = cast(FileLockBackend, import_module("msvcrt"))

    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    while True:
        if cancelled is not None and cancelled():
            stream.close()
            raise InterruptedError("Cancelled while queued")
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return stream
        except OSError:
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            write_json(
                directory / "status.json",
                {**status, "updated_at": timestamp(), "worker_pid": os.getpid()},
            )
            time.sleep(3)


def job_environment(directory: Path, commit: str) -> dict[str, str]:
    environment = dict(os.environ)
    for name, relative in {
        "TEMP": ".tmp",
        "TMP": ".tmp",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "TORCH_HOME": ".cache/torch",
        "CUDA_CACHE_PATH": ".cache/cuda",
        "PIP_CACHE_DIR": ".cache/pip",
    }.items():
        path = directory / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment.update(
        PINN_WORKING_DIRECTORY=str(ROOT),
        PINN_PINNED_REVISION=commit,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
    )
    return environment


class ProcessTree:
    """Own a Windows Job Object so child processes can be stopped together."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        import ctypes
        from ctypes import wintypes

        self.api = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("Windows Job Object operation failed")
        if not self.api.AssignProcessToJobObject(self.handle, getattr(process, "_handle")):
            self.close()
            raise OSError("Windows Job Object operation failed")

    def terminate(self) -> None:
        if not self.api.TerminateJobObject(self.handle, 1):
            raise OSError("Windows Job Object operation failed")

    def close(self) -> None:
        self.api.CloseHandle(self.handle)


def execute(directory: Path) -> None:
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    validate_request(request)
    with (directory / "worker.claim").open("x", encoding="utf-8") as claim:
        claim.write(str(os.getpid()))
    queued = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    write_json(
        directory / "status.json", {**queued, "worker_pid": os.getpid(), "updated_at": timestamp()}
    )
    lock = acquire_lock(ROOT / ".gpu-jobs/gpu.lock", directory)
    status = {
        "job_id": request["job_id"],
        "commit": request["commit"],
        "state": "running",
        "started_at": timestamp(),
        "worker_pid": os.getpid(),
    }
    write_json(directory / "status.json", {**status, "updated_at": timestamp()})
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        str(directory / "checkout/scripts/benchmark_solutions.py"),
        "--profile",
        "reference",
        "--cases",
        request["case"],
        "--steps",
        str(request["steps"]),
        "--seeds",
        *map(str, request["seeds"]),
        "--output-dir",
        str(directory / "results"),
    ]
    tree: ProcessTree | None = None
    try:
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=job_environment(directory, request["commit"]),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            tree = ProcessTree(process)
            status["child_pid"] = process.pid
            started = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - started > request["timeout_seconds"]:
                    tree.terminate()
                    process.wait(timeout=30)
                    status["state"] = "timed_out"
                    break
                write_json(directory / "status.json", {**status, "updated_at": timestamp()})
                time.sleep(3)
            if status["state"] != "timed_out":
                status.update(
                    state="passed" if process.returncode == 0 else "failed",
                    exit_code=process.returncode,
                )
    except Exception as error:
        if tree is not None:
            tree.terminate()
        status.update(state="failed", error=str(error))
    finally:
        status.update(finished_at=timestamp(), updated_at=timestamp())
        write_json(directory / "status.json", status)
        if tree is not None:
            tree.close()
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "status"))
    parser.add_argument("--job-id")
    parser.add_argument("--request-base64")
    args = parser.parse_args()
    if sys.platform != "win32" or Path.cwd().resolve() != ROOT.resolve():
        parser.error("Windows worker must run from C:\\Projects\\pinn-runner")
    if args.action == "prepare":
        print(json.dumps(prepare(json.loads(base64.b64decode(args.request_base64)))))
    else:
        directory = ROOT / ".gpu-jobs" / valid_id(args.job_id)
        if args.action == "execute":
            execute(directory)
        else:
            print((directory / "status.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
