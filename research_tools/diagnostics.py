"""Bounded, read-only diagnosis; never infer Windows ownership from a Mac PID."""

from __future__ import annotations

from pathlib import Path

from .schema import Json, inside

LOGS = {"run": "run.log", "launcher": "launcher.log"}


def validate_tail(log: str | None, lines: int, max_bytes: int) -> None:
    if log is not None and log not in LOGS:
        raise ValueError("Choose run or launcher log")
    if type(lines) is not int or not 1 <= lines <= 100:
        raise ValueError("Log lines must be between 1 and 100")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 16384:
        raise ValueError("Log byte budget must be between 1 and 16384")


def tail(directory: Path, log: str, lines: int, max_bytes: int) -> Json:
    validate_tail(log, lines, max_bytes)
    path = inside(directory, LOGS[log])
    with path.open("rb") as stream:
        stream.seek(0, 2)
        total = stream.tell()
        stream.seek(max(0, total - max_bytes))
        data = stream.read(max_bytes)
    decoded = data.decode("utf-8", errors="replace")
    selected = decoded.splitlines()[-lines:]
    return {
        "path": LOGS[log],
        "bytes_read": len(data),
        "file_bytes": total,
        "truncated": total > len(data) or len(decoded.splitlines()) > lines,
        "text": "\n".join(selected),
    }


def diagnose(
    directory: Path,
    *,
    live: bool = False,
    log: str | None = None,
    lines: int = 40,
    max_bytes: int = 4096,
) -> Json:
    validate_tail(log, lines, max_bytes)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    from .queue import job_status, read_metadata

    try:
        stored = read_metadata(inside(directory, "status.json"))
        if not isinstance(stored.get("state"), str):
            raise ValueError("Recorded state must be a string")
        state = (
            job_status(directory)
            if live
            else {
                "state": "unknown",
                "reason_code": "ownership_not_checked",
                "recorded_state": stored.get("state"),
            }
        )
    except (OSError, ValueError) as error:
        state = {
            "state": "unknown",
            "reason_code": "metadata_unreadable",
            "error": str(error)[:500],
        }
    output: Json = {
        "contract": 1,
        "kind": "research-diagnosis",
        "job_id": directory.name,
        "live_owner_check": live,
        "status": {
            k: (v[:500] if isinstance(v, str) else v)
            for k, v in state.items()
            if (
                isinstance(v, str)
                or v is None
                or type(v) is bool
                or (type(v) is int and abs(v) < 2**63)
            )
            and k
            in {
                "state",
                "recorded_state",
                "reason_code",
                "reason",
                "error",
                "commit",
                "started_at",
                "finished_at",
                "updated_at",
                "exit_code",
                "probe",
            }
        },
        "available_logs": [
            name for name, filename in LOGS.items() if (directory / filename).is_file()
        ],
        "cancel_requested": (directory / "cancel.request").exists(),
    }
    if log is not None:
        try:
            output["log"] = tail(directory, log, lines, max_bytes)
        except (OSError, ValueError) as error:
            output["log"] = {"error": str(error)[:500]}
    return output
