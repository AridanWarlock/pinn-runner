"""Mac SSH control: start/status/collect pinned Windows CUDA jobs."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ROOT = r"C:\Projects\pinn-runner"


def remote(script: str) -> str:
    temporary = WINDOWS_ROOT + r"\.tmp"
    prefix = f"""Set-Location -LiteralPath '{WINDOWS_ROOT}' -ErrorAction Stop
if ((Get-Location).Path -ne '{WINDOWS_ROOT}') {{ exit 1 }}
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = '{temporary}'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
"""
    encoded = base64.b64encode((prefix + script).encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "aridanPC",
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def json_output(output: str) -> dict[str, Any]:
    value, _ = json.JSONDecoder().raw_decode(output[output.index("{") :])
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def start(request: dict[str, Any]) -> dict[str, Any]:
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    prepared = json_output(
        remote(
            f"& '.\\.venv\\Scripts\\python.exe' scripts/gpu_worker.py prepare --request-base64 {encoded}\nif ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}"
        )
    )
    identifier = prepared["job_id"]
    worker_script = f"""Set-Location -LiteralPath '{WINDOWS_ROOT}' -ErrorAction Stop
if ((Get-Location).Path -ne '{WINDOWS_ROOT}') {{ exit 1 }}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = '{WINDOWS_ROOT}\\.tmp'
$env:TMP = $env:TEMP
& '{WINDOWS_ROOT}\\.venv\\Scripts\\python.exe' '{WINDOWS_ROOT}\\.gpu-jobs\\{identifier}\\checkout\\scripts\\gpu_worker.py' execute --job-id {identifier}
exit $LASTEXITCODE
"""
    worker_encoded = base64.b64encode(worker_script.encode("utf-16le")).decode("ascii")
    script = f"""$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly
$startup.ShowWindow = 0
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='powershell.exe -NoProfile -NonInteractive -EncodedCommand {worker_encoded}'; CurrentDirectory='{WINDOWS_ROOT}'; ProcessStartupInformation=$startup}}
if ($result.ReturnValue -ne 0) {{ throw "Worker launch failed: $($result.ReturnValue)" }}
@{{job_id='{identifier}'; launcher_pid=$result.ProcessId; state='queued'}} | ConvertTo-Json -Compress
"""
    return json_output(remote(script))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "collect"))
    parser.add_argument("--job-id")
    parser.add_argument("--commit")
    parser.add_argument(
        "--case", choices=("heat2d", "heat3d", "advection", "burgers"), default="heat3d"
    )
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 4321])
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, default=Path("numerical-reports/collected"))
    args = parser.parse_args()
    from scripts.gpu_worker import valid_id, validate_request

    if args.action == "start":
        commit = (
            args.commit
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
        request = {
            "job_id": args.job_id or uuid.uuid4().hex[:12],
            "commit": commit,
            "case": args.case,
            "steps": args.steps,
            "seeds": args.seeds,
            "timeout_seconds": args.timeout_seconds,
        }
        validate_request(request)
        print(json.dumps(start(request), indent=2))
    else:
        identifier = valid_id(args.job_id or "")
        status = json_output(
            remote(
                f"& '.\\.venv\\Scripts\\python.exe' scripts/gpu_worker.py status --job-id {identifier}\nif ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}"
            )
        )
        if args.action == "collect":
            if status["state"] not in ("passed", "failed", "timed_out"):
                parser.error("Job is still active; nothing collected")
            directory = args.output_dir.resolve() / identifier
            directory.mkdir(parents=True, exist_ok=False)
            for name, relative in (
                ("status.json", "status.json"),
                ("request.json", "request.json"),
                ("report.json", "results/report.json"),
                ("run.log", "run.log"),
            ):
                path = f"{WINDOWS_ROOT}\\.gpu-jobs\\{identifier}\\{relative}"
                output = remote(
                    f"if (Test-Path -LiteralPath '{path}') {{ [Convert]::ToBase64String([IO.File]::ReadAllBytes('{path}')) }}"
                )
                if output.strip():
                    (directory / name).write_bytes(base64.b64decode(output.strip()))
            status["collected_to"] = str(directory)
        print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
