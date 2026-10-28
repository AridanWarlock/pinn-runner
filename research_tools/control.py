"""Small SSH requests; scientific evidence stays machine-parsed, not in stdout."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.gpu_jobs import remote
from scripts.gpu_worker import valid_id

from .comparison import compare
from .diagnostics import validate_tail
from .evidence import write_bundle
from .planning import plan
from .queue import validate_request
from .schema import Json, inside, object_value


def call(
    action: str,
    job_id: str,
    *,
    request: Json | None = None,
    log: str | None = None,
    lines: int = 40,
    max_bytes: int = 4096,
    bootstrap_commit: str | None = None,
) -> Json:
    valid_id(job_id)
    if action not in {"prepare", "launch", "status", "cancel", "collect", "diagnose"}:
        raise ValueError("Unknown research action")
    arguments = f"{action} --job-id {job_id}"
    if action == "diagnose":
        validate_tail(log, lines, max_bytes)
        arguments += f" --lines {lines} --max-bytes {max_bytes}"
        if log is not None:
            arguments += f" --log {log}"
    prefix = ""
    if request is not None:
        validate_request(request)
        encoded = base64.b64encode(json.dumps(request).encode()).decode()
        arguments += f" --request-base64 {encoded}"
        prefix = "git pull --ff-only | Out-Null\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\ngit fetch origin --quiet\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
    worker = "scripts/research_worker.py"
    if bootstrap_commit is not None:
        if not re.fullmatch("[0-9a-f]{40}", bootstrap_commit):
            raise ValueError("Bootstrap requires a full SHA")
        bootstrap = f".gpu-jobs/bootstrap-{bootstrap_commit}"
        prefix += (
            f"$refs = git branch -r --contains {bootstrap_commit}\n"
            "if ($LASTEXITCODE -ne 0 -or -not $refs) { throw 'Bootstrap revision is not published' }\n"
            f"git archive --format=zip {bootstrap_commit} --output '.gpu-jobs/bootstrap-{bootstrap_commit}.zip'\n"
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
            f"if (-not (Test-Path -LiteralPath '{bootstrap}')) {{\n"
            f" & '.\\.venv\\Scripts\\python.exe' -B -c \"from pathlib import Path; from scripts.gpu_worker import extract_snapshot; extract_snapshot(Path('.gpu-jobs/bootstrap-{bootstrap_commit}.zip'),Path('{bootstrap}'))\"\n"
            " if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n}\n"
        )
        verification = (
            "import zipfile; from pathlib import Path; "
            f"z=zipfile.ZipFile('.gpu-jobs/bootstrap-{bootstrap_commit}.zip'); "
            f"p=Path('{bootstrap}'); "
            "assert all((p/i.filename).read_bytes()==z.read(i) for i in z.infolist() if not i.is_dir())"
        )
        encoded_verification = base64.b64encode(verification.encode()).decode()
        prefix += (
            f"& '.\\.venv\\Scripts\\python.exe' -B -c \"import base64;exec(base64.b64decode('{encoded_verification}'))\"\n"
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
        )
        worker = bootstrap + "/scripts/research_worker.py"
    elif request is None:
        manifest_path = f".gpu-jobs/{job_id}/manifest.json"
        pinned_worker = f".gpu-jobs/{job_id}/checkout/scripts/research_worker.py"
        prefix += (
            "$worker = 'scripts/research_worker.py'\n"
            f"if (Test-Path -LiteralPath '{manifest_path}') {{\n"
            f" $manifest = Get-Content -Raw -LiteralPath '{manifest_path}' | ConvertFrom-Json\n"
            f" if ($manifest.contract -eq 9) {{ $worker = '{pinned_worker}' }}\n"
            "}\n"
        )
        worker = "$worker"
    value = remote(
        prefix
        + f"& '.\\.venv\\Scripts\\python.exe' {worker} {arguments}\nif ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}"
    )
    result = object_value(json.loads(value), "worker response")
    return (
        result
        if action in {"collect", "diagnose"}
        else {"contract": 1, "kind": "research-job", "job_id": job_id, **result}
    )


def submit(
    root: Path, protocol_commit: str, protocol_path: str, job_id: str, timeout: int, probe: bool
) -> Json:
    inside(root, protocol_path)
    # SHA validation happens before Git so branch names/options are never accepted.
    request = dict(
        job_id=job_id,
        protocol_commit=protocol_commit,
        protocol_path=protocol_path,
        protocol_sha256="0" * 64,
        timeout_seconds=timeout,
        probe=probe,
    )
    validate_request(request)
    source = subprocess.check_output(
        ["git", "show", f"{protocol_commit}:{protocol_path}"], cwd=root, stderr=subprocess.PIPE
    )
    manifest = object_value(json.loads(source), "protocol")
    plan(manifest)
    request["protocol_sha256"] = hashlib.sha256(source).hexdigest()
    prepared = call(
        "prepare",
        job_id,
        request=request,
        bootstrap_commit=protocol_commit if manifest["contract"] == 9 else None,
    )
    if prepared["state"] != "queued":
        return prepared
    return call("launch", job_id)


def collect_remote(job_id: str, output: Path) -> Json:
    if output.exists():
        raise FileExistsError(output)
    evidence = call("collect", job_id)
    comparison = compare(evidence)
    write_bundle(output, evidence)
    return {
        "contract": 1,
        "kind": "research-collection",
        "job_id": job_id,
        "trials": len(evidence["trials"]),
        "bundle": str(output.resolve()),
        "comparison_passed": comparison["comparison_passed"],
    }
