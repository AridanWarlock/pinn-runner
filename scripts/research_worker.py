"""Guarded Windows control plane for pinned research protocols."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_tools import queue  # noqa: E402
from research_tools.diagnostics import diagnose  # noqa: E402
from research_tools.evidence import collect  # noqa: E402
from research_tools.schema import inside  # noqa: E402
from scripts.gpu_worker import ROOT, valid_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "launch", "execute", "status", "cancel", "collect", "diagnose"),
    )
    parser.add_argument("--job-id")
    parser.add_argument("--request-base64")
    parser.add_argument("--log", choices=("run", "launcher"))
    parser.add_argument("--lines", type=int, default=40)
    parser.add_argument("--max-bytes", type=int, default=4096)
    args = parser.parse_args()
    if sys.platform != "win32" or Path.cwd().resolve() != ROOT.resolve():
        parser.error("Windows project working directory required")
    if args.action == "prepare":
        output = queue.prepare(ROOT, json.loads(base64.b64decode(args.request_base64)))
    else:
        directory = inside(ROOT, ".gpu-jobs/" + valid_id(args.job_id or ""))
        if args.action == "execute":
            queue.execute(ROOT, directory)
            return
        if args.action == "diagnose":
            output = diagnose(
                directory, live=True, log=args.log, lines=args.lines, max_bytes=args.max_bytes
            )
        elif args.action == "launch":
            output = queue.launch(ROOT, directory)
        elif args.action == "cancel":
            output = queue.cancel(directory)
        elif args.action == "collect":
            if queue.job_status(directory)["state"] != "completed":
                raise ValueError("A completed job is required for collection")
            if queue.read_object(directory / "request.json")["probe"]:
                raise ValueError("Transport probes do not produce scientific evidence")
            path = directory / "manifest.json"
            manifest = queue.registered_manifest(directory)
            if manifest["contract"] == 3:
                from research_tools.frozen_profile import collect as collect_profile

                output = collect_profile(directory / "r", manifest)
            else:
                output = collect(directory / "r", manifest, path)
            output["provenance"] = {
                "protocol_commit": queue.read_object(directory / "request.json")[
                    "protocol_commit"
                ],
                "protocol_sha256": queue.read_object(directory / "request.json")[
                    "protocol_sha256"
                ],
            }
        else:
            output = queue.job_status(directory)
    print(json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
