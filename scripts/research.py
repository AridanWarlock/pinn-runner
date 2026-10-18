"""Compact research control plane; Torch runs only inside Windows workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_tools.board import brief, task_page  # noqa: E402
from research_tools.comparison import compare  # noqa: E402
from research_tools.control import call, collect_remote, submit  # noqa: E402
from research_tools.diagnostics import diagnose  # noqa: E402
from research_tools.evidence import collect, read_bundle, status, write_bundle  # noqa: E402
from research_tools.planning import plan  # noqa: E402
from research_tools.remote_status import remote_status  # noqa: E402
from research_tools.schema import Json, load_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("status", help="Read only the small study status.json")
    state_input = state.add_mutually_exclusive_group(required=True)
    state_input.add_argument("--root")
    state_input.add_argument("--job-id")
    state.add_argument(
        "--remote", action="store_true", help="Read Windows status through guarded SSH"
    )
    gather = sub.add_parser("collect", help="Validate a complete study and write compact evidence")
    gather.add_argument("--root", type=Path)
    gather.add_argument("--manifest", type=Path)
    gather.add_argument("--remote", action="store_true")
    gather.add_argument("--job-id")
    gather.add_argument("--output", type=Path, required=True)
    comparison = sub.add_parser("compare", help="Evaluate explicit paired gates")
    comparison.add_argument("--bundle", type=Path, required=True)
    comparison.add_argument("--details", action="store_true")
    task = sub.add_parser("task", help="Return task goals without implementation history")
    task.add_argument("task_id")
    task.add_argument("--criteria", action="store_true", help="Include full acceptance criteria")
    task.add_argument("--board", type=Path, default=ROOT / ".project-board")
    tasks = sub.add_parser("tasks", help="Read a bounded page of current tasks and requirements")
    tasks.add_argument("--status", help="Exact board status; default excludes Done and Deferred")
    tasks.add_argument("--limit", type=int, default=5)
    tasks.add_argument("--skip", type=int, default=0)
    tasks.add_argument("--board", type=Path, default=ROOT / ".project-board")
    launch = sub.add_parser("submit", help="Submit a published protocol to the Windows GPU queue")
    launch.add_argument("--protocol-commit", required=True)
    launch.add_argument("--protocol-path", required=True)
    launch.add_argument("--job-id", required=True)
    launch.add_argument("--timeout-seconds", type=int, default=3600)
    launch.add_argument(
        "--probe", action="store_true", help="Test lifecycle without importing Torch or training"
    )
    cancel = sub.add_parser(
        "cancel", help="Request cooperative termination of the owned process tree"
    )
    cancel.add_argument("--job-id", required=True)
    diagnosis = sub.add_parser("diagnose", help="Bounded job metadata and optional log tail")
    diagnosis_input = diagnosis.add_mutually_exclusive_group(required=True)
    diagnosis_input.add_argument("--root", type=Path)
    diagnosis_input.add_argument("--job-id")
    diagnosis.add_argument("--remote", action="store_true")
    diagnosis.add_argument("--log", choices=("run", "launcher"))
    diagnosis.add_argument("--lines", type=int, default=40)
    diagnosis.add_argument("--max-bytes", type=int, default=4096)
    planning = sub.add_parser("plan", help="Offline protocol validation and budget summary")
    planning.add_argument("--manifest", type=Path, required=True)
    planning.add_argument("--details", action="store_true", help="Include all seeds")
    args = parser.parse_args(argv)
    code = 0
    try:
        output: Json
        if args.command == "tasks":
            output = task_page(args.board, status=args.status, limit=args.limit, skip=args.skip)
        elif args.command == "plan":
            output = plan(load_manifest(args.manifest), details=args.details)
        elif args.command == "diagnose":
            if args.job_id:
                if not args.remote:
                    parser.error("--job-id requires --remote")
                output = call(
                    "diagnose",
                    args.job_id,
                    log=args.log,
                    lines=args.lines,
                    max_bytes=args.max_bytes,
                )
            else:
                if args.remote:
                    parser.error("Use --job-id for remote diagnostics")
                output = diagnose(
                    args.root, log=args.log, lines=args.lines, max_bytes=args.max_bytes
                )
        elif args.command == "status":
            if args.job_id:
                if not args.remote:
                    parser.error("--job-id requires --remote")
                output = call("status", args.job_id)
            else:
                output = remote_status(args.root) if args.remote else status(Path(args.root))
        elif args.command == "submit":
            output = submit(
                ROOT,
                args.protocol_commit,
                args.protocol_path,
                args.job_id,
                args.timeout_seconds,
                args.probe,
            )
        elif args.command == "cancel":
            output = call("cancel", args.job_id)
        elif args.command == "collect" and args.remote:
            if not args.job_id or args.root or args.manifest:
                parser.error("Remote collection requires only --job-id and --output")
            output = collect_remote(args.job_id, args.output)
        elif args.command == "collect":
            if not args.root or not args.manifest or args.job_id:
                parser.error("Local collection requires --root and --manifest")
            manifest = load_manifest(args.manifest)
            evidence = collect(args.root, manifest, args.manifest)
            # Validate pairing before writing any output.
            compare(evidence)
            write_bundle(args.output, evidence)
            output = {
                "contract": 1,
                "kind": "research-collection",
                "trials": len(evidence["trials"]),
                "bundle": str(args.output.resolve()),
                "registration": manifest["registration"],
            }
        elif args.command == "compare":
            output = compare(read_bundle(args.bundle))
            code = 0 if output["comparison_passed"] else 1
            if not args.details:
                output.pop("details", None)
                for case in output.get("cases", []):
                    case.pop("details", None)
        else:
            output = brief(args.board, args.task_id, criteria=args.criteria)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
    except (
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        OSError,
        StopIteration,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {"contract": 1, "kind": "research-error", "error": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
