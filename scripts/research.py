"""Small, read-only research interface. Does not import Torch or launch training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_tools.board import brief  # noqa: E402
from research_tools.comparison import compare  # noqa: E402
from research_tools.evidence import collect, status, write_bundle  # noqa: E402
from research_tools.remote_status import remote_status  # noqa: E402
from research_tools.schema import Json, load_manifest, read_object  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("status", help="Read only the small study status.json")
    state.add_argument("--root", required=True)
    state.add_argument(
        "--remote", action="store_true", help="Read Windows status through guarded SSH"
    )
    gather = sub.add_parser("collect", help="Validate a complete study and write compact evidence")
    gather.add_argument("--root", type=Path, required=True)
    gather.add_argument("--manifest", type=Path, required=True)
    gather.add_argument("--output", type=Path, required=True)
    comparison = sub.add_parser("compare", help="Evaluate explicit paired gates")
    comparison.add_argument("--bundle", type=Path, required=True)
    comparison.add_argument("--details", action="store_true")
    task = sub.add_parser("task", help="Return task goals without implementation history")
    task.add_argument("task_id")
    task.add_argument("--criteria", action="store_true", help="Include full acceptance criteria")
    task.add_argument("--board", type=Path, default=ROOT / ".project-board")
    args = parser.parse_args(argv)
    code = 0
    try:
        output: Json
        if args.command == "status":
            output = remote_status(args.root) if args.remote else status(Path(args.root))
        elif args.command == "collect":
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
            output = compare(read_object(args.bundle / "evidence.json"))
            code = 0 if output["comparison_passed"] else 1
            if not args.details:
                for case in output["cases"]:
                    del case["details"]
        else:
            output = brief(args.board, args.task_id, criteria=args.criteria)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
    except (
        ValueError,
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
