"""Run a registered frozen profile under the existing Windows queue lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from research_tools.frozen_profile import run
    from research_tools.queue import registered_manifest, verify_snapshot

    verify_snapshot(args.manifest.parent)
    run(
        Path(r"C:\Projects\pinn-runner"),
        ROOT,
        args.output,
        registered_manifest(args.manifest.parent),
    )


if __name__ == "__main__":
    main()
