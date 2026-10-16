"""Diagnose finite-precision background differences without training or optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.background_equivalence import BackgroundEquivalence  # noqa: E402
from benchmarks.cases import CASES  # noqa: E402
from pinnstorch.runner.checkpoint_io import load_checkpoint  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402
from scripts.benchmark_solutions import environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument(
        "--case", choices=tuple(name for name in CASES if "polynomial" in name), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 150])
    parser.add_argument("--points", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (
        args.report.exists()
        or min(args.points) < 1
        or (args.device == "cpu" and max(args.points) > 16)
    ):
        parser.error(
            "New report and positive counts required; CPU is limited to small tool checks"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no fallback")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    settings = TaskSettings.load(args.trial / "config.yaml")
    settings = replace(settings, runtime=replace(settings.runtime, device=args.device, devices=1))
    case = CASES[args.case]
    probe = BackgroundEquivalence(
        settings, load_problem(case.source), args.trial / "equivalence", case.source.parent
    )
    rows: list[dict[str, Any]] = []
    hashes = {}
    for step in args.snapshots:
        checkpoint = args.trial / f"lb-{step}.pt"
        state = load_checkpoint(checkpoint, map_location="cpu")
        hashes[str(step)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for dtype in (torch.float32, torch.float64):
            for count in args.points:
                rows.append({"snapshot": step, **probe.compare(count, dtype, state)})
    report = {
        "contract": 1,
        "case": case.name,
        "environment": environment(args.device),
        "snapshot_sha256": hashes,
        "collocation_sha256": probe.collocation_hashes,
        "rows": rows,
        "limitations": "No training/optimizer update or timing claim. Same saved float32 weights and coordinates cast to each dtype. model_step and actual backward policy used. Collocation reconstructed from saved config/seed, not recovered from original training snapshots. Symmetric warmup, self-repeat checks, independent backgrounds. FP64 evaluation is not FP64 training; it cannot establish the exact cause of historical CUDA optimizer divergence.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
