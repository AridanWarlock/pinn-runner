"""Explain a saved Cartesian heat model's modal error by projected PDE forcing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import CASES, BoxNeumannHeatCase  # noqa: E402
from benchmarks.modal_balance import HeatEigenmode, modal_heat_balance  # noqa: E402
from pinnstorch.runner.dirichlet import trace_value  # noqa: E402
from scripts.benchmark_robin_execution import load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--space-points", type=int, default=13)
    parser.add_argument("--time-points", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite a report")
    if not torch.cuda.is_available():
        parser.error("Saved-model numerical diagnostics require remote CUDA")
    original = json.loads((args.trial / "result.json").read_text())
    case = CASES[original["case"]]
    if not isinstance(case, BoxNeumannHeatCase):
        parser.error("Requires a documented box heat eigenmode case")
    if (args.trial / "task/functions.py").read_bytes() != case.source.read_bytes():
        parser.error("Case source differs from the documented analytical problem")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    factory, model = load_model(args.trial, "robin_grouped")
    settings = factory.settings
    if settings.domain.time.interval[0] != 0 or any(
        axis.interval != (0, 1) for axis in settings.domain.space.values()
    ):
        parser.error("Analytical benchmark assumes a unit box and initial time zero")
    names = tuple(settings.domain.space)
    modes = tuple(case.mode_for(axis) for axis in names)
    mode = HeatEigenmode(
        tuple(axis.frequency for axis in modes), tuple(axis.shift for axis in modes)
    )

    def field(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, model(spatial, clock)["u"])

    def source(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(clock)
        for x in spatial:
            result = result + x
        return result

    def reference(spatial: list[torch.Tensor], clock: torch.Tensor) -> torch.Tensor:
        background = torch.ones_like(clock)
        for x in spatial:
            background = background + clock * x
        return cast(
            torch.Tensor, background + mode.spatial(spatial) * clock.mul(-mode.decay_rate).exp()
        )

    lookup = {boundary.axis: boundary for boundary in settings.boundaries}
    if len(lookup) != len(settings.boundaries) or any(
        boundary.side != "both" for boundary in lookup.values()
    ):
        parser.error("Diagnostic supports documented paired-face benchmark definitions")
    from functools import partial

    targets = tuple((partial(trace_value, lookup[axis], factory.adapter),) * 2 for axis in names)
    operators = tuple(
        ((lookup[axis].alpha if lookup[axis].kind == "robin" else 0, 1),) * 2 for axis in names
    )
    if any(
        boundary.kind not in ("neumann", "robin") or boundary.beta != 1
        for boundary in lookup.values()
    ):
        parser.error("Boundary operators differ from documented benchmark cases")
    torch.cuda.synchronize()
    started = time.perf_counter()
    budget = modal_heat_balance(
        field,
        reference,
        source,
        targets,
        operators,
        mode,
        tuple(axis.interval for axis in settings.domain.space.values()),
        settings.domain.time.interval,
        "cuda",
        args.space_points,
        args.time_points,
        args.batch_size,
    )
    torch.cuda.synchronize()
    report = {
        "case": original["case"],
        "seed": original["seed"],
        "sampling_recipe": original["sampling_recipe"],
        "hard_lifting": original.get("hard_lifting", "recursive"),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "space_points": args.space_points,
        "time_points": args.time_points,
        "decay_rate": mode.decay_rate,
        "diagnostic_seconds": time.perf_counter() - started,
        **asdict(budget),
        "limitations": "Gauss spatial quadrature and trapezoidal time integration; linear constant-coefficient heat and one known unit-amplitude eigenmode. Independent physical autograd; no training or new acceptance gate. Boundary forcing retained, not assumed zero.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "report": str(args.report),
                **budget.metrics,
                "diagnostic_seconds": report["diagnostic_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
