"""Independent CPU analytical and integral balance controls, without training."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cases import CASES  # noqa: E402
from pinnstorch.api import Equation  # noqa: E402
from pinnstorch.runner.problem import ProblemAdapter, load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

SOURCE = ROOT / "benchmarks/problems/heat_spatial_scale_surrogate/functions.py"
PREFIX = ROOT / "docs/experiments/2026-10-21-spatial-scale"


def main():
    torch.set_num_threads(1)
    config = CASES["heat_constant_surrogate"].config("cpu", 1, 11)
    config["domain"]["parameters"] = {"a": {"interval": [0.05, 0.25], "points": 3}}
    adapter = ProblemAdapter(load_problem(SOURCE), TaskSettings.read(config).domain)
    rows = []
    for amplitude in (0.05, 0.1, 0.2, 0.25):
        for instant in (0, 0.2, 0.6, 1):
            x = torch.linspace(0, 1, 257, dtype=torch.float64).reshape(-1, 1).requires_grad_()
            a = torch.full_like(x, amplitude, requires_grad=True)
            t = torch.full_like(x, instant, requires_grad=True)
            field = 4 * x * (1 - x) * torch.exp(-12 * a * t)
            equation = Equation({"u": field}, adapter.context([x, a], t))
            conductivity = equation.coefficients["conductivity"]
            gradient = equation.diff("u", "x")
            dt = equation.dt("u")
            residual = adapter.residuals({"u": field}, x, a, t)["heat"]
            wrong = dt - conductivity * equation.diff("u", "x", 2)
            expected_wrong = 8 * a * (1 - 2 * x).square() * torch.exp(-12 * a * t)
            flow = (-conductivity * gradient).detach().numpy().reshape(-1)
            domain = x.detach().numpy().reshape(-1)
            heat = float(simpson(field.detach().numpy().reshape(-1), x=domain))
            rate = float(simpson(dt.detach().numpy().reshape(-1), x=domain))
            left, right = -float(flow[0]), float(flow[-1])
            rows.append(
                dict(
                    a=amplitude,
                    time=instant,
                    heat_content=heat,
                    heat_content_error=abs(heat - 2 / 3 * math.exp(-12 * amplitude * instant)),
                    heat_rate=rate,
                    outward_left=left,
                    outward_right=right,
                    balance_error=abs(rate + left + right),
                    pde_max=float(residual.detach().abs().max()),
                    missing_gradient_max=float(wrong.detach().abs().max()),
                    wrong_formula_error=float((wrong - expected_wrong).detach().abs().max()),
                )
            )
    with Path(f"{PREFIX}-analytic-controls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        device="cpu",
        dtype="float64",
        optimizer_updates=0,
        source_sha256=hashlib.sha256(SOURCE.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        controls=len(rows),
        **{
            key: max(row[key] for row in rows)
            for key in ("heat_content_error", "balance_error", "pde_max", "wrong_formula_error")
        },
        smallest_detected_missing_gradient=max(
            0, min(row["missing_gradient_max"] for row in rows)
        ),
    )
    summary["all_passed"] = (
        all(
            summary[key] < 1e-12
            for key in ("heat_content_error", "balance_error", "pde_max", "wrong_formula_error")
        )
        and summary["smallest_detected_missing_gradient"] > 0.05
    )
    Path(f"{PREFIX}-analytic-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, allow_nan=False))
    assert summary["all_passed"]


if __name__ == "__main__":
    main()
