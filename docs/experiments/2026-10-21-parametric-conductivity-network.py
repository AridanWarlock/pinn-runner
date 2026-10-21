"""CPU controls for actual public parametric FCNs; no optimizer steps."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.api import Equation  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
SOURCE = Path(f"{PREFIX}-problem.py")


def config(seed):
    result = CASES["heat2d"].config("cpu", 1, seed)
    result["domain"] = {
        "space": {"x": {"interval": [0, 1], "points": 7}},
        "time": {"interval": [0, 1], "points": 5},
        "parameters": {
            "a": {"interval": [-0.7, 0.7], "points": 3},
            "b": {"interval": [-0.7, 0.7], "points": 3},
        },
    }
    result["network"]["hidden_layers"] = [32, 32, 32]
    result["training"]["constraints"] = "soft"
    result["training"]["collocation"] = {"sampling": "lhs", "points": 37}
    result["boundaries"] = [{"kind": "dirichlet", "field": "u", "axis": "x", "value": 0}]
    return result


def main():
    torch.set_num_threads(1)
    points = np.random.default_rng(20261021).uniform(size=(37, 4))
    points[:, 0] = 0.05 + 0.9 * points[:, 0]
    points[:, 1:3] = -0.65 + 1.3 * points[:, 1:3]
    points[:, 3] = 0.05 + 0.9 * points[:, 3]
    rows = []
    finite_difference_rows = []
    folder = ROOT / "numerical-reports/parametric038-network"
    folder.mkdir(exist_ok=True)
    for dtype in (torch.float32, torch.float64):
        for seed in (11, 29):
            directory = folder / f"{dtype}-{seed}"
            directory.mkdir(exist_ok=True)
            (directory / "functions.py").write_bytes(SOURCE.read_bytes())
            factory = SolverFactory(
                TaskSettings.read(config(seed)), load_problem(SOURCE), directory, directory
            )
            model = factory.build(predicting=False).model.to(dtype=dtype).eval()
            parameters = tuple(model.parameters())
            saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
            fingerprint = tensor_fingerprint(saved)
            assert sum(p.numel() for p in parameters) == 2305

            def evaluate(mode, selected_points=None):
                query = points if selected_points is None else selected_points
                x, a, b, t = [
                    torch.tensor(query[:, i : i + 1], dtype=dtype, requires_grad=True)
                    for i in range(4)
                ]
                outputs = model([x, a, b], t)
                c = factory.adapter.context([x, a, b], t)
                e = Equation(outputs, c)
                value = outputs["u"]
                ux, ut, uxx = e.diff("u", "x"), e.dt("u"), e.diff("u", "x", 2)
                conductivity = c.coefficients["conductivity"]
                cx = conductivity * (
                    -torch.pi * a * torch.sin(torch.pi * x)
                    - 2 * torch.pi * b * torch.sin(2 * torch.pi * x)
                )
                if mode == "public":
                    residual = factory.adapter.residuals(outputs, x, a, b, t)["heat"]
                elif mode == "expanded":
                    residual = ut - conductivity * uxx - cx * ux
                elif mode == "wrong_detach":
                    residual = ut - e.diff(conductivity.detach() * ux, "x")
                else:
                    residual = (
                        ut
                        - conductivity * (uxx + e.diff("u", "a", 2) + e.diff("u", "b", 2))
                        - cx * ux
                    )
                return value, ux, ut, uxx, residual, residual.square().mean()

            public = evaluate("public")
            expanded = evaluate("expanded")
            gradient = torch.cat(
                [g.flatten() for g in torch.autograd.grad(public[-1], parameters)]
            )
            control_gradient = torch.cat(
                [g.flatten() for g in torch.autograd.grad(expanded[-1], parameters)]
            )
            atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (1e-10, 1e-9)
            order = np.random.default_rng(83).permutation(len(points))
            inverse = np.argsort(order)
            permuted = evaluate("public", points[order])
            chunks = np.array_split(points, 3)
            chunk_values = [evaluate("public", chunk) for chunk in chunks]
            chunk_gradient = torch.zeros_like(gradient)
            for chunk, values in zip(chunks, chunk_values):
                chunk_gradient += torch.cat(
                    [g.flatten() for g in torch.autograd.grad(values[-1], parameters)]
                ) * (len(chunk) / len(points))
            batch_passed = (
                all(
                    torch.allclose(p, q[inverse], atol=atol, rtol=rtol)
                    for p, q in zip(public[:-1], permuted[:-1])
                )
                and all(
                    torch.allclose(
                        public[j], torch.cat([v[j] for v in chunk_values]), atol=atol, rtol=rtol
                    )
                    for j in range(5)
                )
                and torch.allclose(gradient, chunk_gradient, atol=atol, rtol=rtol)
            )
            differences = {}
            for mode in ("wrong_detach", "wrong_parameter_laplacian"):
                wrong = evaluate(mode)
                differences[mode] = float((wrong[-2] - public[-2]).square().mean().sqrt().detach())
            generator = torch.Generator().manual_seed(73)
            directions = [
                torch.randn(p.shape, generator=generator, dtype=dtype) for p in parameters
            ]
            norm = torch.cat([v.flatten() for v in directions]).norm()
            directions = [v / norm for v in directions]
            directional = float(
                sum(
                    (g * d).sum()
                    for g, d in zip(
                        torch.split(gradient, [p.numel() for p in parameters]),
                        [d.flatten() for d in directions],
                    )
                )
            )
            finite_rows = []
            if dtype == torch.float64:
                for h in (1e-5, 5e-6):
                    values = []
                    for sign in (1, -1):
                        model.load_state_dict(saved)
                        with torch.no_grad():
                            for p, d in zip(parameters, directions):
                                p.add_(d, alpha=sign * h)
                        values.append(float(evaluate("public")[-1].detach()))
                    finite = (values[0] - values[1]) / (2 * h)
                    finite_rows.append(
                        dict(
                            step=h,
                            difference=finite,
                            analytic=directional,
                            passed=abs(finite - directional) <= 1e-9 + 1e-4 * abs(directional),
                        )
                    )
                    finite_difference_rows.append(
                        dict(
                            seed=seed,
                            dtype=str(dtype),
                            **finite_rows[-1],
                            absolute_error=abs(finite - directional),
                        )
                    )
                model.load_state_dict(saved)
            # Probe constraints for arbitrary parameters, including intersections.
            aa = torch.tensor([[-0.7], [0], [0.7]], dtype=dtype)
            bb = aa.flip(0)
            zero, one = torch.zeros_like(aa), torch.ones_like(aa)
            with torch.no_grad():
                bc = max(float(model([xx, aa, bb], one)["u"].abs().max()) for xx in (zero, one))
                xx = torch.tensor([[0.13], [0.51], [0.87]], dtype=dtype)
                ic = float((model([xx, aa, bb], zero)["u"] - torch.sin(torch.pi * xx)).abs().max())
            row = dict(
                seed=seed,
                dtype=str(dtype),
                parameters=gradient.numel(),
                optimizer_steps=0,
                physical_max_difference=max(
                    float((p - q).abs().max().detach()) for p, q in zip(public[:-1], expanded[:-1])
                ),
                physical_passed=all(
                    torch.allclose(p, q, atol=atol, rtol=rtol) for p, q in zip(public, expanded)
                ),
                gradient_max_difference=float((gradient - control_gradient).abs().max()),
                gradient_passed=torch.allclose(gradient, control_gradient, atol=atol, rtol=rtol),
                gradient_finite=bool(torch.isfinite(gradient).all()),
                mixed_material_batches_passed=batch_passed,
                split_batch_gradient_max_difference=float((gradient - chunk_gradient).abs().max()),
                initial_state_sha256=fingerprint,
                **differences,
                initial_max_error=ic,
                boundary_max=bc,
                finite_difference_checks=len(finite_rows),
                finite_difference_passed=all(r["passed"] for r in finite_rows),
                state_unchanged=tensor_fingerprint(model.state_dict()) == fingerprint,
            )
            row["passed"] = bool(
                row["physical_passed"]
                and row["gradient_passed"]
                and row["gradient_finite"]
                and batch_passed
                and row["finite_difference_passed"]
                and row["state_unchanged"]
                and ic <= atol
                and bc <= atol
                and all(d > 1e-6 for d in differences.values())
            )
            rows.append(row)
            print(row, finite_rows, flush=True)
    with Path(f"{PREFIX}-network.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with Path(f"{PREFIX}-network-finite-differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(finite_difference_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(finite_difference_rows)
    summary = dict(
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        torch=torch.__version__,
        points_sha256=hashlib.sha256(points.astype("<f8").tobytes()).hexdigest(),
        models=len(rows),
        optimizer_steps=0,
        finite_difference_checks=sum(r["finite_difference_checks"] for r in rows),
        all_passed=all(r["passed"] for r in rows),
    )
    Path(f"{PREFIX}-network.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    assert summary["all_passed"], "Preserve failed controls"


if __name__ == "__main__":
    main()
