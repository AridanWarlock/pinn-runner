"""Real-network CPU derivative controls for a proposed odd representation."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.convergence import ConvergenceArm, ConvergencePlan  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402


def main():
    torch.set_num_threads(1)
    folder = ROOT / "numerical-reports/unforced037-odd-lifting"
    folder.mkdir(exist_ok=False)
    m = json.loads(
        (
            ROOT / "docs/experiments/2026-10-21-nonheat034-unforced-quality-protocol.json"
        ).read_text()
    )
    case = CASES["burgers_unforced_nu001"]
    rows = []
    points = np.random.default_rng(20261021).uniform(size=(37, 2))
    points[:, 0] = 1.8 * points[:, 0] - 0.9
    points[:, 1] = 0.05 + 0.9 * points[:, 1]
    for dtype in (torch.float32, torch.float64):
        for seed in (11, 29):
            directory = folder / f"{dtype}-{seed}"
            directory.mkdir()
            plan = ConvergencePlan(
                case,
                500,
                (0, 250, 500, 1000, 2000),
                m["case_limits"][case.name],
                "hard_initial_dirichlet",
                4096,
                "cpu",
                1e-12,
            )
            model = (
                SolverFactory(
                    TaskSettings.read(plan.config(seed, ConvergenceArm())),
                    load_problem(case.source),
                    directory,
                    case.source.parent,
                )
                .build(predicting=False)
                .model.to(dtype=dtype)
                .eval()
            )
            before = tensor_fingerprint(model.state_dict())
            named = list(model.named_parameters())
            parameters = tuple(p for _, p in named)
            assert (
                sum(p.numel() for p in parameters) == 2241
                and named[-1][0].endswith("bias")
                and parameters[-1].shape == (1,)
            )

            def derivatives(value, x, t):
                vx, vt = torch.autograd.grad(
                    value, (x, t), torch.ones_like(value), create_graph=True, retain_graph=True
                )
                vxx = torch.autograd.grad(
                    vx, x, torch.ones_like(vx), create_graph=True, retain_graph=True
                )[0]
                return vx, vt, vxx

            def evaluate(mode):
                x, t = [
                    torch.tensor(points[:, j : j + 1], dtype=dtype, requires_grad=True)
                    for j in (0, 1)
                ]
                y = -x
                if mode == "direct":
                    value = (model([x], t)["u"] - model([y], t)["u"]) / 2
                    vx, vt, vxx = derivatives(value, x, t)
                else:
                    n = model.net([x], t)["u"]
                    if mode == "wrong":
                        frozen = {k: p.detach() for k, p in model.net.named_parameters()} | {
                            k: b.detach() for k, b in model.net.named_buffers()
                        }
                        reflected = torch.func.functional_call(model.net, frozen, ([y], t))["u"]
                    else:
                        reflected = model.net([y], t)["u"]
                    nx, nt, nxx = derivatives(n, x, t)
                    my, mt, myy = derivatives(reflected, y, t)
                    odd = (n - reflected) / 2
                    oddx = (nx + my) / 2
                    oddt = (nt - mt) / 2
                    oddxx = (nxx - myy) / 2
                    a = t * (1 - x * x)
                    value = -torch.sin(math.pi * x) + a * odd
                    vt = (1 - x * x) * odd + a * oddt
                    vx = -math.pi * torch.cos(math.pi * x) - 2 * t * x * odd + a * oddx
                    vxx = (
                        math.pi**2 * torch.sin(math.pi * x)
                        - 2 * t * odd
                        - 4 * t * x * oddx
                        + a * oddxx
                    )
                residual = vt + value * vx - 0.01 * vxx
                return dict(
                    value=value, vt=vt, vx=vx, vxx=vxx, residual=residual
                ), residual.square().mean()

            direct, loss = evaluate("direct")
            gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, parameters)])
            controlled, control_loss = evaluate("control")
            control_gradient = torch.cat(
                [g.flatten() for g in torch.autograd.grad(control_loss, parameters)]
            )
            wrong, wrong_loss = evaluate("wrong")
            wrong_gradient = torch.cat(
                [g.flatten() for g in torch.autograd.grad(wrong_loss, parameters)]
            )
            absolute, relative = (1e-5, 1e-4) if dtype == torch.float32 else (1e-10, 1e-9)
            assert (
                torch.isfinite(gradient).all()
                and torch.isfinite(control_gradient).all()
                and torch.isfinite(wrong_gradient).all()
            )
            row = dict(
                seed=seed,
                dtype=str(dtype),
                parameters=2241,
                optimizer_steps=0,
                maximum_physical_difference=max(
                    float((direct[k] - controlled[k]).abs().max().detach()) for k in direct
                ),
                physical_controls_passed=all(
                    torch.allclose(direct[k], controlled[k], atol=absolute, rtol=relative)
                    for k in direct
                ),
                loss_control_passed=torch.allclose(
                    loss, control_loss, atol=absolute, rtol=relative
                ),
                gradient_max_absolute_difference=float((gradient - control_gradient).abs().max()),
                gradient_relative_l2=float((gradient - control_gradient).norm() / gradient.norm()),
                gradient_controls_passed=torch.allclose(
                    gradient, control_gradient, atol=absolute, rtol=relative
                ),
                output_bias_name=named[-1][0],
                output_bias_gradient=float(gradient[-1]),
                output_bias_zero_passed=abs(float(gradient[-1])) <= absolute,
                wrong_preserves_residual=torch.allclose(
                    direct["residual"], wrong["residual"], atol=absolute, rtol=relative
                ),
                wrong_gradient_relative_difference=float(
                    (wrong_gradient - gradient).norm() / gradient.norm()
                ),
            )
            row["wrong_gradient_detected"] = row["wrong_gradient_relative_difference"] > 1e-3
            if dtype == torch.float64:
                direction = torch.ones_like(gradient)
                direction[1::2] = -1
                direction /= direction.norm()
                expected = float(torch.dot(gradient, direction))
                original = torch.cat([p.detach().flatten() for p in parameters])

                def assign(vector):
                    start = 0
                    with torch.no_grad():
                        for p in parameters:
                            p.copy_(vector[start : start + p.numel()].reshape(p.shape))
                            start += p.numel()

                for h, label in ((1e-4, "h1"), (5e-5, "h2")):
                    try:
                        assign(original + h * direction)
                        _, plus = evaluate("direct")
                        positive = float(plus.detach())
                        assign(original - h * direction)
                        _, minus = evaluate("direct")
                        negative = float(minus.detach())
                    finally:
                        assign(original)
                    error = abs((positive - negative) / (2 * h) - expected)
                    row[label + "_fd_error"] = error
                    row[label + "_fd_passed"] = error <= 1e-8 + 1e-4 * abs(expected)
            assert tensor_fingerprint(model.state_dict()) == before
            row["state_unchanged"] = True
            rows.append(row)
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-odd-lifting")
    assert not base.with_suffix(".csv").exists() and not base.with_suffix(".json").exists()
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        models=4,
        optimizer_steps=0,
        device="cpu",
        physical_passed=sum(r["physical_controls_passed"] for r in rows),
        gradient_passed=sum(r["gradient_controls_passed"] for r in rows),
        bias_controls_passed=sum(r["output_bias_zero_passed"] for r in rows),
        negative_controls_passed=sum(
            r["wrong_preserves_residual"] and r["wrong_gradient_detected"] for r in rows
        ),
        finite_differences_passed=sum(
            r.get(k + "_fd_passed", False) for r in rows for k in ("h1", "h2")
        ),
        all_states_unchanged=True,
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/unforced-burgers-odd-lifting-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        scope="CPU mathematical prototype only; no public representation, CUDA validation or quality training.",
    )
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
