"""Every-weight operator checks for transport/Burgers, without optimizer steps."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402


def audit(
    directory: Path, case_name: str, dtype: torch.dtype, seed: int, device: str = "cpu"
) -> dict[str, Any]:
    if case_name not in ("advection", "burgers") or dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected transport/Burgers and FP32/FP64")
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit CPU/CUDA device required")
    case = CASES[case_name]
    config = case.config(device, 1, seed)
    config["training"].update(constraints="soft", hard_execution="eager")
    settings = TaskSettings.read(config)
    expected_faces = [("x", "lower")] if case_name == "advection" else [("x", "both")]
    assert [(b.axis, b.side) for b in settings.boundaries] == expected_faces
    output = directory
    output.mkdir(parents=True, exist_ok=False)
    factory = SolverFactory(settings, load_problem(case.source), output, case.source.parent)
    model = factory.build(predicting=True).model.to(device=device, dtype=dtype).eval()
    assert model.device.type == device
    assert all(p.device.type == device for p in model.parameters())
    parameters = tuple(model.parameters())
    original_sha = tensor_fingerprint(model.state_dict())
    values = torch.rand((64, 2), generator=torch.Generator().manual_seed(2718), dtype=dtype)
    values[:, 1] *= 0.5
    xx, tt = torch.meshgrid(
        torch.tensor([0, 0.17, 0.5, 0.83, 1], dtype=dtype),
        torch.tensor([0, 0.13, 0.5], dtype=dtype),
        indexing="ij",
    )
    values = torch.cat([values, torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)])

    values = values.to(device)

    def residuals() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        wrong: torch.Tensor | None
        x, t = [c.detach().clone().requires_grad_() for c in values.split(1, dim=1)]
        u = model([x], t)["u"]
        actual = factory.adapter.residuals({"u": u}, x, t)[next(iter(factory.problem.equations))]
        ux, ut = torch.autograd.grad(
            u, (x, t), torch.ones_like(u), create_graph=True, retain_graph=True
        )
        if case_name == "advection":
            control, wrong = ut + ux, None
        else:
            uxx = torch.autograd.grad(
                ux, x, torch.ones_like(ux), create_graph=True, retain_graph=True
            )[0]
            xn, tn = (v.detach().cpu().numpy().astype(np.float64) for v in (x, t))
            known = np.exp(-tn) * np.sin(np.pi * xn)
            source = known * (-1 + 0.1 * np.pi**2) + np.pi * np.exp(-2 * tn) * np.sin(
                np.pi * xn
            ) * np.cos(np.pi * xn)
            control = ut + u * ux - 0.1 * uxx - torch.tensor(source, dtype=dtype, device=device)
            wrong = actual + (u.detach() - u) * ux
            assert torch.equal(actual, wrong)
        return actual, control, wrong, u

    actual, control, wrong, u = residuals()
    rtol, atol = (2e-4, 5e-6) if dtype == torch.float32 else (1e-10, 1e-11)
    torch.testing.assert_close(actual, control, rtol=rtol, atol=atol)
    loss = actual.square().mean()
    assert float(loss.detach()) > 1e-5
    raw_gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    raw_controls = torch.autograd.grad(
        control.square().mean(), parameters, retain_graph=True, allow_unused=True
    )
    last_bias = [layer for layer in model.net.modules() if isinstance(layer, torch.nn.Linear)][
        -1
    ].bias
    unused: list[str] = []
    for (name, parameter), observed, expected in zip(
        model.named_parameters(), raw_gradients, raw_controls
    ):
        if observed is None or expected is None:
            assert observed is None and expected is None
            assert case_name == "advection" and parameter is last_bias
            unused.append(name)
    gradients = tuple(
        torch.zeros_like(p) if g is None else g for p, g in zip(parameters, raw_gradients)
    )
    controls = tuple(
        torch.zeros_like(p) if g is None else g for p, g in zip(parameters, raw_controls)
    )
    for (name, _), observed, expected in zip(model.named_parameters(), gradients, controls):
        assert torch.isfinite(observed).all(), name
        torch.testing.assert_close(
            observed, expected, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}"
        )
    flat = torch.cat([g.reshape(-1) for g in gradients])
    reference = torch.cat([g.reshape(-1) for g in controls])
    assert float(flat.norm()) > 1e-5
    wrong_relative: float | None = None
    if wrong is not None:
        wrong_gradient = torch.cat(
            [g.reshape(-1) for g in torch.autograd.grad(wrong.square().mean(), parameters)]
        )
        wrong_relative = float((flat - wrong_gradient).norm() / flat.norm())
        assert wrong_relative > 1e-4
    initial_mask = values[:, 1] == 0
    initial_reference = torch.tensor(
        np.sin(np.pi * values[initial_mask, :1].cpu().numpy().astype(np.float64)), dtype=dtype
    )
    initial_reference = initial_reference.to(device)
    initial_max = float((u[initial_mask].detach() - initial_reference).abs().max())
    boundary_max = 0.0
    for location in (0,) if case_name == "advection" else (0, 1):
        mask = values[:, 0] == location
        target = (
            -torch.tensor(
                np.sin(np.pi * values[mask, 1:].cpu().numpy().astype(np.float64)), dtype=dtype
            )
            if case_name == "advection"
            else torch.zeros_like(u[mask])
        )
        target = target.to(device)
        boundary_max = max(boundary_max, float((u[mask].detach() - target).abs().max()))
    # Soft constraints: an untrained network need not fit these data.
    # Validate the prescribed data independently instead of assuming hard satisfaction.
    x_initial = values[initial_mask, :1]
    t_initial = values[initial_mask, 1:]
    prescribed_initial = factory.problem.initial(factory.adapter.context([x_initial], t_initial))[
        "u"
    ]
    initial_definition_max = float((prescribed_initial - initial_reference).abs().max())
    if case_name == "advection":
        face_time = values[values[:, 0] == 0, 1:]
        prescribed_boundary = factory.problem.boundary_values["inflow"](
            factory.adapter.context([torch.zeros_like(face_time)], face_time)
        )
        target_boundary = -torch.tensor(
            np.sin(np.pi * face_time.cpu().numpy().astype(np.float64)), dtype=dtype
        )
        target_boundary = target_boundary.to(device)
        boundary_definition_max = float((prescribed_boundary - target_boundary).abs().max())
    else:
        boundary_definition_max = 0.0
        assert all(b.value == 0 for b in settings.boundaries)
    assert max(initial_definition_max, boundary_definition_max) <= (
        2e-6 if dtype == torch.float32 else 1e-11
    )
    finite_difference_error: float | None = None
    directional: float | None = None
    if dtype == torch.float64:
        direction = torch.randn(
            flat.shape, dtype=dtype, generator=torch.Generator().manual_seed(815)
        )
        direction = direction.to(device)
        direction /= direction.norm()
        ends = list(itertools.accumulate(p.numel() for p in parameters))
        pieces = [direction[a:b] for a, b in zip([0, *ends[:-1]], ends)]
        originals = [p.detach().clone() for p in parameters]
        losses: list[float] = []
        try:
            for sign in (-1, 1):
                with torch.no_grad():
                    for parameter, original, piece in zip(parameters, originals, pieces):
                        parameter.copy_(original + sign * 1e-4 * piece.reshape_as(parameter))
                losses.append(float(residuals()[0].square().mean().detach()))
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
        directional = float(flat @ direction)
        finite_difference_error = abs((losses[1] - losses[0]) / 2e-4 - directional)
        assert finite_difference_error <= 5e-7 + 2e-5 * abs(directional)
    assert tensor_fingerprint(model.state_dict()) == original_sha
    return dict(
        case=case_name,
        device=device,
        dtype=str(dtype),
        seed=seed,
        points=len(values),
        parameters=flat.numel(),
        loss=float(loss.detach()),
        residual_control_max=float((actual - control).detach().abs().max()),
        gradient_control_max=float((flat - reference).abs().max()),
        gradient_relative_l2=float((flat - reference).norm() / flat.norm()),
        detached_gradient_relative_l2=wrong_relative,
        initial_max=initial_max,
        boundary_max=boundary_max,
        directional_derivative=directional,
        finite_difference_error=finite_difference_error,
        state_unchanged=True,
        unused_constant_bias=";".join(unused),
        initial_definition_max=initial_definition_max,
        boundary_definition_max=boundary_definition_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = [
        audit(args.output / f"{case}-{dtype}-{seed}", case, dtype, seed)
        for case, dtype, seed in itertools.product(
            ("advection", "burgers"), (torch.float32, torch.float64), (19, 73)
        )
    ]
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        device="cpu",
        models=len(rows),
        optimizer_steps=0,
        finite_differences=sum(r["finite_difference_error"] is not None for r in rows),
        all_states_unchanged=all(r["state_unchanged"] for r in rows),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
