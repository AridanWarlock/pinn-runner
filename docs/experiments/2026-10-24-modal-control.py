"""Контрольное модальное представление: только проекция IC, без решения PDE в подгонке."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebfit, chebvander
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.catalog import QUADRATIC_CLOCK_TRAINING_ROWS  # noqa: E402

OUT = Path(__file__).resolve().parent
MODES, DEGREE = 64, 13  # Закреплены до первой оценки; подбора размерности нет.


def initial_projection(b, nodes):
    z, weights = leggauss(nodes)
    xi, weights = (z + 1) / 2, weights / 2
    length = np.log1p(1 / b)
    x = b * np.expm1(length * xi)
    initial = np.exp(length * xi / 2) * np.sin(np.pi * x) ** 3
    sine = np.sin(np.pi * xi[:, None] * np.arange(1, MODES + 1)[None, :])
    return 2 * ((weights * initial) @ sine)


def fit_initial_only(nodes):
    shapes = np.array([b for a, b in QUADRATIC_CLOCK_TRAINING_ROWS])
    targets = np.stack([initial_projection(b, nodes) for b in shapes])
    weights = chebfit(2 * shapes - 3, targets, DEGREE)
    return weights, float(np.linalg.cond(chebvander(2 * shapes - 3, DEGREE)))


class ModalControl(torch.nn.Module):
    def __init__(self, weights, ignore_b=False, wrong_decay=False):
        super().__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float64))
        self.ignore_b, self.wrong_decay = ignore_b, wrong_decay

    def factors(self, x, a, b, t):
        if self.ignore_b:
            b = b * 0 + 1.5
        z = 2 * b - 3
        polynomials = [torch.ones_like(z), z]
        for _ in range(2, DEGREE + 1):
            polynomials.append(2 * z * polynomials[-1] - polynomials[-2])
        cheb = torch.cat(polynomials, dim=1)
        length = torch.log1p(1 / b)
        n = torch.arange(1, MODES + 1, dtype=x.dtype, device=x.device)[None, :]
        phase = torch.pi * n * torch.log1p(x / b) / length
        eigen = a / b**2 * ((torch.pi * n / length) ** 2 + (0 if self.wrong_decay else 0.25))
        basis = (1 + x / b) ** -0.5 * torch.sin(phase) * torch.exp(-eigen * t)
        return cheb, basis

    def forward(self, features, t):
        x, a, b = features
        cheb, basis = self.factors(x, a, b, t)
        return {"u": ((cheb @ self.coefficients) * basis).sum(dim=1, keepdim=True)}


def operator_check(model, query):
    x, a, b, t = [
        torch.tensor(query[:, i : i + 1], dtype=model.coefficients.dtype, requires_grad=True)
        for i in range(4)
    ]
    u = model([x, a, b], t)["u"]
    dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    div = torch.autograd.grad((a * (1 + x / b) ** 2 * dx).sum(), x)[0]
    return (
        float((dt - div).abs().max().detach()),
        (dt - div).detach(),
        (a / (4 * b**2) * u).detach(),
    )


def gradient_audit(weights, dtype):
    model = ModalControl(weights).to(dtype=dtype)
    rng = np.random.default_rng(20261024)
    perturbed = weights + rng.normal(size=weights.shape) * 1e-3
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor(perturbed, dtype=dtype))
    x = torch.tensor(rng.uniform(0.01, 0.99, (97, 1)), dtype=dtype)
    b = torch.tensor(rng.uniform(1.01, 1.99, (97, 1)), dtype=dtype)
    a, t = torch.full_like(x, 0.15), torch.zeros_like(x)
    target = torch.sin(torch.pi * x) ** 3
    error = model([x, a, b], t)["u"] - target
    loss = error.square().mean()
    gradient = torch.autograd.grad(loss, model.coefficients)[0].detach().numpy()
    # Независимое NumPy-раскрытие всех 896 производных IC-функционала.
    xx, bb = x.numpy(), b.numpy()
    length = np.log1p(1 / bb)
    basis = (1 + xx / bb) ** -0.5 * np.sin(
        np.pi * np.arange(1, MODES + 1)[None, :] * np.log1p(xx / bb) / length
    )
    cheb = chebvander((2 * bb - 3).ravel(), DEGREE)
    residual = ((cheb @ model.coefficients.detach().numpy()) * basis).sum(
        axis=1, keepdims=True
    ) - np.sin(np.pi * xx) ** 3
    expected = 2 / len(xx) * cheb.T @ (residual * basis)
    np.testing.assert_allclose(
        gradient,
        expected,
        atol=2e-7 if dtype == torch.float32 else 2e-14,
        rtol=2e-3 if dtype == torch.float32 else 1e-10,
    )
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-4
    finite_differences = []
    if dtype == torch.float64:
        direction = rng.normal(size=weights.shape)
        direction /= np.linalg.norm(direction)
        exact = float(np.sum(gradient * direction))
        for h in (1e-5, 5e-6):
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    model.coefficients.copy_(
                        torch.tensor(perturbed + sign * h * direction, dtype=dtype)
                    )
                    values.append(float((model([x, a, b], t)["u"] - target).square().mean()))
            fd = (values[0] - values[1]) / (2 * h)
            np.testing.assert_allclose(fd, exact, atol=1e-11, rtol=1e-6)
            finite_differences.append(dict(h=h, actual=fd, expected=exact))
    return dict(
        dtype=str(dtype),
        parameters=int(gradient.size),
        max_gradient_disagreement=float(np.abs(gradient - expected).max()),
        finite_differences=finite_differences,
    )


def main():
    torch.set_num_threads(1)
    weights, condition = fit_initial_only(256)
    independent, _ = fit_initial_only(512)
    projection_disagreement = float(np.abs(weights - independent).max())
    assert projection_disagreement < 1e-10
    audits = [gradient_audit(weights, dtype) for dtype in (torch.float32, torch.float64)]
    rng = np.random.default_rng(20261024)
    query = np.column_stack(
        [
            rng.uniform(0.01, 0.99, 71),
            rng.uniform(0.05, 0.15, 71),
            rng.uniform(1, 2, 71),
            rng.uniform(0.01, 0.99, 71),
        ]
    )
    pde, _, _ = operator_check(ModalControl(weights), query)
    wrong, actual, expected = operator_check(ModalControl(weights, wrong_decay=True), query)
    assert pde < 1e-11 and wrong > 1e-4
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    # Эталон и его оценщик импортируются после фиксации контрольных весов.
    from benchmarks.cases import CASES
    from benchmarks.quadratic_surrogate_quality import evaluate
    from pinnstorch.runner.problem import ProblemAdapter, load_problem
    from pinnstorch.runner.settings import TaskSettings
    from tests.numerics.test_quadratic_clock_quality import NAME, clock_manifest

    case = CASES[NAME]
    adapter = ProblemAdapter(
        load_problem(case.source), TaskSettings.read(case.config("cpu", 1, 11)).domain
    )
    limits = {**clock_manifest()["case_limits"][NAME], **clock_manifest()["physical_limits"]}
    results = {}
    for name, model in [
        ("modal_control", ModalControl(weights).float()),
        ("ignored_shape", ModalControl(weights, ignore_b=True).float()),
    ]:
        metrics, details = evaluate(model, adapter, "cpu", NAME)
        values = {
            **metrics,
            "boundary_max": max(row["boundary_max"] for row in details["hypotheses"]),
        }
        failed = [key for key, threshold in limits.items() if values[key] > threshold]
        results[name] = dict(
            metrics=values, failed=failed, state_unchanged=details["evaluation_state_unchanged"]
        )
        with (OUT / f"2026-10-24-modal-{name}-hypotheses.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(details["hypotheses"][0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(details["hypotheses"])
        results[name]["responses"] = details["responses"]
    assert results["ignored_shape"]["failed"]
    with (OUT / "2026-10-24-modal-control-weights.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["mode", *[f"cheb_{i}" for i in range(DEGREE + 1)]])
        writer.writerows([[n + 1, *weights[:, n]] for n in range(MODES)])
    summary = dict(
        kind="initial_data_only_non_neural_modal_control",
        modes=MODES,
        degree=DEGREE,
        training_rows=QUADRATIC_CLOCK_TRAINING_ROWS,
        initial_quadrature_nodes=256,
        independent_quadrature_nodes=512,
        chebyshev_condition=condition,
        weight_quadrature_disagreement=projection_disagreement,
        pde_max_fp64=pde,
        wrong_decay_pde_max_fp64=wrong,
        gradient_audits=audits,
        limits=limits,
        results=results,
        timing_comparable=False,
    )
    (OUT / "2026-10-24-modal-control-result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
