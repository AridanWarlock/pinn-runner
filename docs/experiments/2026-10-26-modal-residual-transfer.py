"""Диагностика выведенной L2-границы и необходимости точного начального поля."""

import csv
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from benchmarks.quadratic_modal_case import SOURCE, modal_config
from benchmarks.quadratic_modal_objective import ModalResidualObjective
from pinnstorch.api import Equation
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings

OUT = Path(__file__).parent


def main():
    torch.set_num_threads(1)
    helper = runpy.run_path(str(OUT / "2026-10-26-modal-capacity-control.py"))
    oracle = helper["InitialProjectionOracle"]
    forms = helper["FORMS"]
    adapter = ProblemAdapter(
        load_problem(SOURCE), TaskSettings.read(modal_config("cpu", 1, 11)).domain
    )
    objective = ModalResidualObjective(adapter.initial_tensors, forms, quadrature=512)
    nodes, weights = np.polynomial.legendre.leggauss(512)
    x = torch.tensor((nodes + 1) / 2, dtype=torch.float64)[:, None]
    quadrature = torch.tensor(weights / 2, dtype=torch.float64)[:, None]
    # Один заранее фиксированный ненулевой вектор ошибки; не оптимизация и не подбор по T.
    n = torch.arange(2, 33, dtype=torch.float64)[None, :]
    perturbation = 0.002 * torch.cos(n) / (n.square() - 1)
    coefficients = objective.targets + perturbation
    model = oracle(adapter, coefficients).double()
    rows = []
    wrong_counterexamples = 0
    # Эталон положительного времени импортируется после фиксации коэффициентов.
    from benchmarks.quadratic_reference import reference_fields

    for index, shape in enumerate(forms):
        if shape not in (1.0, 1.5, 2.0):
            continue
        length = np.log1p(1 / shape)
        r = 1 + x / shape
        phi1 = r**-0.5 * torch.sin(torch.pi * torch.log1p(x / shape) / length)
        f = torch.sin(torch.pi * x) ** 3
        norm_phi = shape * length / 2
        c1 = float((quadrature * f * phi1).sum() / norm_phi)
        integral_A = float(
            (objective.metric_weights[index] * perturbation[0].square()).sum()
            + objective.truncation_floor[index]
        )
        first_A = 0.15 / shape**2 * ((np.pi / length) ** 2 + 0.25)
        tau_A = -np.expm1(-2 * first_A) / (2 * first_A)
        for scale in (0.05, 0.15):
            first = scale / shape**2 * ((np.pi / length) ** 2 + 0.25)
            second = scale / shape**2 * ((2 * np.pi / length) ** 2 + 0.25)
            tau = -np.expm1(-2 * first) / (2 * first)
            integral = integral_A * (scale / 0.15) ** 2 * tau / tau_A
            D = norm_phi * tau * (second - first) ** 2 * c1**2
            field_bound = np.sqrt(integral / D)
            gradient_bound = np.sqrt((1 + 1 / shape) ** 2 * second / first * integral / D)
            for instant in (0.0, 0.001, 0.1, 0.5, 1.0):
                xx = x.clone().requires_grad_()
                aa, bb, tt = (
                    torch.full_like(xx, scale),
                    torch.full_like(xx, shape),
                    torch.full_like(xx, instant, requires_grad=True),
                )
                value = model([xx, aa, bb], tt)["u"]
                derivative = Equation({"u": value}, adapter.context([xx, aa, bb], tt)).diff(
                    "u", "x"
                )
                exact = reference_fields(
                    xx.detach().numpy(), tt.detach().numpy(), a=scale, b=shape
                )
                exact_value, exact_dx = torch.tensor(exact.temperature), torch.tensor(exact.dx)
                conductivity = scale * (1 + xx / shape) ** 2

                def relative(actual, expected):
                    return float(
                        torch.sqrt(
                            (quadrature * (actual - expected).square()).sum()
                            / (quadrature * expected.square()).sum()
                        ).detach()
                    )

                field_error = relative(value, exact_value)
                derivative_error = relative(derivative, exact_dx)
                flux_error = relative(-conductivity * derivative, -conductivity * exact_dx)
                assert field_error <= field_bound * (1 + 1e-8)
                assert max(derivative_error, flux_error) <= gradient_bound * (1 + 1e-8)
                wrong_value = value + 0.1 * phi1 * torch.exp(-first * tt)
                wrong_error = relative(wrong_value, exact_value)
                wrong_counterexamples += wrong_error > field_bound
                if instant == 1.0:
                    correct_residual = adapter.residuals({"u": value}, xx, aa, bb, tt)["heat"]
                    # phi1 здесь пересчитывается по тому же leaf xx для реальных производных.
                    wrong_phi = (1 + xx / shape) ** -0.5 * torch.sin(
                        torch.pi * torch.log1p(xx / shape) / length
                    )
                    wrong_field = value + 0.1 * wrong_phi * torch.exp(-first * tt)
                    wrong_residual = adapter.residuals({"u": wrong_field}, xx, aa, bb, tt)["heat"]
                    torch.testing.assert_close(
                        correct_residual, wrong_residual, atol=1e-12, rtol=1e-10
                    )
                rows.append(
                    dict(
                        a=scale,
                        b=shape,
                        t=instant,
                        field=field_error,
                        dx=derivative_error,
                        flux=flux_error,
                        field_bound=field_bound,
                        dx_flux_bound=gradient_bound,
                        wrong_initial_field_error=wrong_error,
                    )
                )
    assert wrong_counterexamples == len(rows)
    path = OUT / "2026-10-26-modal-residual-transfer.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        cases=len(rows),
        bounds_satisfied=True,
        incorrect_initial_counterexamples=int(wrong_counterexamples),
        quadrature=512,
        reference_modes=64,
        interval_certificate=False,
        trained=False,
    )
    (OUT / "2026-10-26-modal-residual-transfer.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
