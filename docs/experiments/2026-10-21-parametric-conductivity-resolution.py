"""CPU study of function-code resolution, separate from neural training."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
spec = importlib.util.spec_from_file_location("reference", f"{PREFIX}-reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    functions = (
        (
            "two_modes",
            lambda x: 0.5 * np.cos(np.pi * x) + 0.3 * np.cos(2 * np.pi * x),
            np.pi**2 * 1.7,
        ),
        ("localized", lambda x: 0.6 * np.exp(-(((x - 0.63) / 0.08) ** 2)), 2 * 0.6 / 0.08**2),
        ("oscillatory", lambda x: 0.7 * np.sin(8 * np.pi * x) ** 2, 2 * 0.7 * (8 * np.pi) ** 2),
    )
    rows = []
    alias = {}
    sample = np.linspace(0, 1, 65537)
    for name, g, second_bound in functions:

        def true_coefficient(x):
            return np.exp(g(x)) / 8

        true_fine = module.reference(1024, 0, 0, coefficient_function=true_coefficient)
        true_coarse = module.reference(512, 0, 0, coefficient_function=true_coefficient)
        x, u = true_fine[:2]
        initial_norm = np.sqrt(np.trapz(np.sin(np.pi * x) ** 2, x))
        reference_difference = float(np.abs(u[::2] - true_coarse[1]).max())
        previous = None
        for count in (9, 17, 33, 65, 129):
            nodes = np.linspace(0, 1, count)
            code = g(nodes)

            def interpolated(x):
                return np.exp(np.interp(x, nodes, code)) / 8

            fine = module.reference(1024, 0, 0, coefficient_function=interpolated)
            coarse = module.reference(512, 0, 0, coefficient_function=interpolated)
            eta = second_bound / (8 * (count - 1) ** 2)
            eta_measured = float(np.abs(g(sample) - np.interp(sample, nodes, code)).max())
            bound = np.sqrt(2) * np.sinh(eta / 2)
            measured = np.sqrt(np.trapz((fine[1] - u) ** 2, x, axis=0)).max() / initial_norm
            grid_difference = float(np.abs(fine[1][::2] - coarse[1]).max())
            row = dict(
                material=name,
                code_nodes=count,
                log_second_derivative_bound=float(second_bound),
                log_error_measured=eta_measured,
                log_error_upper_bound=float(eta),
                continuous_solution_error_upper_bound=float(bound),
                maximum_field_error_relative_to_initial=float(measured),
                reference_grid_difference=reference_difference,
                code_grid_difference=grid_difference,
                previous_nodes=0 if previous is None else previous[0],
                previous_log_code_max_difference=0.0
                if previous is None
                else float(np.abs(np.interp(sample, nodes, code) - previous[1]).max()),
                previous_solution_max_difference=0.0
                if previous is None
                else float(np.abs(fine[1] - previous[2]).max()),
                passed=bool(
                    eta_measured <= eta + 1e-12
                    and measured <= bound + 2e-5
                    and reference_difference <= 2e-5
                    and grid_difference <= 2e-5
                ),
            )
            rows.append(row)
            previous = (count, np.interp(sample, nodes, code), fine[1])
            print(name, count, row["passed"], measured, bound, flush=True)
            if name == "oscillatory" and count == 9:
                alias = dict(
                    code_max_log_difference_from_constant=float(np.abs(code).max()),
                    continuous_material_solution_gap=float(np.abs(fine[1] - u).max()),
                )
    with Path(f"{PREFIX}-resolution.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/parametric-conductivity-resolution-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        optimizer_steps=0,
        codes=len(rows),
        eigensolver_fallbacks=module.EIGENSOLVER_FALLBACKS,
        nine_sensor_alias=alias,
        all_passed=bool(
            all(r["passed"] for r in rows)
            and alias["code_max_log_difference_from_constant"] <= 1e-12
            and alias["continuous_material_solution_gap"] > 0.01
        ),
    )
    Path(f"{PREFIX}-resolution.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
    assert result["all_passed"], "Keep all outcomes and unchanged limits"


if __name__ == "__main__":
    main()
