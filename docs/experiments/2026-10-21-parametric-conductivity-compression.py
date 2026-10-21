"""Information loss, solution sensitivity and energy-bound CPU diagnostics."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
spec = importlib.util.spec_from_file_location("reference", f"{PREFIX}-reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def spacetime_inner(left, right, x):
    return np.trapz(np.trapz(left * right, x, axis=0), module.TIMES)


def main():
    fine = {b: module.reference(1024, 0, b) for b in (-0.7, 0, 0.7)}
    coarse = {b: module.reference(512, 0, b) for b in (-0.7, 0.7)}
    x = fine[0][0]
    local_gap = fine[-0.7][1][256, 10] - fine[0.7][1][256, 10]
    coarse_gap = coarse[-0.7][1][128, 10] - coarse[0.7][1][128, 10]
    sensor_error, field_error = 0.0, 0.0
    for a, b in module.PAIRS:
        log_sensor0, log_sensor1 = np.log(8 * module.coefficient(np.array([0, 0.5]), a, b))
        recovered_a, recovered_b = log_sensor0 + log_sensor1, -log_sensor1
        sensor_error = max(sensor_error, abs(a - recovered_a), abs(b - recovered_b))
        field_error = max(
            field_error,
            np.max(
                np.abs(
                    module.coefficient(x, a, b) - module.coefficient(x, recovered_a, recovered_b)
                )
            ),
        )
    outside = module.reference(1024, 0, 0, extra=0.3)
    outside_coarse = module.reference(512, 0, 0, extra=0.3)
    alias_difference = outside[1] - fine[0][1]
    alias_max = float(np.abs(alias_difference).max())
    alias_refinement = float(np.abs(outside[1][::2] - outside_coarse[1]).max())
    derivatives = []
    derivative_errors = []
    for axis in range(2):
        estimates = []
        for h in (1e-3, 5e-4):
            pair = [0.0, 0.0]
            pair[axis] = h
            plus = module.reference(1024, *pair)[1]
            pair[axis] = -h
            minus = module.reference(1024, *pair)[1]
            estimates.append((plus - minus) / (2 * h))
        derivative_errors.append(
            float(
                np.sqrt(
                    spacetime_inner(estimates[0] - estimates[1], estimates[0] - estimates[1], x)
                    / spacetime_inner(estimates[1], estimates[1], x)
                )
            )
        )
        derivatives.append(estimates[1])
    gram = np.array([[spacetime_inner(p, q, x) for p in derivatives] for q in derivatives])
    singular_values = np.sqrt(np.linalg.eigvalsh(gram))
    cosine = gram[0, 1] / np.sqrt(gram[0, 0] * gram[1, 1])
    reflected = [module.reference(1024, a, 0)[1] for a in (-0.7, 0.7)]
    resistances = [
        quad(lambda xx: 1 / module.coefficient(xx, a, 0), 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
        for a in (-0.7, 0.7)
    ]
    energy_rows = []
    initial_norm = np.sqrt(np.trapz(np.sin(np.pi * x) ** 2, x))
    for label, u, v, eta in (
        ("omit_b_minus", fine[-0.7][1], fine[0][1], 0.7),
        ("omit_b_plus", fine[0.7][1], fine[0][1], 0.7),
        ("unseen_extra_mode", outside[1], fine[0][1], 0.3),
    ):
        observed = np.sqrt(np.trapz((u - v) ** 2, x, axis=0)).max() / initial_norm
        bound = np.sqrt(2) * np.sinh(eta / 2)
        energy_rows.append(
            dict(
                label=label,
                log_coefficient_error=eta,
                maximum_field_error_relative_to_initial=float(observed),
                continuous_bound=float(bound),
                measurement_within_bound=bool(observed <= bound),
            )
        )
    result = dict(
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/parametric-conductivity-compression-protocol-2026-10-21.md").read_bytes()
        ).hexdigest(),
        optimizer_steps=0,
        eigensolver_fallbacks=module.EIGENSOLVER_FALLBACKS,
        local_value_collision=dict(
            x=0.25,
            t=0.25,
            c_both=0.125,
            solution_gap=float(local_gap),
            grid_difference=float(abs(local_gap - coarse_gap)),
        ),
        two_sensor_reconstruction=dict(
            parameter_max_error=float(sensor_error), coefficient_max_error=float(field_error)
        ),
        outside_family_sensor_alias=dict(
            extra_mode=0.3,
            sensor_max_difference=float(
                np.max(np.abs(module.coefficient(np.array([0, 0.5]), 0, 0, 0.3) - 0.125))
            ),
            solution_max_difference=alias_max,
            grid_difference=alias_refinement,
        ),
        sensitivity=dict(
            relative_step_changes=derivative_errors,
            gram=gram.tolist(),
            singular_values=singular_values.tolist(),
            normalized_cross_inner_product=float(cosine),
        ),
        effective_coefficient_collision=dict(
            resistances=resistances,
            maximum_field_gap=float(np.abs(reflected[0] - reflected[1]).max()),
            reflection_max_error=float(np.abs(reflected[0] - reflected[1][::-1]).max()),
        ),
        compression_energy_controls=energy_rows,
    )
    result["all_passed"] = bool(
        local_gap > 0.01
        and abs(local_gap - coarse_gap) <= 1e-5
        and sensor_error <= 1e-12
        and field_error <= 1e-12
        and result["outside_family_sensor_alias"]["sensor_max_difference"] <= 1e-12
        and alias_max > 0.005
        and alias_refinement <= 1e-5
        and max(derivative_errors) <= 1e-5
        and abs(cosine) <= 1e-8
        and singular_values.min() > 1e-3
        and abs(resistances[0] - resistances[1]) <= 1e-12
        and result["effective_coefficient_collision"]["reflection_max_error"] <= 1e-10
        and result["effective_coefficient_collision"]["maximum_field_gap"] > 0.01
        and all(r["measurement_within_bound"] for r in energy_rows)
    )
    Path(f"{PREFIX}-compression.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
    assert result["all_passed"], "Keep failures; do not loosen the protocol"


if __name__ == "__main__":
    main()
