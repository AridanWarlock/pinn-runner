"""The existing convergence executor trains and independently evaluates both representations."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from benchmarks.cases import CASES
from benchmarks.conductivity import physics, source_sha256
from benchmarks.convergence import ConvergenceArm, ConvergencePlan, convergence_trial
from benchmarks.policies import CartesianGrid, TrainingBudget

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("beta", [0, 1])
def test_conductivity_convergence_uses_materialized_physics_and_physical_flux(
    tmp_path: Path, beta: int
) -> None:
    torch.set_num_threads(1)
    name = f"heat_conductivity_beta{beta}"
    case = replace(
        CASES[name],
        grid=CartesianGrid((3, 3), (3, 3), time_interval=(0, 1)),
        budget=TrainingBudget(layers=(4, 4)),
    )
    plan = ConvergencePlan(
        case,
        1,
        (0, 1),
        dict(
            relative_l2=0.002,
            final_relative_l2=0.003,
            pde_rmse=0.003,
            worst_time_relative_l2=0.005,
            physical_flux_max=2e-5,
        ),
        "hard_initial_robin",
        7,
        "cpu",
    )
    reports = []
    for representation in ("formula", "table"):
        arm = ConvergenceArm(
            encoding="neumann_polynomial", coefficient_representation=representation
        )
        result = convergence_trial(plan, arm, 19, tmp_path / representation)
        assert result["physics"] == physics(name)
        assert result["problem_source_sha256"] == source_sha256(name, representation)
        assert result["adam_steps"] == 1
        assert result["snapshots"][-1]["lb_steps"] == 1
        assert result["snapshots"][-1]["metrics"]["pde_rmse"] > 0.003
        for snapshot in result["snapshots"]:
            assert snapshot["metrics"]["initial_rmse"] <= 2e-6
            assert snapshot["metrics"]["physical_flux_max"] <= 2e-5
            assert len(snapshot["boundary_operators"]) == 4
        reports.append(result)
    assert reports[0]["initial_parameters_sha256"] == reports[1]["initial_parameters_sha256"]
    assert reports[0]["fixed_objective_sha256"] == reports[1]["fixed_objective_sha256"]
    assert reports[0]["initial_state_sha256"] != reports[1]["initial_state_sha256"]
    with pytest.raises(ValueError, match="explicit"):
        _ = case.source
