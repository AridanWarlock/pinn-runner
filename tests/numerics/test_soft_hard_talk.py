"""Physical objective, isolated RNG streams and safe wall-clock states for the talk study."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.fingerprints import tensor_fingerprint
from benchmarks.soft_hard_talk import Objective, make_model, samples, train, trapezoid
from research_tools.soft_hard_talk import ENVIRONMENT, ORDER, SETTINGS, compare, validate


def manifest():
    return dict(
        contract=9,
        kind="soft-hard-talk",
        registration="prospective",
        revision="a" * 40,
        settings=deepcopy(SETTINGS),
        order=[list(v) for v in ORDER],
        environment_requirements=ENVIRONMENT.copy(),
    )


def test_pair_rng_and_stock_loss_have_same_pde_and_face_average():
    torch.set_num_threads(1)
    points = samples(19)
    soft, hard = make_model("soft", 19, "cpu"), make_model("hard", 19, "cpu")
    assert tensor_fingerprint(dict(soft.named_parameters())) == tensor_fingerprint(
        dict(hard.named_parameters())
    )
    assert all(np.array_equal(points[k], samples(19)[k]) for k in points)
    objective = Objective(soft, "soft", points, "cpu")
    loss = objective.loss()
    assert set(soft.loss_terms) == {
        "equations",
        "initial",
        "x_lower",
        "x_upper",
        "y_lower",
        "y_upper",
    }
    torch.testing.assert_close(loss, sum(soft.loss_terms.values()))
    for face in ("x_lower", "x_upper", "y_lower", "y_upper"):
        assert len(objective.fixed.batches[face][1]) == 128
        assert torch.equal(
            objective.fixed.batches[face][2]["weight_0"], torch.full((128, 1), 0.25)
        )
    hard_objective = Objective(hard, "hard", points, "cpu")
    hard_loss = hard_objective.loss()
    assert set(hard.loss_terms) == {"equations"}
    for model, value in [(soft, loss), (hard, hard_loss)]:
        gradients = torch.autograd.grad(value, tuple(model.parameters()))
        assert sum(g.numel() for g in gradients) == 2273
        assert all(torch.isfinite(g).all() for g in gradients)
    # Counterexample: treating y faces as one merged boundary must not halve their weights.
    assert len(objective.fixed.batches) == 6


def test_trapezoid_weights_are_explicit_normalized_and_not_uniform():
    w = trapezoid(81)
    assert w.sum() == pytest.approx(1)
    assert w[0] == w[-1] == pytest.approx(w[1] / 2)


@pytest.mark.parametrize("field", ["total_seconds", "adam_seconds", "pde_points", "face_weight"])
def test_protocol_does_not_allow_silent_budget_or_loss_changes(field):
    m = manifest()
    m["settings"][field] += 1
    with pytest.raises(ValueError):
        validate(m)


def test_small_cpu_engineering_timer_saves_only_completed_optimizer_states(tmp_path: Path):
    # This is a shortened unit check, not an additional scientific trial or hyperparameter comparison.
    torch.set_num_threads(1)
    data = {k: v[:8] for k, v in samples(19).items()}
    settings = deepcopy(SETTINGS)
    settings.update(
        adam_seconds=0.05,
        total_seconds=0.16,
        snapshot_seconds=[0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.16],
    )
    model = make_model("soft", 19, "cpu")
    result = train(model, "soft", 19, data, tmp_path / "trial", "cpu", settings)
    assert result["seconds"] >= 0.16 and result["adam_end_seconds"] >= 0.05
    assert result["snapshots"][0]["stage"] == "initial"
    assert result["snapshots"][-1]["stage"] == "final"
    assert result["closures"] >= result["lbfgs_steps"] >= 1
    for snapshot in result["snapshots"]:
        checkpoint = torch.load(tmp_path / "trial" / snapshot["path"], map_location="cpu")
        assert checkpoint["closures"] == snapshot["closures"]
        assert snapshot["accepted"] is True


def test_contract_inventory_rejects_partial_results_before_any_speed_claim():
    e = dict(
        contract=1,
        kind="soft-hard-talk-evidence",
        manifest=manifest(),
        environment=ENVIRONMENT.copy(),
        gpu="RTX 4070 Ti",
        source_unchanged=True,
        trials=[],
    )
    with pytest.raises(ValueError, match="Incomplete"):
        compare(e)


def test_independent_evaluator_detects_exact_pde_and_does_not_divide_by_zero_field(tmp_path):
    from benchmarks.soft_hard_talk import evaluate

    class Analytical(torch.nn.Module):
        def forward(self, features, time):
            x, y = features
            return {
                "u": torch.exp(-0.2 * torch.pi**2 * time)
                * torch.cos(torch.pi * x)
                * torch.cos(torch.pi * y)
            }

    torch.set_num_threads(1)
    result = evaluate(Analytical(), "cpu", tmp_path)
    assert result["relative_l2"] < 1e-6
    assert result["pde_rmse"] < 1e-6
    assert len(result["temporal"]) == 41
    assert max(face["max"] for face in result["boundaries"].values()) < 1e-6
    assert all(np.isfinite(row["relative_l2"]) for row in result["temporal"])
