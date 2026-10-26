"""Один шаг интегрального обучения и независимая физическая публикация доказательств."""

from copy import deepcopy

import pytest

from research_tools.comparison import compare
from research_tools.evidence import read_bundle

from tests.integration.test_quadratic_surrogate_convergence import audit_quadratic_convergence
from tests.numerics.test_quadratic_modal_quality import NAME, modal_manifest

pytestmark = pytest.mark.e2e


def test_modal_integral_training_preserves_physical_evidence(tmp_path):
    trial = audit_quadratic_convergence(tmp_path, modal_manifest(), NAME)
    assert trial["modal_training"]["forms"] == 14
    assert trial["modal_training"]["xt_sampling"] is False
    for row in trial["snapshots"]:
        assert row["adam_residual_points"] == row["lbfgs_residual_points"] == 0
        assert row["modal_form_evaluations"] >= 14
        assert row["modal_integral"] >= 0

    evidence = read_bundle(tmp_path / "bundle")
    for damage in ("objective", "counter", "integral"):
        changed = deepcopy(evidence)
        if damage == "objective":
            changed["trials"][0]["modal_training"]["xt_sampling"] = True
        elif damage == "counter":
            changed["trials"][0]["snapshots"][0]["modal_form_evaluations"] += 14
        else:
            del changed["trials"][0]["snapshots"][0]["modal_integral"]
        with pytest.raises(ValueError):
            compare(changed)
