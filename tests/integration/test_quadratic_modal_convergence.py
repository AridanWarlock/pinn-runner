"""Один шаг интегрального обучения и независимая физическая публикация доказательств."""

import pytest

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
