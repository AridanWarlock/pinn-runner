"""Малый цикл оценки приведённого времени: 12 физических гипотез и публикация."""

import pytest

from tests.integration.test_quadratic_surrogate_convergence import audit_quadratic_convergence
from tests.numerics.test_quadratic_clock_quality import NAME, clock_manifest

pytestmark = pytest.mark.e2e


def test_clock_convergence_preserves_physical_evidence(tmp_path):
    audit_quadratic_convergence(tmp_path, clock_manifest(), NAME)
