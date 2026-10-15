"""Real tensor-product MAT layout and cross-platform source identity."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from benchmarks.cases import CASES
from scripts.benchmark_cosine_spectrum import analyze


def test_saved_spectrum_accepts_windows_line_endings_and_recovers_slow_error(
    tmp_path: Path,
) -> None:
    case = CASES["heat_neumann3d"]
    task = tmp_path / "task"
    task.mkdir()
    source = case.source.read_text()
    (task / "functions.py").write_bytes(source.replace("\n", "\r\n").encode())
    (tmp_path / "result.json").write_text(json.dumps({"case": case.name, "seed": 19}))
    prediction = tmp_path / "prediction"
    prediction.mkdir()
    nodes = np.linspace(0, 1, 5)
    mesh = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    spatial = {name: c.reshape(-1, 1) for name, c in zip("xyz", mesh)}
    times = np.array([[0.0], [0.5]])
    error = 0.2 * np.cos(np.pi * spatial["x"]) * np.ones((1, 2))
    values = case.reference(spatial, times.T) + error
    savemat(prediction / "predictions.mat", {**spatial, "t": times, "u": values})
    report = analyze(tmp_path)
    assert report["shape"] == [5, 5, 5]
    final = report["rows"][-1]
    assert final["slower_nonconstant_fraction"] == pytest.approx(1, abs=1e-12)
    assert final["error_energy"] == pytest.approx(0.02, rel=1e-12)
    assert report["largest_final_modes"][0]["index"] == (1, 0, 0)
    assert report["functions_sha256"] != report["normalized_functions_sha256"]
    (task / "functions.py").write_text(source + "\n# changed source\n")
    with pytest.raises(ValueError, match="source differs"):
        analyze(tmp_path)
