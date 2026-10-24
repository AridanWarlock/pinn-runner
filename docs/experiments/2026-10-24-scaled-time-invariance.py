"""Учитель не нужен: проверка симметрии и чувствительности физического масштаба."""

import csv
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.fingerprints import tensor_fingerprint  # noqa: E402
from benchmarks.quadratic_clock_case import SOURCE, clock_config  # noqa: E402
from pinnstorch.data import PointCloud  # noqa: E402
from pinnstorch.runner.factory import SolverFactory  # noqa: E402
from pinnstorch.runner.problem import load_problem  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402

rows = []
for seed in (11, 29):
    for dtype in (torch.float32, torch.float64):
        settings = TaskSettings.read(clock_config("cpu", 1, seed))
        with tempfile.TemporaryDirectory() as directory:
            factory = SolverFactory(settings, load_problem(SOURCE), Path(directory), SOURCE.parent)
            torch.manual_seed(seed)
            model = (
                factory._model(PointCloud(lambda: settings.domain.grid())).to(dtype=dtype).eval()
            )
        before = tensor_fingerprint(model.state_dict())
        x = torch.linspace(0.03, 0.97, 17, dtype=dtype)[:, None]
        max_pair = 0.0
        for shape in (1.0, 1.4, 2.0):
            b = torch.full_like(x, shape)
            for scale, time, reference_time in (
                (0.05, 0.9, 0.3),
                (0.075, 0.8, 0.4),
                (0.125, 0.6, 0.5),
            ):
                actual = model([x, torch.full_like(x, scale), b], torch.full_like(x, time))["u"]
                expected = model(
                    [x, torch.full_like(x, 0.15), b], torch.full_like(x, reference_time)
                )["u"]
                max_pair = max(max_pair, float((actual - expected).abs().max().detach()))
        a = torch.full_like(x, 0.075, requires_grad=True)
        t = torch.full_like(x, 0.6, requires_grad=True)
        b = torch.full_like(x, 1.4)
        field = model([x, a, b], t)["u"]
        da, dt = torch.autograd.grad(field.sum(), (a, t))
        euler = float((a * da - t * dt).abs().max().detach())
        wrong = model([x, torch.full_like(x, 0.1), b], t)["u"]
        wrong_dt = torch.autograd.grad(wrong.sum(), t)[0]
        negative = float((t * wrong_dt).abs().max().detach())
        tolerance = 2e-6 if dtype == torch.float32 else 1e-13
        assert max_pair < tolerance and euler < tolerance
        assert negative > 1e-4 and da.abs().max() > 1e-4
        assert tensor_fingerprint(model.state_dict()) == before
        rows.append(
            dict(
                seed=seed,
                dtype=str(dtype),
                pair_max_absolute=max_pair,
                euler_max_absolute=euler,
                ignored_a_euler_defect=negative,
                scale_derivative_max=float(da.abs().max()),
            )
        )
with Path(__file__).with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(rows)
