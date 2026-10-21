"""Measure small public input grids; project large grids without allocating them."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pinnstorch.data import PointCloud  # noqa: E402
from pinnstorch.runner.settings import TaskSettings  # noqa: E402


def main():
    rows = []
    for dimension in (0, 2, 4, 6, 8, 12, 16, 32, 129):
        points_per_parameter = 3 if dimension < 32 else 2
        expected = 7 * 5 * (1 + dimension) * points_per_parameter**dimension * 8
        actual = 0
        if dimension <= 6:
            settings = TaskSettings.read(
                dict(
                    version=1,
                    domain=dict(
                        space={"x": {"interval": [0, 1], "points": 7}},
                        time={"interval": [0, 1], "points": 5},
                        parameters={
                            f"p{j}": {"interval": [-0.7, 0.7], "points": points_per_parameter}
                            for j in range(dimension)
                        },
                    ),
                )
            )
            mesh = PointCloud(lambda: settings.domain.grid())
            actual = mesh.spatial_domain_mesh.nbytes
            assert actual == expected
        rows.append(
            dict(
                parameter_dimensions=dimension,
                points_per_parameter=points_per_parameter,
                spatial_points=7,
                time_points=5,
                spatial_mesh_bytes=expected,
                spatial_mesh_gib=expected / 2**30,
                actually_allocated=dimension <= 6,
                measured_bytes=actual,
                includes_other_arrays=False,
                optimizer_steps=0,
            )
        )
    prefix = ROOT / "docs/experiments/2026-10-21-parametric-conductivity"
    with Path(f"{prefix}-input-scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        optimizer_steps=0,
        measured_dimensions=[0, 2, 4, 6],
        projected_dimensions=[8, 12, 16, 32, 129],
        scope="spatial_domain_mesh only; excludes coordinate meshes and other allocations",
        measured_formulas_match=all(
            r["measured_bytes"] == r["spatial_mesh_bytes"] for r in rows if r["actually_allocated"]
        ),
    )
    Path(f"{prefix}-input-scaling.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print([(r["parameter_dimensions"], r["spatial_mesh_gib"]) for r in rows])


if __name__ == "__main__":
    main()
