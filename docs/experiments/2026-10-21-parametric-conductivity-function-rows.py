"""Existing internal aligned rows avoid Cartesian codes; LHS does not preserve them."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pinnstorch.data import PointCloud  # noqa: E402
from pinnstorch.data.domains.point_cloud import PointCloudData  # noqa: E402


def make_mesh(codes):
    nx, nt = 7, 5
    dimension = codes.shape[1]
    x = np.tile(np.linspace(0, 1, nx), len(codes))[:, None]
    columns = [x, *[np.repeat(codes[:, i], nx)[:, None] for i in range(dimension)]]
    bounds = (np.array([0, *[-0.7] * dimension, 0]), np.array([1, *[0.7] * dimension, 1]))
    return PointCloud(
        lambda: PointCloudData(
            spatial=columns,
            time=np.linspace(0, 1, nt)[:, None],
            solution={},
            spatial_dimension=1,
            parameter_names=tuple(f"p{j}" for j in range(dimension)),
            normalization_bounds=bounds,
        )
    )


def main():
    codes = np.array([[-0.7, -0.7], [-0.2, -0.2], [0.2, 0.2], [0.7, 0.7]])
    small = make_mesh(codes)
    aligned, _ = small.collection_points(None, use_lhs=False)
    np.random.seed(20261021)
    independent, _ = small.collection_points(256, use_lhs=True)
    aligned_error = float(np.abs(aligned[:, 1] - aligned[:, 2]).max())
    independent_error = float(np.abs(independent[:, 1] - independent[:, 2]).max())
    amplitudes = np.linspace(-0.7, 0.7, 128)
    sensor_locations = np.linspace(0, 1, 129)
    large_codes = amplitudes[:, None] * np.cos(np.pi * sensor_locations)
    large = make_mesh(large_codes)
    expected = 7 * 5 * 128 * 130 * 8
    result = dict(
        optimizer_steps=0,
        small_material_rows=4,
        aligned_material_relation_max_error=aligned_error,
        independent_lhs_material_relation_max_error=independent_error,
        lhs_rows_outside_original_relation=int(
            np.count_nonzero(np.abs(independent[:, 1] - independent[:, 2]) > 1e-12)
        ),
        lhs_rows=256,
        large_function_codes=128,
        code_features=129,
        aligned_spatial_mesh_shape=list(large.spatial_domain_mesh.shape),
        aligned_spatial_mesh_bytes=large.spatial_domain_mesh.nbytes,
        public_domain_grid_modified=False,
        training_sampler_modified=False,
        all_passed=bool(
            aligned_error == 0
            and independent_error > 0.1
            and large.spatial_domain_mesh.nbytes == expected
        ),
    )
    path = ROOT / "docs/experiments/2026-10-21-parametric-conductivity-function-rows.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    assert result["all_passed"]


if __name__ == "__main__":
    main()
