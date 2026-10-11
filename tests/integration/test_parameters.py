"""Parameterized forward problems with named inputs, not inverse coefficients."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import loadmat

from pinnstorch.contracts import ArrayOutputs, FloatArray, TensorOutputs
from pinnstorch.data import (
    BoundaryCondition,
    InitialCondition,
    PointCloud,
    create_interval_data,
)
from pinnstorch.parameter_functions import bind_parameters
from tests.integration.test_runner_files import launch, prepare_input


def test_parameter_sampling_and_callbacks() -> None:
    data = create_interval_data(
        [0, 1],
        [0, 1],
        3,
        2,
        parameters={
            "k": {"interval": [0.1, 0.5], "points": 3},
            "amplitude": {"interval": [1.0, 2.0], "points": 2},
        },
    )
    mesh = PointCloud(lambda: data)
    assert mesh.spatial_dim == 1
    assert mesh.input_dim == 3
    assert mesh.parameter_names == ("k", "amplitude")
    assert mesh.spatial_domain_mesh.shape == (18, 2, 3)
    np.testing.assert_allclose(mesh.lb, [0, 0.1, 1, 0])
    np.testing.assert_allclose(mesh.ub, [1, 0.5, 2, 1])
    columns, times = mesh.collection_points(100, use_lhs=True)
    assert columns.shape == (100, 3)
    assert times.shape == (100, 1)
    assert np.all(columns >= mesh.lb[:-1]) and np.all(columns <= mesh.ub[:-1])

    def initial(spatial: FloatArray, *, parameters: dict[str, FloatArray]) -> ArrayOutputs:
        assert spatial.shape == (18, 1)
        return {"T": parameters["amplitude"] * np.ones_like(spatial)}

    def pde(
        outputs: TensorOutputs,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        parameters: dict[str, torch.Tensor],
    ) -> TensorOutputs:
        return {**outputs, "residual": x + t + parameters["k"]}

    def boundary(
        outputs: TensorOutputs,
        spatial: list[torch.Tensor],
        t: torch.Tensor,
        *,
        parameters: dict[str, torch.Tensor],
    ) -> TensorOutputs:
        assert len(spatial) == 1
        return {**outputs, "bc": outputs["T"] - parameters["amplitude"]}

    functions = bind_parameters(
        {"initial_conditions": initial, "pde_fn": pde, "boundary_functions": {"bc": boundary}},
        1,
        mesh.parameter_names,
    )
    initial_dataset = InitialCondition(
        mesh, solution=["T"], initial_fun=functions["initial_conditions"]
    )
    np.testing.assert_array_equal(initial_dataset[:][2]["T"].numpy(), data.spatial[2])
    boundary_dataset = BoundaryCondition(
        mesh, "bc", functions["boundary_functions"]["bc"], "x", "both"
    )
    assert len(boundary_dataset) == 2 * 3 * 2 * 2
    coordinates, time, _ = boundary_dataset[:]
    output = {"T": coordinates[-1]}
    assert (
        torch.count_nonzero(functions["boundary_functions"]["bc"](output, coordinates, time)["bc"])
        == 0
    )
    residual = functions["pde_fn"](output, *coordinates, time)["residual"]
    torch.testing.assert_close(residual, coordinates[0] + time + coordinates[1])


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_parameterized_runner(tmp_path: Path, dimension: int) -> None:
    input_dir = tmp_path / "input"
    prepare_input(input_dir)
    cfg = yaml.safe_load((input_dir / "config.yaml").read_text())
    cfg["domain"].update(
        x_points=3, parameters={"conductivity": {"interval": [0.1, 0.5], "points": 3}}
    )
    for axis in ("y", "z")[: dimension - 1]:
        cfg["domain"].update({f"{axis}_interval": [0, 1], f"{axis}_points": 2})
    cfg["net"]["layers"][0] = dimension + 2
    cfg["train_datasets"] = cfg["train_datasets"][:2] + [
        {
            f"bc_{axis}": {
                "_target_": "pinnstorch.data.BoundaryCondition",
                "bc_func_name": "boundary",
                "axis": axis,
                "location": "both",
            }
        }
        for axis in ("x", "y", "z")[:dimension]
    ]
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (input_dir / "functions.py").write_text("""import numpy as np
from pinnstorch.utils import gradient

def initial_conditions(spatial, *, parameters):
    return {"T": parameters["conductivity"] * np.prod(np.sin(np.pi * spatial), axis=1, keepdims=True)}

def pde_fn(outputs, *coordinates, parameters):
    spatial = coordinates[:-1]
    derivatives = gradient(outputs["T"], list(coordinates))
    laplacian = sum(gradient(derivatives[i], axis)[0] for i, axis in enumerate(spatial))
    outputs["residual"] = derivatives[-1] - parameters["conductivity"] * laplacian
    return outputs

def boundary(outputs, spatial, time, *, parameters):
    assert len(spatial) in (1, 2, 3)
    outputs["boundary"] = outputs["T"]
    return outputs

boundary_functions = {"boundary": boundary}
""")
    output_dir = tmp_path / "train"
    result = launch(input_dir, output_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    # The same weights must predict at a coefficient absent from the training grid.
    cfg["domain"]["parameters"]["conductivity"] = {"interval": [0.2, 0.4], "points": 2}
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    prediction_dir = tmp_path / "predict"
    result = launch(
        input_dir, prediction_dir, "predict", str(output_dir / "checkpoints/best.ckpt")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    saved = loadmat(prediction_dir / "predictions.mat")
    expected = create_interval_data(**cfg["domain"])
    np.testing.assert_array_equal(saved["conductivity"], expected.spatial[-1])
    assert saved["T"].shape == (3 * 2 ** (dimension - 1) * 2, 3)
    assert np.isfinite(saved["T"]).all()

    checkpoint = torch.load(output_dir / "checkpoints/best.ckpt", map_location="cpu")
    assert checkpoint["hyper_parameters"]["input_names"] == (
        *("x", "y", "z")[:dimension],
        "conductivity",
        "t",
    )
    if dimension == 1:
        retrain_dir = tmp_path / "retrain"
        result = launch(
            input_dir, retrain_dir, "retrain", str(output_dir / "checkpoints/best.ckpt")
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (retrain_dir / "checkpoints/best.ckpt").is_file()

    cfg["domain"]["parameters"] = {"other_coefficient": {"interval": [0.2, 0.4], "points": 2}}
    (input_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    rejected_dir = tmp_path / "wrong-parameter"
    result = launch(input_dir, rejected_dir, "predict", str(output_dir / "checkpoints/best.ckpt"))
    assert result.returncode == 1
    assert "Checkpoint input names" in (rejected_dir / "error.log").read_text()


@pytest.mark.parametrize(
    "parameters",
    [
        {"x": {"interval": [0, 1], "points": 2}},
        {"bad-name": {"interval": [0, 1], "points": 2}},
        {"k": {"interval": [0, 1]}},
        {"k": {"interval": [1, 0], "points": 2}},
        {"k": {"interval": [0, 1], "points": True}},
    ],
)
def test_invalid_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        create_interval_data([0, 1], [0, 1], 3, 3, parameters=parameters)


def test_analytic_heat_residual_for_varying_coefficient() -> None:
    from pinnstorch.utils import gradient

    x = torch.linspace(0.1, 0.9, 7, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    k = torch.linspace(0.1, 0.5, 7, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    t = torch.full_like(x, 0.2).requires_grad_()

    def pde(
        outputs: TensorOutputs,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        parameters: dict[str, torch.Tensor],
    ) -> TensorOutputs:
        dx, dt = gradient(outputs["T"], [x, t])
        outputs["residual"] = dt - parameters["k"] * gradient(dx, x)[0]
        return outputs

    def initial(spatial: FloatArray, *, parameters: dict[str, FloatArray]) -> ArrayOutputs:
        return {"T": np.sin(np.pi * spatial)}

    def output_transform(
        outputs: TensorOutputs,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        parameters: dict[str, torch.Tensor],
    ) -> TensorOutputs:
        return {"T": outputs["potential"] * parameters["k"]}

    functions = bind_parameters(
        {"pde_fn": pde, "initial_conditions": initial, "output_fn": output_transform}, 1, ["k"]
    )
    temperature = torch.exp(-k * torch.pi**2 * t) * torch.sin(torch.pi * x)
    result = functions["pde_fn"]({"T": temperature}, x, k, t)
    torch.testing.assert_close(result["residual"], torch.zeros_like(x), atol=1e-14, rtol=0)
    transformed = functions["output_fn"]({"potential": x.square()}, x, k, t)
    torch.testing.assert_close(gradient(transformed["T"], x)[0], 2 * k * x)
