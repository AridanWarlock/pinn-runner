"""Exercise the real PINN data pipeline and Lightning Trainer on CPU."""

import math
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import lightning as L
import numpy as np
import pytest
import torch

from pinnstorch.data import (
    InitialCondition,
    MeshSampler,
    PINNDataModule,
    PointCloud,
    PointCloudData,
    create_interval_data,
)
from pinnstorch.models import FCN, PINNModule
from pinnstorch.utils import gradient


def heat_pde(
    outputs: dict[str, torch.Tensor], x: torch.Tensor, time: torch.Tensor
) -> dict[str, torch.Tensor]:
    dx, dt = gradient(outputs["temperature"], [x, time])
    outputs["residual"] = dt - 0.3 * gradient(dx, x)[0]
    return outputs


def make_problem() -> tuple[PINNModule, PINNDataModule]:
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    time = np.linspace(0.0, 0.5, 3).reshape(-1, 1)
    temperature = np.sin(math.pi * x) * np.exp(-0.3 * math.pi**2 * time.T)

    def read_data() -> PointCloudData:
        return PointCloudData(spatial=[x], time=time, solution={"temperature": temperature})

    mesh = PointCloud(read_data_fn=read_data)
    initial = InitialCondition(mesh, solution=["temperature"])
    pde = MeshSampler(mesh, collection_points=["residual"], use_lhs=False)
    validation = MeshSampler(mesh, solution=["temperature"])
    data = PINNDataModule(train_datasets=[initial, pde], val_dataset=validation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        net = FCN([2, 16, 16, 1], mesh.lb, mesh.ub, ["temperature"])
    model = PINNModule(
        net=net,
        pde_fn=heat_pde,
        optimizer=partial(torch.optim.Adam, lr=0.01),
        amp=False,
        jit_compile=False,
        cudagraph_compile=False,
    )
    return model, data


@pytest.mark.parametrize("pde_first", [False, True])
def test_combined_loss_counts_each_condition_once(pde_first: bool) -> None:
    model, data = make_problem()
    initial, pde = data.train_datasets
    if pde_first:
        data.train_datasets.reverse()
    data.setup("fit")
    model.function_mapping = data.function_mapping
    model.functions["val"] = False
    model.functions["batch_size"] = False
    batch = {key: next(iter(loader)) for key, loader in data.train_dataloader().items()}

    initial_x, initial_time, target = initial[:]
    initial_outputs = model(initial_x, initial_time)
    initial_loss = (initial_outputs["temperature"] - target["temperature"]).square().mean()
    pde_x, pde_time, _ = pde[:]
    pde_x[0].requires_grad_()
    pde_time.requires_grad_()
    residual = heat_pde(model(pde_x, pde_time), pde_x[0], pde_time)["residual"]
    expected = initial_loss + residual.square().mean()

    actual, _ = model.model_step(batch)

    torch.testing.assert_close(actual, expected)


def test_lightning_cpu_fit_validate_and_predict(tmp_path: Path) -> None:
    model, data = make_problem()
    original_weights = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        max_epochs=3,
        num_sanity_val_steps=0,
        inference_mode=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(model, datamodule=data)
    training_loss = trainer.callback_metrics["train/loss"].detach().clone()
    metrics = trainer.validate(model, datamodule=data, verbose=False)
    predictions = trainer.predict(model, datamodule=data)

    assert trainer.global_step == 3
    assert model.device.type == "cpu"
    assert torch.isfinite(training_loss)
    assert math.isfinite(metrics[0]["val/loss"])
    assert math.isfinite(metrics[0]["val/error_temperature"])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_weights, model.parameters())
    )
    assert predictions is not None and len(predictions) == 1
    temperature = predictions[0]["temperature"]
    assert temperature.shape == (24, 1)
    assert torch.isfinite(temperature).all()


def test_prediction_applies_output_transform_without_pde(tmp_path: Path) -> None:
    mesh = PointCloud(read_data_fn=lambda: create_interval_data([0, 1], [0, 1], 4, 3))
    prediction = MeshSampler(mesh, use_lhs=False)
    data = PINNDataModule(train_datasets=[], val_dataset=None, pred_dataset=prediction)
    pde = Mock(side_effect=AssertionError("Inference must not evaluate PDE"))

    def output_transform(
        outputs: dict[str, torch.Tensor], x: torch.Tensor, time: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs["derivative"] = gradient(outputs["potential"], x)[0]
        return outputs

    model = PINNModule(
        net=FCN([2, 8, 1], mesh.lb, mesh.ub, ["potential"]),
        pde_fn=pde,
        output_fn=output_transform,
        amp=False,
        jit_compile=False,
    )
    spatial, time, _ = prediction[:]
    expected = model([spatial[0].requires_grad_()], time.requires_grad_())
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        inference_mode=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    predictions = trainer.predict(model, datamodule=data)

    pde.assert_not_called()
    assert predictions is not None
    torch.testing.assert_close(predictions[0]["derivative"], expected["derivative"])
