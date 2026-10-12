"""Lightning glue for vectorized PINN losses; execution optimizations belong outside it."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics import MeanMetric, MinMetric

if TYPE_CHECKING:
    from pinnstorch.data.pinn_datamodule import PINNDataModule

from pinnstorch.contracts import (
    FieldFunction,
    LossAccumulator,
    OptimizerFactory,
    SamplerBatch,
    SamplerFunctions,
    SamplerLoss,
    SchedulerFactory,
    TensorOutputs,
)
from pinnstorch.utils.module_fn import mse, relative_l2_error, set_requires_grad, sse


class PINNModule(LightningModule):
    """Train, validate and predict with one shared field evaluation path."""

    def __init__(
        self,
        net: torch.nn.Module,
        pde_fn: FieldFunction,
        optimizer: OptimizerFactory = torch.optim.Adam,
        scheduler: SchedulerFactory | None = None,
        output_fn: FieldFunction | None = None,
        loss_fn: str = "mse",
        input_names: tuple[str, ...] = (),
        input_bounds: tuple[tuple[float, float], ...] = (),
        coefficients: torch.nn.Module | None = None,
        coefficient_signature: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if loss_fn not in ("mse", "sse"):
            raise ValueError("loss_fn must be mse or sse")
        self.save_hyperparameters(
            logger=False, ignore=["net", "pde_fn", "output_fn", "coefficients"]
        )
        self.net, self.pde_fn, self.output_fn = net, pde_fn, output_fn
        self.coefficients = coefficients
        self.function_mapping: dict[str, SamplerLoss] = {}
        self.train_loss, self.val_loss, self.val_loss_best = (
            MeanMetric(),
            MeanMetric(),
            MinMetric(),
        )
        self.functions: SamplerFunctions = {
            "forward": self.forward,
            "pde_fn": pde_fn,
            "loss_fn": mse if loss_fn == "mse" else sse,
        }

    def forward(self, spatial: list[torch.Tensor], time: torch.Tensor) -> TensorOutputs:
        outputs = cast(TensorOutputs, self.net(spatial, time))
        return self.output_fn(outputs, *spatial, time) if self.output_fn else outputs

    def _data_module(self) -> PINNDataModule:
        return cast("PINNDataModule", getattr(self.trainer, "datamodule"))

    def on_fit_start(self) -> None:
        self.function_mapping = self._data_module().function_mapping

    def model_step(self, batch: dict[str, SamplerBatch]) -> tuple[torch.Tensor, TensorOutputs]:
        loss: LossAccumulator = 0.0
        outputs: TensorOutputs = {}
        for name, inputs in batch.items():
            spatial, time, _ = inputs
            set_requires_grad(spatial, time, True)
            loss, outputs = self.function_mapping[name](inputs, loss, self.functions)
        return torch.as_tensor(loss, device=self.device), outputs

    def training_step(self, batch: dict[str, SamplerBatch], batch_idx: int) -> torch.Tensor:
        loss, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        return loss

    def eval_step(self, batch: SamplerBatch) -> tuple[torch.Tensor, TensorOutputs, TensorOutputs]:
        data = self._data_module()
        with torch.set_grad_enabled(True):
            spatial, time, targets = batch
            set_requires_grad(spatial, time, True)
            loss, outputs = data.loss_fn(batch, 0.0, self.functions)
        errors = {
            name: relative_l2_error(outputs[name], targets[name])
            for name in data.solution_names or []
        }
        return torch.as_tensor(loss, device=self.device), errors, outputs

    def validation_step(self, batch: SamplerBatch, batch_idx: int) -> None:
        loss, errors, _ = self.eval_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)
        for name, error in errors.items():
            self.log(f"val/error_{name}", error, batch_size=len(batch[1]))

    def on_validation_epoch_end(self) -> None:
        self.val_loss_best(self.val_loss.compute())
        self.log("val/loss_best", self.val_loss_best.compute())

    def predict_step(self, batch: SamplerBatch, batch_idx: int) -> TensorOutputs:
        spatial, time, _ = batch
        with torch.set_grad_enabled(self.output_fn is not None):
            spatial, time = set_requires_grad(spatial, time, self.output_fn is not None)
            return self.forward(spatial, time)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        factory = cast(OptimizerFactory, self.hparams["optimizer"])
        optimizer = factory(params=self.parameters())
        scheduler = cast(SchedulerFactory | None, self.hparams["scheduler"])
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler(optimizer=optimizer),
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }
