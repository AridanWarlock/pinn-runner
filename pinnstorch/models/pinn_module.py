"""Lightning glue for vectorized PINN losses; execution optimizations belong outside it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics import MeanMetric, MinMetric

if TYPE_CHECKING:
    from pinnstorch.data.pinn_datamodule import PINNDataModule

from pinnstorch.contracts import (
    FieldFunction,
    OptimizerFactory,
    SamplerBatch,
    SamplerFunctions,
    SamplerLoss,
    SchedulerFactory,
    TensorOutputs,
)
from pinnstorch.training.loss import EqualWeights, FixedWeights, LossPolicy, gradient_norms
from pinnstorch.training.optimization import ConstantRate, ExponentialRate, RatePolicy
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
        loss_settings: dict[str, str | float | int] | None = None,
        rate_settings: dict[str, str | float | int] | None = None,
        constraint_kind: str = "soft",
        constraint_signature: str = "",
        refinement_settings: dict[str, int | float] | None = None,
        hard_execution: str = "eager",
        hard_lifting: str = "recursive",
        backward_mode: str = "all",
        hard_background: str = "general",
        spatial_encoding: str = "identity",
        input_transform: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if loss_fn not in ("mse", "sse"):
            raise ValueError("loss_fn must be mse or sse")
        if backward_mode not in ("all", "parameters"):
            raise ValueError("backward_mode must be all or parameters")
        self.save_hyperparameters(
            logger=False, ignore=["net", "pde_fn", "output_fn", "coefficients"]
        )
        self.net, self.pde_fn, self.output_fn = net, pde_fn, output_fn
        self.coefficients = coefficients
        recipe = loss_settings or {"kind": "equal"}
        self.loss_policy: LossPolicy = (
            FixedWeights(
                {
                    name: float(recipe.get(name, 1.0))
                    for name in ("initial", "equations", "boundaries")
                }
            )
            if recipe["kind"] == "fixed"
            else EqualWeights()
        )
        self.gradient_norm_every = int(recipe.get("gradient_norm_every", 0))
        self.loss_terms: dict[str, torch.Tensor] = {}
        self.refinement_state: dict[str, Any] | None = None
        self.adam_evaluations = 0
        self.adam_residual_points = 0
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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.refinement_state is not None:
            checkpoint["refinement_state"] = self.refinement_state

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Keep physical AD intact; restrict only terminal leaf accumulation."""
        if (
            self.hparams["backward_mode"] == "parameters"
            and "inputs" not in kwargs
            and len(args) < 4  # Tensor.backward also accepts inputs as its fourth argument.
        ):
            parameters = tuple(p for p in self.parameters() if p.requires_grad)
            if not parameters:
                raise ValueError("Parameter backward requires trainable parameters")
            kwargs["inputs"] = parameters
        # Preserve Lightning's backward hook and its normal optimizer lifecycle.
        super().backward(loss, *args, **kwargs)

    def _data_module(self) -> PINNDataModule:
        return cast("PINNDataModule", getattr(self.trainer, "datamodule"))

    def on_fit_start(self) -> None:
        self.function_mapping = self._data_module().function_mapping

    def model_step(self, batch: dict[str, SamplerBatch]) -> tuple[torch.Tensor, TensorOutputs]:
        terms: dict[str, torch.Tensor] = {}
        outputs: TensorOutputs = {}
        for name, inputs in batch.items():
            spatial, time, _ = inputs
            set_requires_grad(spatial, time, True)
            loss, outputs = self.function_mapping[name](inputs, 0.0, self.functions)
            terms[name] = torch.as_tensor(loss, device=self.device)
        self.loss_terms = terms
        return self.loss_policy.combine(terms), outputs

    def training_step(self, batch: dict[str, SamplerBatch], batch_idx: int) -> torch.Tensor:
        self.adam_evaluations += 1
        if "equations" in batch:
            self.adam_residual_points += len(batch["equations"][1])
        loss, _ = self.model_step(batch)
        for name, value in self.loss_terms.items():
            self.log(f"train/terms/{name}", value.detach(), on_step=True, on_epoch=False)
        if self.gradient_norm_every and (self.global_step + 1) % self.gradient_norm_every == 0:
            for name, value in gradient_norms(self.loss_terms, list(self.parameters())).items():
                self.log(f"train/gradient_norms/{name}", value, on_step=True, on_epoch=True)
        self.log(
            "train/learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
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
        coordinate_grad = self.output_fn is not None and bool(
            getattr(self.output_fn, "requires_coordinate_grad", True)
        )
        with torch.set_grad_enabled(coordinate_grad):
            spatial, time = set_requires_grad(spatial, time, coordinate_grad)
            return self.forward(spatial, time)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        factory = cast(OptimizerFactory, self.hparams["optimizer"])
        optimizer = factory(params=self.parameters())
        recipe = self.hparams.get("rate_settings")
        if recipe is not None:
            policy: RatePolicy = (
                ExponentialRate(int(recipe["steps"]), float(recipe["final_ratio"]))
                if recipe["kind"] == "exponential"
                else ConstantRate()
            )
            built = policy.build(optimizer)
            if built is None:
                return {"optimizer": optimizer}
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": built, "interval": "step"},
            }
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
