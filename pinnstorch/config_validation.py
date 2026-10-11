"""Early checks for the public file runner, before constructing training objects."""

from collections.abc import Sequence

from omegaconf import DictConfig, ListConfig


def validate_task_settings(cfg: DictConfig, dimension: int, parameters: Sequence[str]) -> None:
    """Validate supported model dimensions and dataset configurations."""
    outputs = cfg.net.get("output_names")
    if not isinstance(outputs, (ListConfig, list)) or not outputs:
        raise ValueError("net.output_names must be a nonempty list")
    if any(
        not isinstance(name, str)
        or not name.isidentifier()
        or name.startswith("_")
        or not name.isascii()
        for name in outputs
    ):
        raise ValueError("Output names must be ASCII identifiers without leading underscores")
    if len(set(outputs)) != len(outputs) or set(outputs) & {"x", "y", "z", "t", *parameters}:
        raise ValueError(
            "Output names must be distinct from each other, coordinates and parameters"
        )
    if cfg.net._target_ == "pinnstorch.models.FCN":
        layers = cfg.net.get("layers")
        if not isinstance(layers, (ListConfig, list)) or len(layers) < 2:
            raise ValueError("net.layers must contain at least input and output sizes")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in layers
        ):
            raise ValueError("net.layers sizes must be positive integers")
        if layers[0] != dimension + len(parameters) + 1:
            raise ValueError("FCN input layer must match spatial dimensions + parameters + time")
        if layers[-1] != len(outputs):
            raise ValueError("FCN output layer must match net.output_names")
    if cfg.get("batch_size") is not None:
        raise ValueError("Set batch size under data.batch_size, not at the top level")
    batch = cfg.data.get("batch_size")
    if batch is not None and (isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0):
        raise ValueError("data.batch_size must be a positive integer or null")
    if cfg.trainer.get("inference_mode", False):
        raise ValueError("trainer.inference_mode must be false for PINN derivatives")
    for group in ("val_dataset", "pred_dataset"):
        datasets = cfg.get(group)
        if not datasets:
            if group == "pred_dataset":
                raise ValueError("pred_dataset must define one full-grid MeshSampler")
            continue
        if not isinstance(datasets, ListConfig) or len(datasets) != 1 or len(datasets[0]) != 1:
            raise ValueError(f"{group} must contain exactly one dataset")
        dataset = next(iter(datasets[0].values()))
        if dataset.get("_target_") != "pinnstorch.data.MeshSampler":
            raise ValueError(f"{group} must use pinnstorch.data.MeshSampler")
        if dataset.get("solution"):
            raise ValueError(f"{group} cannot request reference solution values")
        if group == "pred_dataset":
            if dataset.get("use_lhs", True) or dataset.get("collection_points"):
                raise ValueError("pred_dataset requires use_lhs: false without collection_points")
        elif not dataset.get("collection_points"):
            raise ValueError("val_dataset must specify PDE collection_points")
        elif dataset.get("use_lhs", True):
            count = dataset.get("num_sample")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("LHS validation requires a positive integer num_sample")
    for group in cfg.train_datasets:
        for dataset in group.values():
            if dataset.get("_target_") == "pinnstorch.data.MeshSampler" and dataset.get(
                "use_lhs", True
            ):
                count = dataset.get("num_sample")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError("LHS MeshSampler requires a positive integer num_sample")
    checkpoint = cfg.get("callbacks", {}).get("model_checkpoint")
    if checkpoint and checkpoint.get("monitor") == "val/loss" and not cfg.get("val_dataset"):
        raise ValueError("Monitoring val/loss requires val_dataset; otherwise monitor train/loss")
