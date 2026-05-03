from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    data = pinnstorch.utils.load_data(root_path, "heat.mat")
    
    x = data["x"] # [N x 1]
    t = data["t"] # [T x 1]
    T = data["T"] # [N x T]

    return pinnstorch.data.PointCloudData(
            spatial=[x], time=[t], solution={"T": T}
    )


def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    """Define the partial differential equations (PDEs).
    """

    T = outputs["T"]
    T_x, T_t = pinnstorch.utils.gradient(T, [x, t])
    T_xx = pinnstorch.utils.gradient(T_x, x)[0]

    outputs["f_T"] = T_t - T_xx
    return outputs

def boundary_left_dc(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    

    outputs["left_bc"] = outputs["T"] - 100

    return outputs

def boundary_right_dc(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    

    outputs["right_bc"] = outputs["T"]

    return outputs


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstorch.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()