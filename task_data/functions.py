from typing import Any, Dict, List, Optional, Tuple

import torch

import pinnstorch

def read_data_fn(data: Dict):
    x = data["x"] # [N x 1]
    t = data["t"] # [T x 1]
    T = data["T"] # [N x T]

    # Проверка, что это двумерные массивы
    assert len(x.shape) == 2, f"x must be 2D, got shape {x.shape}"
    assert len(t.shape) == 2, f"t must be 2D, got shape {t.shape}"
    assert len(T.shape) == 2, f"T must be 2D, got shape {T.shape}"
    
    # Проверка, что второй размер = 1 для x и t
    assert x.shape[1] == 1, f"x must be (N, 1), got {x.shape}"
    assert t.shape[1] == 1, f"t must be (T, 1), got {t.shape}"

    return {
        "spatial": [x],
        "time": [t], 
        "solution": {"T": T},
    }


def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    """Define the partial differential equations (PDEs).
    """

    T = outputs["T"]
    T_x, T_t = pinnstorch.utils.gradient(T, [x, t])
    T_xx = pinnstorch.utils.gradient(T_x, x)[0]

    alpha = 0.1     

    outputs["f_T"] = T_t - alpha * T_xx
    return outputs

LEFT_BC_KEY = 'left_bc'
RIGHT_BC_KEY = 'right_bc'

def bc_left(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    
    T = outputs["T"]

    outputs[LEFT_BC_KEY] = T - 100

    return outputs

def bc_right(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    
    T = outputs["T"]

    # T = 100
    outputs[RIGHT_BC_KEY] = pinnstorch.utils.gradient(T, x)[0]

    return outputs


boundary_functions = {
    LEFT_BC_KEY: bc_left,
    RIGHT_BC_KEY: bc_right,
}