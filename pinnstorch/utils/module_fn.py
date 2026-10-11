from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from pinnstorch.contracts import LossAccumulator


def sse(
    loss: LossAccumulator,
    preds: Dict[str, torch.Tensor],
    target: Union[Dict[str, torch.Tensor], None] = None,
    keys: Union[List[str], None] = None,
    mid: Union[int, None] = None,
) -> LossAccumulator:
    """Calculate the sum of squared errors (SSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated SSE loss.
    """

    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + torch.sum(torch.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + torch.sum(torch.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + torch.sum(torch.square(preds[key] - target[key]))

    return loss


def mse(
    loss: LossAccumulator,
    preds: Dict[str, torch.Tensor],
    target: Union[Dict[str, torch.Tensor], None] = None,
    keys: Union[List[str], None] = None,
    mid: Union[int, None] = None,
) -> LossAccumulator:
    """Calculate the mean squared error (MSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated MSE loss.
    """

    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + torch.mean(torch.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + torch.mean(torch.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + torch.mean(torch.square(preds[key] - target[key]))

    return loss


def relative_l2_error(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the relative L2 error between predictions and target tensors.

    :param preds: Predicted tensors.
    :param target: Target tensors.
    :return: Relative L2 error value.
    """
    l2_norm_error = torch.mean((preds - target) ** 2)
    l2_norm_target = torch.mean((target) ** 2)
    # print(l2_norm_error, l2_norm_target, relative_l2)
    if l2_norm_target == 0:
        return l2_norm_error
    return torch.sqrt(l2_norm_error / l2_norm_target)


def set_requires_grad(
    x: List[torch.Tensor], t: torch.Tensor, enable_grad: bool = True
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Set the requires_grad attribute for tensors in the input list.

    :param x: List of tensors to modify requires_grad attribute.
    :param t: Tensor to modify requires_grad attribute.
    :param enable_grad: Boolean indicating whether to enable requires_grad or not.
    :return: Modified list of tensors and tensor.
    """
    if t is not None:
        t = t.requires_grad_(enable_grad)
    x = [x_.requires_grad_(enable_grad) for x_ in x]

    return x, t


def fix_predictions(
    preds_list: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """It will concatenate predictions for each solution
    into a tensor and return a dictionary including them.

    :param preds_list: List of dictionary of predictions.
    :return: A dictionary containing the predictions.
    """
    chunks: dict[str, list[torch.Tensor]] = {}
    for predictions in preds_list:
        for name, tensor in predictions.items():
            chunks.setdefault(name, []).append(tensor.detach())
    return {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}
