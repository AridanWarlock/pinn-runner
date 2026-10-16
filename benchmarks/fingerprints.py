"""Content fingerprints for repeatable weights and ordered fixed objectives."""

import hashlib
import json
from collections.abc import Mapping

import torch

from pinnstorch.contracts import SamplerBatch


def tensor_fingerprint(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        metadata = (name, str(tensor.dtype), tuple(tensor.shape), value.numel())
        digest.update(json.dumps(metadata, ensure_ascii=True).encode() + b"\n")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def objective_fingerprint(batches: Mapping[str, SamplerBatch]) -> str:
    tensors: dict[str, torch.Tensor] = {}
    for name, (spatial, time, targets) in batches.items():
        for index, coordinate in enumerate(spatial):
            tensors[json.dumps((name, "spatial", index))] = coordinate
        tensors[json.dumps((name, "time"))] = time
        for field, target in targets.items():
            tensors[json.dumps((name, "target", field))] = target
    return tensor_fingerprint(tensors)
