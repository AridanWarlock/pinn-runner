"""Load tensor state and metadata portably between the training and analysis hosts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from . import _path_pickle


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(
        path, map_location=map_location, pickle_module=_path_pickle, weights_only=False
    )
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise ValueError("Checkpoint must contain a string-keyed state mapping")
    return cast(dict[str, Any], state)
