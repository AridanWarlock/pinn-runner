"""Короткий файловый допуск модального поля; качество отдельно не зарегистрировано."""

from pathlib import Path
from typing import Any

from .quadratic_clock_case import clock_config

SOURCE = Path(__file__).parent / "problems/heat_quadratic_modal/functions.py"


def modal_config(device: str, steps: int, seed: int) -> dict[str, Any]:
    config = clock_config(device, steps, seed)
    config["network"].pop("time_scale")
    config["network"]["quadratic_modal"] = True
    return config
