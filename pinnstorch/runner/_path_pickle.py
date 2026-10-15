"""Pickle codec preserving foreign path metadata without native filesystem access."""

from __future__ import annotations

import os
import pickle
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO


class Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "pathlib":
            if name == "WindowsPath" and os.name != "nt":
                return PureWindowsPath
            if name == "PosixPath" and os.name == "nt":
                return PurePosixPath
        return super().find_class(module, name)


def load(file: BinaryIO, **kwargs: Any) -> Any:
    return Unpickler(file, **kwargs).load()
