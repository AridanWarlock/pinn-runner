## CPU checks on macOS

Use Python 3.10 and an isolated environment:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-test-cpu.txt
.venv/bin/python -m pytest --noconftest -o addopts= tests/runner tests/numerics
.venv/bin/ruff check run.py pinnstorch/utils/pylogger.py tests/runner tests/numerics
```

`--noconftest` bypasses the legacy training fixtures in `tests/conftest.py`.
`-o addopts=` disables project-wide doctest collection for this isolated suite.
These checks use CPU tensors and do not require CUDA or Lightning.

The numerical tests load the actual derivative helper directly because importing
`pinnstorch` also imports the training stack. They verify analytic derivatives,
backpropagation through second derivatives, multiple output components, and the
heat equation residual in one, two, and three spatial dimensions. They do not
verify model convergence or the complete training pipeline.
