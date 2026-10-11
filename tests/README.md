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

The numerical tests load the actual derivative helper and FCN directly because importing
`pinnstorch` also imports the training stack. They verify analytic derivatives,
backpropagation through second derivatives, multiple output components, and the
heat equation residual in one, two, and three spatial dimensions. They also check
FCN outputs and derivatives in these dimensions, and ten Adam steps with PDE and
initial-condition losses on CPU. This short optimization checks finite gradients,
weight updates, and loss reduction. These tests do not
verify model convergence or the complete training pipeline.

## Lightning integration checks

Install the additional dependencies in the same Python 3.10 environment:

```sh
.venv/bin/python -m pip install -r requirements-test-integration.txt
.venv/bin/python -m pytest --noconftest -o addopts= tests/runner tests/numerics tests/integration
.venv/bin/ruff check tests/integration
```

These tests import the actual package and use its PointCloud, samplers, FCN,
PINNModule, and PINNDataModule with a real Lightning Trainer. They check loss
accumulation in both condition orders, three CPU training steps, validation, and
prediction. Test data are generated locally; no datasets are downloaded.
CUDA, compilation, and mixed precision are disabled. The container entry point
and YAML composition are not exercised by this suite yet.
