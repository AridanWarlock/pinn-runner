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
CUDA, compilation, and mixed precision are disabled.

The file-based tests also launch `run.py` as a separate process, compose its YAML
configuration, load `functions.py` and `data.mat`, and inspect predictions,
checkpoints, and metrics. A failing task must return exit code 1 and write an
error report. This checks the runner without building its Docker image.

To run a task locally with the integration dependencies installed:

```sh
.venv/bin/python run.py --mode train --input-dir ./task_data --output-dir ./task_output
```

Set `trainer.accelerator: cpu` and `trainer.devices: 1` in `config.yaml` for CPU
development. Directory arguments default to `/task_data` and `/task_output` for
the existing container workflow.
