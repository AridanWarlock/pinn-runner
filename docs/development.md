# Development checks

Use Python 3.10, matching the pinned training dependencies:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-test-integration.txt
.venv/bin/python scripts/check.py
```

The script stops on the first failing check. It runs Ruff lint, Ruff formatting
verification, mypy, and the maintained CPU tests. It does not modify files.
To apply formatting explicitly, run:

```sh
.venv/bin/python -m ruff format run.py pinnstorch scripts tests/runner tests/numerics tests/integration
```

Ruff owns Python formatting and import ordering; the line length is 99.
The optional pre-commit hooks use the same tools from your activated environment.
Install pre-commit separately, activate `.venv`, then run `pre-commit install`.

## Type checking scope

The exact nine files checked with strict mypy are listed in `pyproject.toml`.
These cover the entry point, shared contracts, coordinate and time data, loader,
networks, numerical helpers, parameter callback adapters, and logging. Other modules are being migrated
gradually. Missing third-party stubs are allowed; this is not a claim that the
whole Lightning/CUDA stack is statically checked.

`contracts.py` documents common batch and callback types. Tensor annotations
describe Python objects, not their dimensions: runtime boundary validation and
numerical tests remain necessary. `cast` informs the checker and performs no
runtime conversion or validation. `TYPE_CHECKING` imports are available to the
checker without loading the training stack merely to resolve annotations.

## Test scope

Default pytest collection covers `tests/runner`, `tests/numerics`, and
`tests/integration`. These use local data and CPU tensors. The integration tests
run real Lightning and the file-based runner, including checkpoint prediction.
They do not establish convergence on production problems or CUDA correctness.

Old upstream tests remain in `tests/legacy` as migration references. They are
excluded explicitly, not counted as passing. See its README for context.
