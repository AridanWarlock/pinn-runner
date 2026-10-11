# Physics-only task inputs

The 1D/2D/3D space + time runner requires `config.yaml` and `functions.py`.
`retrain` and `predict` additionally require a checkpoint. No `data.mat` or
`read_data_fn` is needed; the runner does not read a reference solution.

## Coordinates in config.yaml

```yaml
domain:
  x_interval: [0.0, 1.0]
  t_interval: [0.0, 0.5]
  x_points: 8
  t_points: 3
```

Bounds must be finite and increasing. Each point count must be an integer >= 2.
The coordinate grid is also used for prediction and MAT output. PDE training
points can use the existing Latin hypercube sampling instead.

## Initial conditions in functions.py

Both `pde_fn` and `initial_conditions` must be callable. The latter receives a
NumPy array of spatial coordinates, shape `(N, spatial_dimension)`, and returns a dictionary
of finite `(N, 1)` arrays keyed by solution name. For example:

```python
import numpy as np
from numpy.typing import NDArray

def initial_conditions(spatial: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    return {"T": np.sin(np.pi * spatial[:, 0:1])}
```

These values apply at `t_interval[0]`. Include an `InitialCondition` dataset in
`train_datasets`, with `solution: [T]`; the runner connects the function to that
dataset. The complete runnable heat example includes zero-temperature boundary
conditions on both ends:

- [config.yaml](../tests/integration/fixtures/heat/config.yaml)
- [functions.py](../tests/integration/fixtures/heat/functions.py)

## Validation and prediction

Configure `val_dataset` with PDE `collection_points`, not reference `solution`
values. `val/loss` then measures PDE residual; no error against an exact solution
is calculated. A small residual alone does not establish full solution accuracy.

Configure `pred_dataset` as a `MeshSampler` with `use_lhs: false` and without
`solution` or `collection_points`. Prediction runs the network and any
`output_fn`, without evaluating PDE or initial conditions. The functions are
still required in the task file to keep a consistent task definition.
`predict` writes `predictions.mat`; training can save `predictions.pkl` with
`save_pred: true`. Reference comparison and plotting are separate future work.

## Rectangles and boxes

Add `y_interval` and `y_points` for 2D; also add `z_interval` and `z_points`
for 3D. Interval/count pairs must be supplied together, and z requires y.
The grid is a Cartesian product, so the spatial point count is the product
of the axis counts. Total prediction count also includes `t_points`.

For FCN, the first layer size is spatial dimension + 1 (time): 3 in 2D,
4 in 3D. PDE/output callbacks receive coordinates as separate tensors:
`pde_fn(outputs, x, y, t)` or `pde_fn(outputs, x, y, z, t)`.
Initial conditions receive all spatial columns in one NumPy array.

Use `pinnstorch.data.BoundaryCondition` for complete faces:

```yaml
- boundary_x:
    _target_: pinnstorch.data.BoundaryCondition
    bc_func_name: boundary
    axis: x
    location: both
```

`axis` is x/y/z; `location` is lower/upper/both. Add a dataset for each axis
whose faces need constraints. Boundary callbacks receive `(outputs, spatial, t)`:
`spatial` is a list of coordinate tensors. They must add a residual under the
configured `bc_func_name`, which is minimized against zero. This supports value
or derivative constraints. Intersections can occur in datasets for multiple
faces; each dataset contributes its own mean loss.

`BoundaryCondition1D` still uses left/right/both and rejects multi-axis domains.
The new face sampler supports continuous time, not the legacy discrete solver.

MAT output includes x/y/z columns for the flattened spatial grid and a time
column t. Each field has shape `(spatial_point_count, time_point_count)`.
Grid ordering is NumPy `meshgrid(indexing="ij")` flattened in C order: the
last spatial axis changes fastest; time changes fastest in network batches.
Coordinates are paired rows, not independent axis vectors.

A runnable CPU example is in [examples/heat2d](../examples/heat2d).

## Varying model parameters

A parameterized forward problem represents a family of solutions with one
network. Coefficients are supplied inputs; they are not unknown trainable
variables. For example:

```yaml
domain:
  x_interval: [0, 1]
  x_points: 8
  t_interval: [0, 0.5]
  t_points: 3
  parameters:
    conductivity:
      interval: [0.1, 0.5]
      points: 3
net:
  _target_: pinnstorch.models.FCN
  layers: [3, 16, 16, 1]  # x, conductivity, t
  output_names: [T]
```

Multiple parameters are supported. Their order follows the YAML mapping.
Intervals must be finite/increasing and counts >= 2. Names must be ASCII Python
identifiers without a leading underscore, distinct from x/y/z/t and output
names. Each parameter multiplies the Cartesian grid size. LHS PDE samples
include continuous parameter values inside the bounds, not just grid nodes.
Initial and boundary datasets cover the configured parameter grid.

Parameterized callbacks require a keyword argument `parameters`: a dictionary
of named `(N, 1)` tensors (NumPy arrays for initial conditions). PDE and output
callbacks receive only physical coordinates as positional arguments:

```python
def pde_fn(outputs, x, t, *, parameters):
    dx, dt = gradient(outputs["T"], [x, t])
    outputs["residual"] = dt - parameters["conductivity"] * gradient(dx, x)[0]
    return outputs
```

The initial signature is `initial_conditions(spatial, *, parameters)`, and
boundary callbacks use `boundary(outputs, spatial, t, *, parameters)`.
`spatial` retains only physical coordinates. Tasks without parameters retain
their existing signatures. Use the new face-based BoundaryCondition even for
parameterized 1D tasks; BoundaryCondition1D is not suitable for these grids.

Network input order is physical coordinates, parameters, time. The first FCN
layer must equal spatial dimension + parameter count + 1. Checkpoints record
input names and reject different names/order on prediction or retraining.
The saved normalization bounds are reused with the checkpoint. Prediction can
use a different grid inside the training range; successful execution does not
establish interpolation accuracy. Each parameter is assumed constant across
space and time within one problem instance; spatially varying coefficient
fields need a separate definition of their PDE derivatives.

MAT output includes one column per parameter, aligned with x/y/z rows. Field
arrays remain `(number_of_spatial_parameter_combinations, number_of_times)`.
Parameters vary fastest after the spatial coordinates in C-order flattening.
The `PointCloudData.spatial` storage includes both physical and parameter
columns internally; `spatial_dimension` and `parameter_names` separate their
roles. User callback adapters expose them separately.

This initial parameterized path uses continuous time without JIT compilation
or inverse extra_variables. A runnable typed example is in
[examples/heat_parameterized](../examples/heat_parameterized).
