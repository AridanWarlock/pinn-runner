"""CPU-контроль постановки и существующего hard-адаптера; без оптимизатора."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from pinnstorch.api import Coordinates, Equation
from pinnstorch.data import PointCloud
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import load_problem
from pinnstorch.runner.settings import TaskSettings

HERE = Path(__file__).parent


def config(kind):
    faces = [
        dict(kind="neumann", field="u", axis="x", side="lower", value=0, weight=0.25),
        dict(
            kind="robin",
            field="u",
            axis="x",
            side="upper",
            alpha=1,
            beta=1,
            value_function="right",
            weight=0.25,
        ),
        dict(kind="neumann", field="u", axis="y", side="lower", value=0, weight=0.25),
        dict(kind="neumann", field="u", axis="y", side="upper", value=0, weight=0.25),
    ]
    return dict(
        version=1,
        domain=dict(
            space={n: dict(interval=[0, 1], points=9) for n in ["x", "y"]},
            time=dict(interval=[0, 0.5], points=5),
        ),
        network=dict(hidden_layers=[32, 32, 32]),
        training=dict(
            steps=1,
            epochs=1,
            seed=19,
            batch_size=1024,
            initial_points=256,
            optimizer=dict(name="adam", learning_rate=0.001),
            collocation=dict(sampling="lhs", points=1024),
            constraints=kind,
            hard_execution="eager",
            hard_lifting="recursive",
            hard_background="general",
            backward="all",
        ),
        boundaries=faces,
        validation=False,
        runtime=dict(device="cpu", precision="32-true", progress=False, metrics=False),
    )


def norm_boundary(value, x, y, t, face):
    e = Equation({"u": value}, Coordinates({"x": x, "y": y}, t))
    if face == "x_lower":
        return -e.diff("u", "x")
    if face == "x_upper":
        return (
            value + e.diff("u", "x") + torch.exp(-0.2 * torch.pi**2 * t) * torch.cos(torch.pi * y)
        )
    return (-1 if face == "y_lower" else 1) * e.diff("u", "y")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    rng = np.random.default_rng(20261028)
    queries = rng.uniform(size=(23, 3))
    queries[:, 2] *= 0.5

    def leaves():
        return [
            torch.tensor(queries[:, i : i + 1], dtype=torch.float64, requires_grad=True)
            for i in range(3)
        ]

    problem = load_problem(HERE / "functions.py")
    models = {}
    for kind in ("soft", "hard_initial_robin"):
        cfg = config(kind)
        settings = TaskSettings.read(cfg)
        factory = SolverFactory(settings, problem, HERE, HERE)
        torch.manual_seed(19)
        models[kind] = factory._model(PointCloud(lambda: settings.domain.grid())).double()
        (HERE / f"{kind}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for (_, a), (_, b) in zip(
        models["soft"].named_parameters(), models["hard_initial_robin"].named_parameters()
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    model = models["hard_initial_robin"]
    x, y, t = leaves()
    exact = torch.exp(-0.2 * torch.pi**2 * t) * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
    e = Equation({"u": exact}, Coordinates({"x": x, "y": y}, t))
    pde = float((e.dt("u") - 0.1 * e.laplacian("u")).abs().max().detach())
    assert pde < 1e-12
    defects = {}
    reference_defects = {}
    for face in ["x_lower", "x_upper", "y_lower", "y_upper"]:
        xx, yy, tt = leaves()
        axis = 0 if face.startswith("x") else 1
        edge = torch.full_like(xx, 0 if face.endswith("lower") else 1, requires_grad=True)
        if axis == 0:
            xx = edge
        else:
            yy = edge
        val = model([xx, yy], tt)["u"]
        defects[face] = float(norm_boundary(val, xx, yy, tt, face).abs().max().detach())
        reference = (
            torch.exp(-0.2 * torch.pi**2 * tt)
            * torch.cos(torch.pi * xx)
            * torch.cos(torch.pi * yy)
        )
        reference_defects[face] = float(
            norm_boundary(reference, xx, yy, tt, face).abs().max().detach()
        )
        assert reference_defects[face] < 1e-12
        assert defects[face] < 1e-11, defects
    ic = model([x, y], torch.zeros_like(t))["u"]
    icmax = float((ic - torch.cos(torch.pi * x) * torch.cos(torch.pi * y)).abs().max().detach())
    assert icmax < 1e-12
    # Полная физическая PDE и все весовые градиенты, без шагов оптимизатора.
    value = model([x, y], t)["u"]
    e = Equation({"u": value}, Coordinates({"x": x, "y": y}, t))
    loss = (e.dt("u") - 0.1 * e.laplacian("u")).square().mean()
    params = tuple(model.parameters())
    grad = torch.cat([g.flatten() for g in torch.autograd.grad(loss, params)])
    assert torch.isfinite(grad).all() and grad.norm() > 0
    before = {k: v.clone() for k, v in model.state_dict().items()}
    direction = torch.tensor(rng.normal(size=len(grad)), dtype=torch.float64)
    direction /= direction.norm()
    h = 1e-5
    values = []
    for sign in (1, -1):
        model.load_state_dict(before)
        with torch.no_grad():
            offset = 0
            for p in params:
                n = p.numel()
                p.add_(direction[offset : offset + n].reshape_as(p), alpha=sign * h)
                offset += n
        xx, yy, tt = leaves()
        v = model([xx, yy], tt)["u"]
        eq = Equation({"u": v}, Coordinates({"x": xx, "y": yy}, tt))
        values.append(float((eq.dt("u") - 0.1 * eq.laplacian("u")).square().mean().detach()))
    fd = (values[0] - values[1]) / (2 * h)
    ad = float(grad @ direction)
    np.testing.assert_allclose(fd, ad, rtol=1e-5, atol=1e-8)
    model.load_state_dict(before)
    # Совместимость Robin/Neumann на правых углах: касательная нормальная производная RHS нулевая.
    corners = []
    for end in (0.0, 1.0):
        yy = torch.full_like(y, end, requires_grad=True)
        rhs = -torch.exp(-0.2 * torch.pi**2 * t) * torch.cos(torch.pi * yy)
        dy = torch.autograd.grad(rhs.sum(), yy)[0]
        corners.append(float(dy.abs().max()))
    assert max(corners) < 1e-12
    result = dict(
        kind="cpu_preflight_no_training",
        seed=19,
        reference_boundary_max=reference_defects,
        initial_parameters_sha256=hashlib.sha256(
            torch.cat([p.detach().flatten() for p in models["soft"].parameters()])
            .numpy()
            .tobytes()
        ).hexdigest(),
        pde_max=pde,
        hard_ic_max=icmax,
        hard_boundary_max=defects,
        corner_compatibility_max=max(corners),
        parameters=grad.numel(),
        directional_ad=ad,
        directional_fd=fd,
        same_trainable_initial_parameters=True,
        optimizer_updates=0,
        source_sha256=hashlib.sha256((HERE / "functions.py").read_bytes()).hexdigest(),
    )
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
