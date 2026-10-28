"""Fixed mixed-boundary experiment, using the established runner's physical fields."""

from __future__ import annotations

import copy
import hashlib
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from scipy.stats import qmc

from pinnstorch.api import Coordinates, Equation
from pinnstorch.data import BoundaryConditions, InitialCondition, MeshSampler, PointCloud
from pinnstorch.data.sampler.sampler_base import SamplerBase
from pinnstorch.models.pinn_module import PINNModule
from pinnstorch.runner.factory import SolverFactory
from pinnstorch.runner.problem import ProblemAdapter, load_problem
from pinnstorch.runner.settings import TaskSettings
from pinnstorch.training.refinement import FixedObjective

from .fingerprints import tensor_fingerprint

SOURCE = Path(__file__).parent / "problems/heat_soft_hard_talk/functions.py"
FACES = ("x_lower", "x_upper", "y_lower", "y_upper")


def config(method: str, seed: int, device: str) -> dict[str, Any]:
    return dict(
        version=1,
        domain=dict(
            space={n: dict(interval=[0, 1], points=16) for n in ("x", "y")},
            time=dict(interval=[0, 0.5], points=9),
        ),
        network=dict(hidden_layers=[32, 32, 32]),
        training=dict(
            steps=1,
            epochs=1,
            seed=seed,
            batch_size=1024,
            initial_points=256,
            optimizer=dict(name="adam", learning_rate=0.001, weight_decay=0),
            collocation=dict(sampling="lhs", points=1024),
            constraints="soft" if method == "soft" else "hard_initial_robin",
            hard_execution="eager",
            hard_lifting="recursive",
            hard_background="general",
            backward="all",
        ),
        boundaries=[
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
        ],
        validation=False,
        runtime=dict(device=device, precision="32-true", progress=False, metrics=False),
    )


def make_model(
    method: str, seed: int, device: str, dtype: torch.dtype = torch.float32
) -> PINNModule:
    settings = TaskSettings.read(config(method, seed, device))
    factory = SolverFactory(settings, load_problem(SOURCE), SOURCE.parent, SOURCE.parent)
    torch.manual_seed(seed)
    model = factory._model(PointCloud(lambda: settings.domain.grid()))
    return model.to(device=torch.device(device), dtype=dtype)


def samples(seed: int) -> dict[str, np.ndarray[Any, Any]]:
    pde = qmc.LatinHypercube(d=3, seed=seed + 10000).random(1024).astype(np.float32)
    pde[:, 2] *= 0.5
    initial = qmc.LatinHypercube(d=2, seed=seed + 20000).random(256).astype(np.float32)
    result = {"pde": pde, "ic": np.column_stack((initial, np.zeros(256, dtype=np.float32)))}
    for index, face in enumerate(FACES):
        z = qmc.LatinHypercube(d=2, seed=seed + 30000 + index).random(128).astype(np.float32)
        axis = 0 if face.startswith("x") else 1
        pts = np.empty((128, 3), dtype=np.float32)
        pts[:, axis] = 0 if face.endswith("lower") else 1
        pts[:, 1 - axis] = z[:, 0]
        pts[:, 2] = z[:, 1] * 0.5
        result[face] = pts
    return result


class Objective:
    """Штатные samplers; отдельный объект каждой грани сохраняет среднее по четырём граням."""

    def __init__(
        self, model: PINNModule, method: str, points: dict[str, np.ndarray[Any, Any]], device: str
    ) -> None:
        self.model = model
        settings = TaskSettings.read(config(method, 19, device))
        adapter = ProblemAdapter(load_problem(SOURCE), settings.domain)
        mesh = PointCloud(lambda: settings.domain.grid())
        equation = MeshSampler(mesh, collection_points=["heat"], use_lhs=False)
        pde = points["pde"]
        equation.store(pde[:, :2], pde[:, 2:], {})
        samplers: dict[str, SamplerBase] = {"equations": equation}
        if method == "soft":
            initial = InitialCondition(mesh, solution=["u"], initial_fun=adapter.initial)
            ic = points["ic"]
            initial.store(ic[:, :2], ic[:, 2:], adapter.initial(ic[:, :2]))
            samplers["initial"] = initial
            for index, face in enumerate(FACES):
                sampler = BoundaryConditions(
                    mesh,
                    [settings.boundaries[index].compile()],
                    {
                        name: partial(adapter.boundary_value, name)
                        for name in adapter.problem.boundary_values
                    },
                )
                pts = points[face]
                target = np.zeros((len(pts), 1), dtype=np.float32)
                if face == "x_upper":
                    columns = [torch.as_tensor(pts[:, i : i + 1]) for i in range(3)]
                    target = (
                        adapter.boundary_value("right", columns[:2], columns[2]).detach().numpy()
                    )
                sampler.store(
                    pts[:, :2],
                    pts[:, 2:],
                    {
                        "target_0": target,
                        "weight_0": np.full((len(pts), 1), 0.25, dtype=np.float32),
                    },
                )
                samplers[face] = sampler
        model.function_mapping = {name: sampler.loss_fn for name, sampler in samplers.items()}
        self.fixed = FixedObjective(
            {
                name: (
                    [v.to(device) for v in sampler[:][0]],
                    sampler[:][1].to(device),
                    {k: v.to(device) for k, v in sampler[:][2].items()},
                )
                for name, sampler in samplers.items()
            }
        )

    def loss(self) -> torch.Tensor:
        return self.model.model_step(self.fixed.fresh())[0]


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def train(
    model: PINNModule,
    method: str,
    seed: int,
    points: dict[str, np.ndarray[Any, Any]],
    output: Path,
    device: str,
    settings: dict[str, Any],
    progress: Any = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "functions.py").write_bytes(SOURCE.read_bytes())
    (output / "config.yaml").write_text(
        yaml.safe_dump(config(method, seed, device), sort_keys=False)
    )
    np.savez_compressed(output / "samples.npz", **points)
    objective = Objective(model, method, points, device)
    initial = tensor_fingerprint(dict(model.named_parameters()))
    pde_sha = hashlib.sha256(points["pde"].tobytes()).hexdigest()
    sample_sha = hashlib.sha256(b"".join(points[k].tobytes() for k in sorted(points))).hexdigest()
    adam = torch.optim.Adam(
        model.parameters(),
        lr=settings["adam_lr"],
        weight_decay=0,
        betas=(settings["adam_betas"][0], settings["adam_betas"][1]),
        eps=settings["adam_eps"],
    )
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=settings["lbfgs_lr"],
        history_size=settings["history_size"],
        max_iter=1,
        max_eval=1,
        tolerance_grad=settings["tolerance_grad"],
        tolerance_change=settings["tolerance_change"],
        line_search_fn="strong_wolfe",
    )
    snapshots: list[dict[str, Any]] = []
    updates = 0
    steps = 0
    closures = 0
    targets = list(settings["snapshot_seconds"])
    synchronize(device)
    started = time.perf_counter()

    def save(stage: str, force: bool = False) -> None:
        synchronize(device)
        elapsed = time.perf_counter() - started
        due = [v for v in targets if v <= elapsed]
        if not due and not force:
            return
        for target in due:
            targets.remove(target)
        name = f"checkpoint-{len(snapshots):02d}.ckpt"
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite accepted weights")
        checkpoint = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": version("pytorch-lightning"),
            "stage": stage,
            "seconds": elapsed,
            "adam_updates": updates,
            "lbfgs_steps": steps,
            "closures": closures,
            "optimizer": copy.deepcopy(
                adam.state_dict()
                if stage in ("initial", "adam", "adam_end")
                else lbfgs.state_dict()
            ),
        }
        torch.save(checkpoint, output / name)
        row = dict(
            path=name,
            seconds=elapsed,
            stage=stage,
            nominal_seconds=due,
            accepted=True,
            adam_updates=updates,
            lbfgs_steps=steps,
            closures=closures,
        )
        snapshots.append(row)
        if progress:
            progress(
                dict(
                    seed=seed,
                    method=method,
                    stage=stage,
                    seconds=elapsed,
                    adam_updates=updates,
                    lbfgs_steps=steps,
                    closures=closures,
                )
            )

    save("initial", True)
    while time.perf_counter() - started < settings["adam_seconds"]:
        adam.zero_grad(set_to_none=True)
        model.backward(objective.loss())
        adam.step()
        updates += 1
        synchronize(device)
        if time.perf_counter() - started < settings["adam_seconds"]:
            save("adam")
    save("adam_end", True)
    adam_end = snapshots[-1]["seconds"]

    def closure() -> torch.Tensor:
        nonlocal closures
        lbfgs.zero_grad(set_to_none=True)
        value = objective.loss()
        model.backward(value)
        closures += 1
        return value.detach()

    while time.perf_counter() - started < settings["total_seconds"]:
        lbfgs.step(closure)
        steps += 1
        synchronize(device)
        if time.perf_counter() - started < settings["total_seconds"]:
            save("lbfgs")
    save("final", True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    internal_state: dict[str, Any] = next(iter(lbfgs.state.values()), {})
    return dict(
        seed=seed,
        method=method,
        snapshots=snapshots,
        seconds=elapsed,
        adam_end_seconds=adam_end,
        overshoot_seconds=max(0.0, elapsed - settings["total_seconds"]),
        adam_updates=updates,
        lbfgs_steps=steps,
        closures=closures,
        lbfgs_internal_iterations=int(internal_state.get("n_iter", 0)),
        initial_parameters_sha256=initial,
        pde_sha256=pde_sha,
        samples_sha256=sample_sha,
    )


def field(
    model: PINNModule, points: np.ndarray[Any, Any], device: str, derivative: str | None = None
) -> np.ndarray[Any, Any]:
    out = []
    for start in range(0, len(points), 2048):
        batch = torch.tensor(points[start : start + 2048], device=device, dtype=torch.float32)
        x, y, t = [
            batch[:, i : i + 1].detach().requires_grad_(derivative is not None) for i in range(3)
        ]
        if derivative is None:
            with torch.no_grad():
                value = model([x, y], t)["u"]
        else:
            value = model([x, y], t)["u"]
            eq = Equation({"u": value}, Coordinates({"x": x, "y": y}, t))
            if derivative == "pde":
                value = eq.dt("u") - 0.1 * eq.laplacian("u")
            elif derivative == "robin":
                value = (
                    value
                    + eq.diff("u", "x")
                    + torch.exp(-0.2 * torch.pi**2 * t) * torch.cos(torch.pi * y)
                )
            else:
                value = eq.diff("u", derivative)
        out.append(value.detach().cpu().numpy().reshape(-1))
    return cast(np.ndarray[Any, Any], np.concatenate(out))


def exact(points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    x, y, t = points.T
    return cast(
        np.ndarray[Any, Any], np.exp(-0.2 * np.pi**2 * t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    )


def trapezoid(n: int) -> np.ndarray[Any, Any]:
    w = np.ones(n)
    w[[0, -1]] = 0.5
    return cast(np.ndarray[Any, Any], w / w.sum())


def evaluate(model: PINNModule, device: str, output: Path, retain: bool = False) -> dict[str, Any]:
    x = np.linspace(0, 1, 81)
    t = np.linspace(0, 0.5, 41)
    grid = np.stack(np.meshgrid(x, x, t, indexing="ij"), axis=-1).reshape(-1, 3)
    actual = field(model, grid, device).reshape(81, 81, 41).astype(np.float64)
    reference = exact(grid).reshape(actual.shape)
    error = actual - reference
    ws = trapezoid(81)[:, None] * trapezoid(81)[None, :]
    wt = trapezoid(41)
    mse = (ws[:, :, None] * error**2).sum(axis=(0, 1))
    norm = (ws[:, :, None] * reference**2).sum(axis=(0, 1))
    temporal = np.sqrt(mse / norm)
    probe = qmc.LatinHypercube(d=3, seed=20261028).random(4096)
    probe[:, 2] *= 0.5
    residual = field(model, probe, device, "pde")
    boundaries = {}
    edgegrid = np.stack(
        np.meshgrid(np.linspace(0, 1, 161), np.linspace(0, 0.5, 81), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    for face in FACES:
        pts = np.zeros((len(edgegrid), 3))
        axis = 0 if face.startswith("x") else 1
        pts[:, axis] = 0 if face.endswith("lower") else 1
        pts[:, 1 - axis] = edgegrid[:, 0]
        pts[:, 2] = edgegrid[:, 1]
        defect = field(
            model, pts, device, "robin" if face == "x_upper" else ("x" if axis == 0 else "y")
        )
        boundaries[face] = dict(
            rmse=float(np.sqrt(np.mean(defect**2))), max=float(np.max(np.abs(defect)))
        )
    ic_error = error[:, :, 0]
    if retain:
        np.savez_compressed(
            output / "final-fields.npz",
            x=x,
            t=t,
            prediction=actual.astype(np.float32),
            reference=reference.astype(np.float32),
            error=error.astype(np.float32),
        )
        cuts = []
        for instant in (0.1, 0.25, 0.5):
            pts = np.column_stack((x, np.zeros_like(x), np.full_like(x, instant)))
            cuts.append(field(model, pts, device))
        np.savez_compressed(
            output / "sections.npz",
            x=x,
            times=np.array([0.1, 0.25, 0.5]),
            prediction=np.stack(cuts),
        )
    return dict(
        relative_l2=float(np.sqrt(np.sum(wt * mse) / np.sum(wt * norm))),
        final_relative_l2=float(temporal[-1]),
        worst_time_relative_l2=float(temporal.max()),
        worst_time=float(t[temporal.argmax()]),
        rmse=float(np.sqrt(np.sum(wt * mse))),
        max_error=float(np.max(np.abs(error))),
        pde_rmse=float(np.sqrt(np.mean(residual**2))),
        pde_max=float(np.max(np.abs(residual))),
        ic_rmse=float(np.sqrt(np.sum(ws * ic_error**2))),
        ic_max=float(np.max(np.abs(ic_error))),
        boundaries=boundaries,
        temporal=[
            dict(t=float(a), relative_l2=float(b), rmse=float(c))
            for a, b, c in zip(t, temporal, np.sqrt(mse))
        ],
    )


def costs(
    model: PINNModule,
    method: str,
    points: dict[str, np.ndarray[Any, Any]],
    checkpoint: dict[str, Any],
    device: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    model.load_state_dict(checkpoint["state_dict"])
    objective = Objective(model, method, points, device)
    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    adam.load_state_dict(checkpoint["optimizer"])
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1,
        max_iter=1,
        max_eval=1,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    result = {}
    for kind in ("loss_gradient", "adam_update", "lbfgs_closure"):
        times = []
        for repeat in range(
            settings["cost_warmups"] + settings["cost_blocks"] * settings["cost_repeats"]
        ):
            model.load_state_dict(checkpoint["state_dict"])
            if kind == "adam_update":
                adam.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
            synchronize(device)
            started = time.perf_counter()
            if kind == "adam_update":
                adam.zero_grad(set_to_none=True)
            elif kind == "lbfgs_closure":
                lbfgs.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
            value = objective.loss()
            model.backward(value)
            if kind == "adam_update":
                adam.step()
            elif kind == "lbfgs_closure":
                value = value.detach()
            synchronize(device)
            seconds = time.perf_counter() - started
            if repeat >= settings["cost_warmups"]:
                times.append(seconds)
        blocks = [
            float(np.median(times[i : i + settings["cost_repeats"]]))
            for i in range(0, len(times), settings["cost_repeats"])
        ]
        result[kind] = dict(
            median_ms=float(np.median(blocks) * 1000),
            min_ms=float(np.min(times) * 1000),
            max_ms=float(np.max(times) * 1000),
            block_medians_ms=[v * 1000 for v in blocks],
            repeats=len(times),
            warmups=settings["cost_warmups"],
            state="adam_end",
            raw_seconds=times,
        )
    return result
