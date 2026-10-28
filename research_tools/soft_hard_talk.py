"""Fixed six-run soft/hard wall-time study; separate from convergence contract2."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .schema import Json, digest, inside, read_object

ORDER = [(19, "soft"), (19, "hard"), (73, "hard"), (73, "soft"), (1234, "soft"), (1234, "hard")]
SETTINGS: Json = {
    "study_wall_cap_seconds": 840,
    "pair_reserve_seconds": 420,
    "adam_seconds": 60,
    "total_seconds": 180,
    "snapshot_seconds": [0, 30, 60, 90, 120, 150, 180],
    "layers": [3, 32, 32, 32, 1],
    "pde_points": 1024,
    "ic_points": 256,
    "face_points": 128,
    "grid": [81, 81, 41],
    "pde_holdout": 4096,
    "boundary_grid": [161, 81],
    "evaluation_seed": 20261028,
    "adam_lr": 0.001,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "idle_samples": 3,
    "idle_utilization_max": 5,
    "idle_wait_seconds_max": 60,
    "lbfgs_lr": 1.0,
    "history_size": 50,
    "max_iter": 1,
    "max_eval": 1,
    "tolerance_grad": 1e-7,
    "tolerance_change": 1e-9,
    "line_search": "strong_wolfe",
    "max_ls": 25,
    "soft_weights": [1, 1, 1],
    "face_weight": 0.25,
    "backend": "eager",
    "lifting": "recursive",
    "background": "general",
    "rng_offsets": {"weights": 0, "pde": 10000, "ic": 20000, "bc": 30000},
    "cost_warmups": 3,
    "cost_blocks": 7,
    "cost_repeats": 10,
    "significant_overshoot_seconds": 1.8,
}
ENVIRONMENT: Json = {
    "torch": "2.1.0+cu118",
    "cuda_runtime": "11.8",
    "default_dtype": "torch.float32",
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def validate(m: Json) -> Json:
    if (
        set(m)
        != {
            "contract",
            "kind",
            "registration",
            "revision",
            "settings",
            "order",
            "environment_requirements",
        }
        or m.get("contract") != 9
        or m.get("kind") != "soft-hard-talk"
        or m.get("registration") != "prospective"
        or not isinstance(m.get("revision"), str)
        or not re.fullmatch("[0-9a-f]{40}", m["revision"])
        or m.get("settings") != SETTINGS
        or m.get("order") != [list(row) for row in ORDER]
        or m.get("environment_requirements") != ENVIRONMENT
    ):
        raise ValueError("Expected fixed prospective soft-hard-talk contract9")
    return m


def plan(m: Json) -> Json:
    validate(m)
    return dict(
        contract=1,
        kind="soft-hard-talk-plan",
        revision=m["revision"],
        trials=6,
        training_seconds_total=1080,
        adam_seconds_per_trial=60,
        total_seconds_per_trial=180,
        snapshots_per_trial=7,
        order=m["order"],
        settings=m["settings"],
        extra_training=False,
        comparison="paired final error; actual time and overshoot retained",
    )


def finite(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite scientific evidence")


def compare(e: Json) -> Json:
    if e.get("kind") != "soft-hard-talk-evidence" or e.get("contract") != 1:
        raise ValueError("Expected soft/hard study evidence")
    m = validate(e["manifest"])
    finite(e)
    if e.get("environment") != ENVIRONMENT or "4070 Ti" not in e.get("gpu", ""):
        raise ValueError("Wrong GPU or precision/runtime")
    if e.get("source_unchanged") is not True or len(e.get("trials", [])) not in (2, 4, 6):
        raise ValueError("Incomplete or changed study")
    if [(t["seed"], t["method"]) for t in e["trials"]] != ORDER[: len(e["trials"])]:
        raise ValueError("Study order differs")
    pairs = []
    for seed in (19, 73, 1234):
        ts = {t["method"]: t for t in e["trials"] if t["seed"] == seed}
        if not ts:
            continue
        for key in ("initial_parameters_sha256", "pde_sha256", "samples_sha256"):
            if ts["soft"][key] != ts["hard"][key]:
                raise ValueError("Pair initialization or samples differ")
        for t in ts.values():
            s = t["snapshots"]
            nominal = [v for snap in s for v in snap["nominal_seconds"]]
            if sorted(nominal) != SETTINGS["snapshot_seconds"]:
                raise ValueError("Missing or duplicate scheduled marks")
            if (
                len(s) < 3
                or s[0]["stage"] != "initial"
                or not any(v["stage"] == "adam_end" for v in s)
                or s[-1]["stage"] != "final"
                or t["seconds"] < 180
                or t["adam_end_seconds"] < 60
                or t["seconds"] < t["adam_end_seconds"]
                or any(a["seconds"] > b["seconds"] for a, b in zip(s, s[1:]))
                or any(v.get("accepted") is not True for v in s)
            ):
                raise ValueError("Missing safe time budget/snapshots")
            if (
                t["overshoot_seconds"] != max(0.0, t["seconds"] - 180)
                or t["adam_updates"] < 1
                or t["lbfgs_steps"] < 1
                or t["closures"] < t["lbfgs_steps"]
                or set(t["costs"]) != {"loss_gradient", "adam_update", "lbfgs_closure"}
                or t["scientific_checkpoints_unchanged"] is not True
            ):
                raise ValueError("Optimizer work, costs or checkpoint integrity differs")
            for v in s:
                if len(v["metrics"]["temporal"]) != 41 or len(v["metrics"]["boundaries"]) != 4:
                    raise ValueError("Incomplete independent metrics")
        soft, hard = ts["soft"]["snapshots"][-1]["metrics"], ts["hard"]["snapshots"][-1]["metrics"]
        pairs.append(
            dict(
                seed=seed,
                soft_relative_l2=soft["relative_l2"],
                hard_relative_l2=hard["relative_l2"],
                hard_wins=hard["relative_l2"] < soft["relative_l2"],
                hard_over_soft=hard["relative_l2"] / soft["relative_l2"],
            )
        )
    return dict(
        contract=1,
        kind="soft-hard-talk-comparison",
        comparison_passed=len(e["trials"]) == 6,
        complete=len(e["trials"]) == 6,
        trials=len(e["trials"]),
        planned_trials=6,
        pairs=pairs,
        hard_wins=sum(p["hard_wins"] for p in pairs),
        strict_equal_time=False,
        significant_overshoot=any(
            t["overshoot_seconds"] > m["settings"]["significant_overshoot_seconds"]
            for t in e["trials"]
        ),
        scope="complete paired study; no universal significance or continuous-error certificate",
    )


def collect(root: Path, m: Json) -> Json:
    e = read_object(root / "study.json")
    if e["manifest"] != validate(m):
        raise ValueError("Manifest differs")
    for relative, sha in e["artifacts"].items():
        if digest(inside(root, relative)) != sha:
            raise ValueError("Artifact hash differs")
    compare(e)
    return e
