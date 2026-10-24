"""Offline executability checks and budget accounting, not a convergence proof."""

from __future__ import annotations

from pathlib import Path

from benchmarks.catalog import (
    BACKGROUNDS,
    BACKWARDS,
    CATALOG,
    CONSTRAINTS,
    ENCODINGS,
    EXECUTIONS,
    LIFTINGS,
    CaseSpec,
)

from .schema import Json, face_identity, validate_manifest


def validate_arm(arm: Json, constraints: str, spec: CaseSpec) -> None:
    choices = dict(
        backend=EXECUTIONS,
        lifting=LIFTINGS,
        backward=BACKWARDS,
        background=BACKGROUNDS,
        spatial_encoding=ENCODINGS,
    )
    for field, allowed in choices.items():
        if arm[field] not in allowed:
            raise ValueError(f"Unsupported {field}: {arm[field]}")
    backend, encoding = arm["backend"], arm["spatial_encoding"]
    if constraints == "soft" and backend != "eager":
        raise ValueError("Soft constraints require eager execution")
    robin = constraints == "hard_initial_robin"
    if (backend in ("robin_batch", "robin_grouped", "neumann_analytic")) and not robin:
        raise ValueError("Robin executors require hard_initial_robin")
    if robin and backend == "product_rule":
        raise ValueError("product_rule is a Dirichlet executor")
    if not robin and (arm["lifting"] != "recursive" or arm["background"] != "general"):
        raise ValueError("Expanded lifting and background selection require hard_initial_robin")
    if spec.family == "advection" and constraints != "soft":
        raise ValueError("Advection convergence requires soft inflow without an outflow face")
    if spec.family == "burgers" and constraints not in ("soft", "hard_initial_dirichlet"):
        raise ValueError("Burgers requires soft or compatible exact Dirichlet constraints")
    kinds = {face["kind"] for face in spec.faces()}
    if constraints == "hard_initial_dirichlet" and kinds != {"dirichlet"}:
        raise ValueError("Dirichlet hard constraints require Dirichlet faces")
    if robin and "dirichlet" in kinds:
        raise ValueError("Current Robin projection requires nonzero derivative coefficients")
    if arm["background"] == "static" and not spec.static_background:
        raise ValueError("This reference case has no declared static Neumann background")
    if encoding != "identity" and (
        not robin or backend not in ("eager", "neumann_analytic") or kinds != {"neumann"}
    ):
        raise ValueError(
            "Neumann features require pure Neumann faces and eager/analytic Robin execution"
        )
    if backend == "neumann_analytic" and encoding != "neumann_polynomial":
        raise ValueError("Analytic feature derivatives require polynomial encoding")


def plan(manifest: Json, *, details: bool = False) -> Json:
    if manifest.get("contract") == 8:
        from .public_prediction import plan as public_plan

        return public_plan(manifest)
    if manifest.get("contract") == 7:
        from .allocation_profile import plan as allocation_plan

        return allocation_plan(manifest)
    if manifest.get("contract") == 6:
        from .numerical_acceptance import plan as acceptance_plan

        return acceptance_plan(manifest)
    if manifest.get("contract") == 5:
        from .lbfgs_timing import plan as lbfgs_plan

        return lbfgs_plan(manifest)
    if manifest.get("contract") == 4:
        from .frozen_timing import plan as timing_plan

        return timing_plan(manifest)
    if manifest.get("contract") == 3:
        from .frozen_profile import plan as profile_plan

        return profile_plan(manifest)
    manifest = validate_manifest(manifest, Path.cwd())
    if manifest["contract"] != 2 or manifest["registration"] != "prospective":
        raise ValueError("Execution planning requires prospective manifest contract 2")
    parameters = manifest["report_parameters"]
    if parameters["constraints"] not in CONSTRAINTS:
        raise ValueError("Unsupported constraint strategy")
    if parameters["lbfgs_snapshots"][0] != 0:
        raise ValueError("The snapshot schedule must start at zero")
    reports = manifest["reports"]
    if len(reports) != len(manifest["case_limits"]) or any(
        not path.endswith("/report.json") for path in reports
    ):
        raise ValueError("One report per case ending in /report.json required")
    if any(
        a != b and a.startswith(b + "/")
        for a in (p.rsplit("/", 1)[0] for p in reports)
        for b in (p.rsplit("/", 1)[0] for p in reports)
    ):
        raise ValueError("Case report directories must not be nested")
    precision = manifest["environment_requirements"]
    if (
        precision["default_dtype"] != "torch.float32"
        or precision["float32_matmul_precision"] != "highest"
        or precision["cuda_matmul_allow_tf32"]
        or precision["cudnn_allow_tf32"]
    ):
        raise ValueError("Current convergence executor uses FP32 with TF32 disabled")
    rows: list[Json] = []
    for (name, limits), report in zip(manifest["case_limits"].items(), reports):
        if name not in CATALOG or (
            not CATALOG[name].convergence and "absolute_quality" not in manifest
        ):
            raise ValueError(f"No convergence executor registered for {name}")
        spec = CATALOG[name]
        supported = {"relative_l2", "final_relative_l2", "pde_rmse", "mode_relative_l2"}
        if "physics" in manifest:
            supported |= {"worst_time_relative_l2", "physical_flux_max"}
            if not supported - {"mode_relative_l2"} <= limits.keys():
                raise ValueError("Coefficient study requires temporal and physical flux gates")
        elif "absolute_quality" in manifest:
            if spec.family not in (
                "advection",
                "burgers",
                "burgers_unforced",
                "constant_surrogate",
                "spatial_scale_surrogate",
                "quadratic_shapes",
            ):
                raise ValueError("Absolute quality requires a registered analytical case")
            import hashlib

            source = (
                Path(__file__).resolve().parents[1] / "benchmarks/problems" / name / "functions.py"
            )
            if (
                hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                != manifest["problem_sha256"][name]
            ):
                raise ValueError("Problem source differs from registered case")
            supported |= {"worst_time_relative_l2"}
            if spec.family in ("constant_surrogate", "spatial_scale_surrogate"):
                supported |= {"coefficient_response_relative_l2"}
                reference = Path(__file__).resolve().parents[1] / (
                    "benchmarks/spatial_surrogate_quality.py"
                    if spec.family == "spatial_scale_surrogate"
                    else "benchmarks/surrogate_quality.py"
                )
                if (
                    hashlib.sha256(reference.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    != manifest["hypothesis_evaluation"]["reference_sha256"]
                ):
                    raise ValueError("Surrogate reference differs from protocol")
            if spec.family == "quadratic_shapes":
                from benchmarks.catalog import QUADRATIC_METRICS

                supported |= QUADRATIC_METRICS
                for field, filename in (
                    ("reference_sha256", "quadratic_surrogate_quality.py"),
                    ("analytical_sha256", "quadratic_reference.py"),
                ):
                    reference = Path(__file__).resolve().parents[1] / "benchmarks" / filename
                    if (
                        hashlib.sha256(reference.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                        != manifest["hypothesis_evaluation"][field]
                    ):
                        raise ValueError("Whole-code evaluator or analytical source differs")
            if spec.family == "burgers_unforced":
                from benchmarks.catalog import UNFORCED_METRICS

                supported |= UNFORCED_METRICS
                reference = Path(__file__).resolve().parents[1] / "benchmarks/burgers.py"
                if (
                    hashlib.sha256(reference.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    != manifest["layer_evaluation"]["reference_sha256"]
                ):
                    raise ValueError("Reference source differs from registered protocol")
            if not supported - {"mode_relative_l2"} <= limits.keys():
                raise ValueError("Absolute quality requires global/final/worst-time/PDE gates")
        elif spec.family == "conductivity":
            raise ValueError("Conductivity cases require explicit physics")
        if set(limits) - supported:
            raise ValueError("Unsupported quality metric")
        if "mode_relative_l2" in limits and not spec.modal_metric:
            raise ValueError("The selected case has no modal error evaluator")
        if {face_identity(f) for f in manifest["case_faces"][name]} != {
            face_identity(f) for f in spec.faces()
        }:
            raise ValueError("Expected faces differ from the registered problem")
        arms = (
            [manifest["absolute_quality"]]
            if "absolute_quality" in manifest
            else [manifest["comparison"][k] for k in ("baseline", "candidate")]
        )
        for arm in arms:
            validate_arm(arm, parameters["constraints"], spec)
            expected_network: Json = {
                "hidden_layers": list(spec.layers),
                "spatial_encoding": arm["spatial_encoding"],
            }
            if name == "heat_quadratic_clock":
                expected_network["time_scale"] = {"parameter": "a", "reference": 0.15}
            if manifest["network"] != expected_network:
                raise ValueError("Network differs from the registered case configuration")
        seeds = sorted({t["seed"] for t in manifest["expected_trials"] if t["case"] == name})
        rows.append(
            dict(
                case=name,
                dimensions=spec.dimensions,
                seeds=seeds if details else seeds[:5],
                seeds_truncated=not details and len(seeds) > 5,
                pairs=0 if "absolute_quality" in manifest else len(seeds),
                trials=len(arms) * len(seeds),
                report=report,
                limits=limits,
            )
        )
    count = len(manifest["expected_trials"])
    return {
        **(
            {"density": manifest["density"], "budget_kind": "wall_time_with_step_caps"}
            if "density" in manifest
            else {}
        ),
        "contract": 1,
        "kind": "research-plan",
        "revision": manifest["revision"],
        "registration": manifest["registration"],
        "arms": {"single": manifest["absolute_quality"]}
        if "absolute_quality" in manifest
        else {name: manifest["comparison"][name] for name in ("baseline", "candidate")},
        "constraints": parameters["constraints"],
        "catalog_scope": "current_tool_version",
        "comparison_purpose": "absolute_accuracy"
        if "physics" in manifest or "absolute_quality" in manifest
        else "paired_cost",
        "cases": rows,
        "trials": count,
        "adam_steps_total": count * parameters["adam_steps"],
        "lbfgs_calls_total": count * parameters["lbfgs_snapshots"][-1],
        "saved_states_total": count * len(parameters["lbfgs_snapshots"])
        if "density" not in manifest
        else None,
        **(
            {
                "saved_states_upper_bound": count
                * (
                    int(
                        manifest["density"]["training_seconds"]
                        / manifest["density"]["snapshot_seconds"]
                    )
                    + 2
                )
            }
            if "density" in manifest
            else {}
        ),
        "collocation_override": parameters["points_override"],
        "quality_confirmation_possible": len(parameters["lbfgs_snapshots"]) >= 2,
        "limitations": "Offline structural check only: runtime packages, callbacks and mathematical correctness still require worker validation. L-BFGS closures and time-to-quality cannot be inferred from the step budget.",
    }
