"""Execute a validated convergence protocol on the remote CUDA worker only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or Path.cwd() != Path(r"C:\Projects\pinn-runner"):
        parser.error("Research training is restricted to the Windows project directory")
    from research_tools.schema import inside, load_manifest

    manifest = load_manifest(args.manifest)
    from research_tools.planning import plan as offline_plan

    offline_plan(manifest)
    import torch

    from benchmarks.cases import CASES
    from benchmarks.convergence import ConvergenceArm, ConvergencePlan
    from benchmarks.convergence_processes import run_isolated
    from pinnstorch.runner.settings import TaskSettings
    from research_tools.queue import registered_manifest, verify_snapshot
    from scripts.gpu_worker import write_json

    parameters = manifest["report_parameters"]
    arms = [
        ConvergenceArm(
            *(
                manifest["comparison"][name][key]
                for key in ("backend", "lifting", "backward", "background", "spatial_encoding")
            )
        )
        for name in ("baseline", "candidate")
    ]
    cases = list(manifest["case_limits"])
    if len(cases) != len(manifest["reports"]):
        raise ValueError("One report per case, ordered like case_limits, required")
    plans = []
    for case in cases:
        seeds = sorted({t["seed"] for t in manifest["expected_trials"] if t["case"] == case})
        plan = ConvergencePlan(
            CASES[case],
            parameters["adam_steps"],
            tuple(parameters["lbfgs_snapshots"]),
            manifest["case_limits"][case],
            parameters["constraints"],
            parameters["points_override"],
            tolerance_change=parameters["lbfgs_tolerance_change"],
        )
        for arm in arms:
            config = plan.config(seeds[0], arm)
            if config["network"] != manifest["network"]:
                raise ValueError("Resolved network differs from registered architecture")
            TaskSettings.read(config)
        plans.append((plan, seeds))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for index, (relative, (plan, seeds)) in enumerate(zip(manifest["reports"], plans)):
        write_json(
            args.manifest.parent / "progress.json",
            {
                "completed_cases": index,
                "expected_cases": len(plans),
                "current_case": plan.case.name,
            },
        )
        if not relative.endswith("/report.json"):
            raise ValueError("Report paths must end in /report.json")
        registered_manifest(args.manifest.parent)
        verify_snapshot(args.manifest.parent)
        output = inside(args.output, relative).parent
        run_isolated(plan, arms, seeds, output, ROOT / "scripts/benchmark_time_to_quality.py")
        write_json(
            args.manifest.parent / "progress.json",
            {"completed_cases": index + 1, "expected_cases": len(plans), "current_case": None},
        )


if __name__ == "__main__":
    main()
