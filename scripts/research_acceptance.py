"""Run the one allowlisted numerical suite with a bounded pytest phase report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PhaseReport:
    def __init__(self) -> None:
        self.collected: list[str] = []
        self.records: dict[str, dict[str, Any]] = {}

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report: Any) -> None:
        record = self.records.setdefault(
            report.nodeid,
            dict(
                nodeid=report.nodeid,
                phases=dict(setup="not_run", call="not_run", teardown="not_run"),
                seconds=0.0,
            ),
        )
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xpassed" if report.passed else "xfailed"
        record["phases"][report.when] = outcome
        record["seconds"] += report.duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(r"C:\Projects\pinn-runner")
    if (
        sys.platform != "win32"
        or Path.cwd() != root
        or args.output.exists()
        or not sys.dont_write_bytecode
    ):
        parser.error("Guarded Windows root and fresh output required")
    import pytest
    import torch

    from research_tools.numerical_acceptance import EXPECTED, SUITE, compare
    from research_tools.queue import registered_manifest, verify_snapshot
    from research_tools.schema import digest

    verify_snapshot(args.manifest.parent)
    manifest = registered_manifest(args.manifest.parent)
    if manifest["contract"] != 6 or os.environ.get("PINN_PINNED_REVISION") != manifest["revision"]:
        raise ValueError("Pinned acceptance source differs")
    if not torch.cuda.is_available() or digest(ROOT / SUITE) != manifest["test_sha256"]:
        raise ValueError("CUDA unavailable or numerical tests changed")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def environment() -> dict[str, Any]:
        return dict(
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            default_dtype=str(torch.get_default_dtype()),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        )

    before = environment()
    if before != manifest["environment_requirements"]:
        raise ValueError("Acceptance runtime/precision differs")
    os.environ["PINN_TEST_DEVICE"] = "cuda"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    report = PhaseReport()
    with (args.output / "pytest.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            code = pytest.main(
                [
                    str(ROOT / SUITE),
                    "--rootdir",
                    str(ROOT),
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(args.output / "pytest-temp"),
                    "--maxfail=0",
                    "--tb=short",
                    "-q",
                ],
                plugins=[report],
            )
    raw = dict(
        collected=report.collected,
        trials=[report.records[k] for k in EXPECTED if k in report.records],
        pytest_exit_code=int(code),
    )
    (args.output / "pytest-results.json").write_text(json.dumps(raw), encoding="utf-8")
    evidence = dict(
        contract=1,
        kind="numerical-acceptance-evidence",
        manifest=manifest,
        revision=manifest["revision"],
        test_sha256=manifest["test_sha256"],
        source_unchanged=digest(ROOT / SUITE) == manifest["test_sha256"],
        environment=before,
        environment_after=environment(),
        test_device="cuda",
        cuda_available=True,
        gpu=torch.cuda.get_device_name(),
        python_version=sys.version.split()[0],
        pytest_version=pytest.__version__,
        **raw,
        artifact_sha256={
            name: digest(args.output / name) for name in ("pytest.log", "pytest-results.json")
        },
    )
    comparison = compare(evidence)
    verify_snapshot(args.manifest.parent)
    (args.output / "acceptance.json").write_text(
        json.dumps(evidence, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
