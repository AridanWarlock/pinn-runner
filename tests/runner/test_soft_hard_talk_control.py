"""Contract9 routing is published/snapshot-bound and leaves legacy root workers unchanged."""

import json
from copy import deepcopy

import pytest

from research_tools import control
from research_tools.evidence import read_bundle, write_bundle
from research_tools.soft_hard_talk import ENVIRONMENT, ORDER, SETTINGS, compare


def manifest():
    return dict(
        contract=9,
        kind="soft-hard-talk",
        registration="prospective",
        revision="a" * 40,
        settings=deepcopy(SETTINGS),
        order=[list(v) for v in ORDER],
        environment_requirements=ENVIRONMENT.copy(),
    )


def evidence():
    trials = []
    for seed, method in ORDER:
        metrics = dict(
            relative_l2=0.1 if method == "soft" else 0.05,
            final_relative_l2=0.1,
            temporal=[dict(t=i / 80, relative_l2=0.1, rmse=0.01) for i in range(41)],
            boundaries={k: dict(max=1e-6, rmse=1e-7) for k in range(4)},
        )
        snaps = [
            dict(
                seconds=float(t),
                stage="initial"
                if t == 0
                else "adam_end"
                if t == 60
                else "final"
                if t == 180
                else "adam"
                if t < 60
                else "lbfgs",
                nominal_seconds=[t],
                accepted=True,
                metrics=deepcopy(metrics),
            )
            for t in SETTINGS["snapshot_seconds"]
        ]
        trials.append(
            dict(
                seed=seed,
                method=method,
                snapshots=snaps,
                seconds=180.0,
                adam_end_seconds=60.0,
                overshoot_seconds=0.0,
                initial_parameters_sha256="b" * 64,
                pde_sha256="c" * 64,
                samples_sha256="d" * 64,
                adam_updates=3,
                lbfgs_steps=3,
                closures=6,
                scientific_checkpoints_unchanged=True,
                costs={
                    k: dict(median_ms=1) for k in ("loss_gradient", "adam_update", "lbfgs_closure")
                },
            )
        )
    return dict(
        contract=1,
        kind="soft-hard-talk-evidence",
        manifest=manifest(),
        environment=ENVIRONMENT.copy(),
        gpu="NVIDIA GeForce RTX 4070 Ti",
        source_unchanged=True,
        trials=trials,
    )


def test_full_inventory_survives_normal_bundle_and_compare(tmp_path):
    e = evidence()
    assert compare(e)["hard_wins"] == 3
    write_bundle(tmp_path / "bundle", e)
    assert compare(read_bundle(tmp_path / "bundle"))["trials"] == 6


@pytest.mark.parametrize(
    "damage",
    ["order", "weights", "samples", "time", "schedule", "unsafe", "partial", "cost", "checkpoint"],
)
def test_study_rejects_semantic_or_integrity_substitution(damage):
    e = evidence()
    if damage == "order":
        e["trials"].reverse()
    elif damage == "weights":
        e["trials"][1]["initial_parameters_sha256"] = "0" * 64
    elif damage == "samples":
        e["trials"][1]["pde_sha256"] = "0" * 64
    elif damage == "time":
        e["trials"][0]["seconds"] = 179
    elif damage == "schedule":
        e["trials"][0]["snapshots"][1]["nominal_seconds"] = []
    elif damage == "unsafe":
        e["trials"][0]["snapshots"][1]["accepted"] = False
    elif damage == "partial":
        e["trials"].pop()
    elif damage == "cost":
        e["trials"][0]["costs"] = {}
    else:
        e["trials"][0]["scientific_checkpoints_unchanged"] = False
    with pytest.raises(ValueError):
        compare(e)


def test_branch_worker_is_verified_before_execution(monkeypatch):
    captured = []
    monkeypatch.setattr(
        control,
        "remote",
        lambda script: captured.append(script) or json.dumps(dict(state="queued")),
    )
    request = dict(
        job_id="talk",
        protocol_commit="b" * 40,
        protocol_path="docs/p.json",
        protocol_sha256="c" * 64,
        timeout_seconds=3600,
        probe=False,
    )
    control.call("prepare", "talk", request=request, bootstrap_commit="b" * 40)
    s = captured[0]
    assert "git pull --ff-only" in s and "Bootstrap revision is not published" in s
    assert "git archive --format=zip" in s and "base64.b64decode" in s
    assert s.index("base64.b64decode") < s.index("/scripts/research_worker.py prepare")
    assert "git checkout" not in s and "git switch" not in s


def test_unknown_or_legacy_job_keeps_root_worker_fallback(monkeypatch):
    captured = []
    monkeypatch.setattr(
        control,
        "remote",
        lambda script: captured.append(script) or json.dumps(dict(state="unknown")),
    )
    control.call("status", "legacy")
    assert "$manifest.contract -eq 9" in captured[0]
    assert "$worker = 'scripts/research_worker.py'" in captured[0]
