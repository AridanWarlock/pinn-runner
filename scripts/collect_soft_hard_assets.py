"""Fetch only SHA-verified artifacts of the published study; never execute training."""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

from scripts.gpu_jobs import remote
from scripts.gpu_worker import valid_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoints", action="store_true")
    args = p.parse_args()
    valid_id(args.job_id)
    evidence = json.loads((args.bundle / "evidence.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for relative, sha in evidence["artifacts"].items():
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative) or ".." in relative.split("/"):
            raise ValueError("Unsafe artifact path")
        if not (
            relative.endswith((".npz", ".yaml")) or args.checkpoints and relative.endswith(".ckpt")
        ):
            continue
        target = args.output / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
            continue
        path = (
            "C:\\Projects\\pinn-runner\\.gpu-jobs\\"
            + args.job_id
            + "\\r\\"
            + relative.replace("/", "\\")
        )
        encoded = remote(f"[Convert]::ToBase64String([IO.File]::ReadAllBytes('{path}'))")
        data = base64.b64decode(encoded.strip())
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Artifact checksum differs")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(relative, len(data), flush=True)
    (args.output / "artifact-sha256.json").write_text(
        json.dumps(evidence["artifacts"], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
