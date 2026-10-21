"""Read only bounded temporal summaries from completed Windows reports."""

import argparse
import base64
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_tools.evidence import read_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-id", choices=("nonheat032", "nonheat033", "nonheat034"), required=True
    )
    args = parser.parse_args()
    evidence = read_bundle(ROOT / f"numerical-reports/{args.job_id}-compact")
    names = sorted({t["case"] for t in evidence["trials"]})
    paths = ",".join(f"'{name}'" for name in names)
    command = r"""Set-Location -LiteralPath 'C:\Projects\pinn-runner' -ErrorAction Stop
if ((Get-Location).Path -ne 'C:\Projects\pinn-runner') { exit 1 }
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$rows = @()
$files = @()
foreach ($name in @(__CASES__)) {
  $path = ".gpu-jobs\__JOB__\r\$name\report.json"
  $report = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
  $files += [pscustomobject]@{
    case = $name
    sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    revision = $report.environment.revision
  }
  foreach ($trial in $report.trials) {
    foreach ($snap in $trial.snapshots) {
      $field = $snap.temporal | Sort-Object -Property relative_l2 -Descending | Select-Object -First 1
      $gradient = $snap.temporal | Sort-Object -Property gradient_relative_l2 -Descending | Select-Object -First 1
      $rows += [pscustomobject]@{
        case = $name
        seed = $trial.seed
        lb_steps = $snap.lb_steps
        temporal_points = @($snap.temporal).Count
        field_peak_time = $field.time
        field_peak_error = $field.relative_l2
        gradient_peak_time = $gradient.time
        gradient_peak_error = $gradient.gradient_relative_l2
      }
    }
  }
}
[pscustomobject]@{ files = $files; rows = $rows } | ConvertTo-Json -Depth 6 -Compress
""".replace("__CASES__", paths).replace("__JOB__", args.job_id)
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    output = subprocess.check_output(
        [
            "ssh",
            "aridanPC",
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        text=True,
        timeout=60,
    )
    data = json.loads(output)
    expected = {
        (t["case"], t["seed"], s["lb_steps"]): s["metrics"]
        for t in evidence["trials"]
        for s in t["snapshots"]
    }
    assert len(data["rows"]) == len(expected)
    assert all(f["revision"] == evidence["manifest"]["revision"] for f in data["files"])
    for row in data["rows"]:
        metrics = expected[(row["case"], row["seed"], row["lb_steps"])]
        assert row["temporal_points"] == 401
        assert abs(row["field_peak_error"] - metrics["worst_time_relative_l2"]) < 1e-12
        assert abs(row["gradient_peak_error"] - metrics["worst_time_gradient_relative_l2"]) < 1e-12
        assert 0 <= row["field_peak_time"] <= 1 and 0 <= row["gradient_peak_time"] <= 1
    date = "2026-10-21" if args.job_id == "nonheat034" else "2026-10-20"
    prefix = Path(__file__).with_name(f"{date}-{args.job_id}-unforced-temporal")
    assert not prefix.with_suffix(".csv").exists() and not prefix.with_suffix(".json").exists()
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data["rows"])
    prefix.with_suffix(".json").write_text(
        json.dumps(
            dict(
                job_id=args.job_id,
                files=data["files"],
                rows=len(expected),
                scope="Read-only diagnostic from completed reports; no fitting or GPU evaluation; all peaks matched compact evidence.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(job_id=args.job_id, rows=len(expected), matched=True)))


if __name__ == "__main__":
    main()
