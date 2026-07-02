#!/usr/bin/env python3
"""Audit every sensor CSV in src/data/sensor_models/ against the strict loader.

Runs ErrorModel construction for every (sensor, mode) combination. Failures are
aggregated into a markdown report grouped by sensor, with actionable diagnostics.

Usage:
    python scripts/audit_error_model_data.py                    # print to stdout
    python scripts/audit_error_model_data.py -o results/audit.md  # write to file
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from error_model import (  # noqa: E402
    ErrorModel,
    ErrorModelCoverageError,
    ErrorModelSchemaError,
)

SENSOR_MODELS_DIR = REPO_ROOT / "src" / "data" / "sensor_models"
LOCALIZER_PREFIXES = ("kiss_icp", "orb_slam3", "ct_icp", "CT_ICP", "hd_map", "rtk_v2v4real")


def _is_error_model_regression(csv_path: Path) -> bool:
    """An error-model regression CSV has an `error_type` column as its header.
    Files like *_fp_records.csv have a completely different schema and are not
    error models — skip them. (Header is the first non-comment, non-empty line.)
    """
    try:
        with csv_path.open() as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith("#"):
                    return "error_type" in line.split(",")[0:1] or "error_type" in line
        return False
    except OSError:
        return False


def all_sensors() -> list[str]:
    stems: set[str] = set()
    for csv in SENSOR_MODELS_DIR.glob("*.csv"):
        name = csv.stem
        if name.endswith(("_distributions", "_polar_distributions", "_polar_calibration",
                          "_fp_records")):
            continue
        if not _is_error_model_regression(csv):
            continue
        stems.add(name)
    return sorted(stems)


def is_localizer(sensor: str) -> bool:
    return any(sensor.startswith(p) for p in LOCALIZER_PREFIXES)


def audit() -> dict[str, dict[str, str | None]]:
    """Return {sensor: {mode: error_message_or_None}}."""
    result: dict[str, dict[str, str | None]] = {}
    for sensor in all_sensors():
        independent_var = "velocity" if is_localizer(sensor) else "distance"
        max_range = 22.0 if independent_var == "velocity" else 100.0
        result[sensor] = {}
        modes = ["static", "linear", "quadratic"]
        if independent_var == "distance":
            modes.append("polar")
        for mode in modes:
            try:
                ErrorModel(sensor, mode=mode, independent_var=independent_var,
                           max_range=max_range)
                result[sensor][mode] = None  # passed
            except (ErrorModelCoverageError, ErrorModelSchemaError) as e:
                result[sensor][mode] = str(e)
    return result


def render_markdown(results: dict[str, dict[str, str | None]]) -> str:
    lines = [
        f"# Error Model Data Audit ({date.today().isoformat()})",
        "",
        "Runs the strict-schema `ErrorModel` loader against every sensor CSV in",
        "`src/data/sensor_models/` for each supported mode. Failures are listed",
        "below with the exact loader diagnostic. Re-run with:",
        "",
        "    python scripts/audit_error_model_data.py",
        "",
    ]

    total_attempted = sum(len(v) for v in results.values())
    total_failed = sum(1 for sensor in results for mode in results[sensor] if results[sensor][mode] is not None)
    passed = total_attempted - total_failed

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Sensors audited: **{len(results)}**")
    lines.append(f"- (sensor, mode) combinations attempted: **{total_attempted}**")
    lines.append(f"- Passing: **{passed}**")
    lines.append(f"- Failing: **{total_failed}**")
    lines.append("")

    # Group sensors by failure pattern for readability
    pattern_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for sensor, modes in results.items():
        failing_modes = tuple(sorted(m for m, err in modes.items() if err is not None))
        pattern_groups[failing_modes].append(sensor)

    if () in pattern_groups:
        lines.append("## Sensors passing all supported modes")
        lines.append("")
        for s in sorted(pattern_groups[()]):
            lines.append(f"- `{s}`")
        lines.append("")
        del pattern_groups[()]

    if pattern_groups:
        lines.append("## Sensors with failures")
        lines.append("")
        for failing in sorted(pattern_groups, key=lambda t: (-len(t), t)):
            sensors_in_group = sorted(pattern_groups[failing])
            lines.append(f"### Fails in: {', '.join(failing)}")
            lines.append("")
            for sensor in sensors_in_group:
                lines.append(f"#### `{sensor}`")
                for mode in failing:
                    err = results[sensor][mode]
                    lines.append(f"- **mode={mode}**: {err}")
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## How to fix")
    lines.append("")
    lines.append("- **Missing overall bin** (mode=static failures): run")
    lines.append("  `scripts/synthesize_overall_bins.py <sensor>` (done automatically by")
    lines.append("  `scripts/import_sim_gpem_models.py`) to append the count-weighted")
    lines.append("  `dist_min=0, dist_max=overall_dist_max` row per axis.")
    lines.append("- **Missing quadratic coefficients** (mode=quadratic): regenerate the")
    lines.append("  regression CSV with `quad_a, quad_b, quad_c` populated.")
    lines.append("- **Missing polar file or partial coverage** (mode=polar): regenerate")
    lines.append("  the polar distributions CSV with full coverage of `[0, max_range)` ×")
    lines.append("  `[-180, 180)` for every axis.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write report to this path (default: stdout)")
    args = ap.parse_args()

    results = audit()
    report = render_markdown(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote audit report to {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
