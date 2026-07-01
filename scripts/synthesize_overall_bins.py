#!/usr/bin/env python3
"""Append synthesized overall-bin rows to sensor distributions CSVs that lack them.

Used to unblock the strict loader for sensors whose source data was not
regenerated through `import_mmdet_error_models.py`'s overall-bin generator.

Strategy per axis (e.g. 'distal'):
  - Take every per-range bin row for this axis from {sensor}_distributions.csv
  - Compute std via the existing distribution-type math (matches _DistBin.get_std)
  - Aggregate as sqrt(mean(std²)) — unweighted because per-bin counts are not
    in the CSV. This matches the silent fallback the OLD loader used; making
    it explicit avoids the strict loader's coverage error.
  - Append one row per axis: error_type, dist_min=0, dist_max=overall_dist_max,
    distribution=normal, param1=0.0, param2=aggregated_std

Marks the appended rows with a leading comment so future regenerations can
overwrite cleanly.

Usage:
    python scripts/synthesize_overall_bins.py [sensor1] [sensor2] ...
    # (Pass sensor stems; e.g. pointpillar_v2v4real_tesla)

If no sensors specified, prints the list of sensors that would benefit
(per the loader audit) but does nothing.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from error_model_schema import DETECTOR_AXES, LOCALIZER_AXES, overall_bin_min_width

SENSOR_MODELS_DIR = REPO_ROOT / "src" / "data" / "sensor_models"
MARKER = "# === Synthesized overall bin rows (added by synthesize_overall_bins.py) ==="

# Conversion math taken directly from error_model._DistBin.get_std() so the
# synthesized aggregate matches what the loader would compute per-bin.
def _bin_std(dist_type: str, params: dict) -> float | None:
    """Return per-bin std, or None if the distribution has undefined std."""
    if dist_type == "normal":
        return float(params["sigma"])
    if dist_type == "laplace":
        return math.sqrt(2.0) * float(params["b"])
    if dist_type == "logistic":
        return float(params["s"]) * math.pi / math.sqrt(3.0)
    if dist_type == "student_t":
        nu = float(params["nu"])
        if nu > 2:
            return float(params["sigma"]) * math.sqrt(nu / (nu - 2.0))
        return None
    if dist_type == "cauchy":
        return None
    if dist_type == "constant":
        return 0.0
    return None


def _params_from_row(dist_type: str, p1, p2, p3) -> dict:
    """Map (param1, param2, param3) to distribution-specific params."""
    if dist_type == "normal":
        return {"mu": float(p1), "sigma": float(p2)}
    if dist_type == "laplace":
        return {"mu": float(p1), "b": float(p2)}
    if dist_type == "logistic":
        return {"mu": float(p1), "s": float(p2)}
    if dist_type == "student_t":
        return {"nu": float(p1), "mu": float(p2), "sigma": float(p3)}
    if dist_type == "cauchy":
        return {"x0": float(p1), "gamma": float(p2)}
    if dist_type == "constant":
        return {"value": float(p1)}
    return {}


def aggregate_axis(rows: list[dict], csv_key: str, threshold: float) -> float | None:
    """Compute sqrt(mean(std²)) across per-range bins for this axis.

    Skip wide bins (dist_max - dist_min > threshold) — those would be existing
    overall rows and shouldn't be double-counted.
    """
    variances: list[float] = []
    for row in rows:
        if (row.get("error_type") or "").strip() != csv_key:
            continue
        try:
            lo = float(row["dist_min"]); hi = float(row["dist_max"])
        except (TypeError, ValueError, KeyError):
            continue
        if hi - lo > threshold:
            continue  # already a wide/overall bin
        dt = (row.get("distribution") or "normal").strip()
        try:
            p1 = float(row.get("param1") or 0)
            p2 = float(row.get("param2") or 0)
            p3 = float(row.get("param3") or 0) if (row.get("param3") or "").strip() else None
        except ValueError:
            continue
        std = _bin_std(dt, _params_from_row(dt, p1, p2, p3))
        if std is None or std <= 0:
            continue
        variances.append(std * std)
    if not variances:
        return None
    return math.sqrt(sum(variances) / len(variances))


def synthesize_for_sensor(sensor: str, dry_run: bool = False) -> bool:
    """Return True if the CSV was modified."""
    # Determine independent_var by the sensor's regression CSV header — we don't
    # special-case localizer prefixes here.
    reg_path = SENSOR_MODELS_DIR / f"{sensor}.csv"
    dist_path = SENSOR_MODELS_DIR / f"{sensor}_distributions.csv"
    if not dist_path.exists():
        print(f"[skip] no distributions CSV for {sensor}: {dist_path}")
        return False

    # Find which axis spec matches by checking the regression's unit on each axis.
    # Simpler: try detector first, fall back to localizer based on which set of
    # CSV keys appear in distributions.
    with dist_path.open() as f:
        text = f.read()
    # Use error_type column to decide.
    detector_keys = {a.distribution_csv_key for a in DETECTOR_AXES}
    localizer_keys = {a.distribution_csv_key for a in LOCALIZER_AXES}
    found_keys = set()
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        return False
    reader = csv.DictReader(lines)
    rows = list(reader)
    for r in rows:
        et = (r.get("error_type") or "").strip()
        if et:
            found_keys.add(et)

    detector_hits = len(found_keys & detector_keys)
    localizer_hits = len(found_keys & localizer_keys)
    if detector_hits >= localizer_hits and detector_hits > 0:
        axes = DETECTOR_AXES
        independent_var = "distance"
        max_aggregate_range = 100.0
    elif localizer_hits > 0:
        axes = LOCALIZER_AXES
        independent_var = "velocity"
        max_aggregate_range = 22.0
    else:
        print(f"[skip] {sensor}: distributions CSV has no recognized error_type rows")
        return False

    threshold = overall_bin_min_width(independent_var)

    # Aggregate per axis
    new_rows: list[str] = []
    for axis in axes:
        # Only synthesize if no wide bin already exists.
        wide_existing = any(
            (r.get("error_type") or "").strip() == axis.distribution_csv_key
            and (float(r["dist_max"]) - float(r["dist_min"])) > threshold
            for r in rows
            if r.get("dist_min") and r.get("dist_max")
        )
        if wide_existing:
            continue
        agg = aggregate_axis(rows, axis.distribution_csv_key, threshold)
        if agg is None:
            print(f"  {sensor}/{axis.name}: no per-range bins to aggregate; skipping")
            continue
        new_rows.append(
            f"{axis.distribution_csv_key},0,{max_aggregate_range:g},normal,0.0,{agg:.6f},"
        )

    if not new_rows:
        print(f"[no-op] {sensor}: all axes already have wide bins")
        return False

    print(f"[append] {sensor}: {len(new_rows)} synthesized overall rows")
    for r in new_rows:
        print(f"    {r}")
    if dry_run:
        return False

    with dist_path.open("a") as f:
        f.write("\n" + MARKER + "\n")
        f.write(f"# Aggregated as sqrt(mean(per-bin variance)) across per-{independent_var} bins.\n")
        f.write("# This is an unweighted aggregate; the upstream generator should emit a\n")
        f.write("# count-weighted overall row to replace these.\n")
        for r in new_rows:
            f.write(r + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sensors", nargs="*", help="Sensor stems (e.g. pointpillar_v2v4real_tesla)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.sensors:
        print("Usage: synthesize_overall_bins.py <sensor1> [<sensor2> ...] [--dry-run]")
        return 1
    modified = 0
    for s in args.sensors:
        if synthesize_for_sensor(s, dry_run=args.dry_run):
            modified += 1
    print(f"\n{'(dry run) ' if args.dry_run else ''}Modified {modified} of {len(args.sensors)} CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
