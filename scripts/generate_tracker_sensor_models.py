#!/usr/bin/env python3
"""Generate sensor model CSVs for tracker-based sensor models.

Reads *_polar_calibration.csv from the mmdetection3d sensor_models directory
and produces two files per model:

  1. *_distributions.csv — binned normal error distributions (10m slabs + global)
     used by the GPEM polar/binned mode.

  2. *.csv — polynomial regression parameters (linear + quadratic fits of std and
     bias vs range), matching the format of centerpoint_54m_v2v4real_finetune_*.csv.
     Used by the GPEM linear and quadratic modes.

Models processed:
    cobevt_tracker_astuff, cobevt_tracker_tesla,
    dmstrack_astuff,       dmstrack_tesla

Usage:
    python scripts/generate_tracker_sensor_models.py [--mm-root PATH]
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

# ── local_paths: developer-set external repos (see paths.local.yaml) ──
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import local_paths as _LOCAL_PATHS  # noqa: E402


DEFAULT_MM_ROOT = _LOCAL_PATHS.get("mmdet3d_root")

MODELS = [
    "cobevt_tracker_astuff",
    "cobevt_tracker_tesla",
    "dmstrack_astuff",
    "dmstrack_tesla",
]

# (regression error_type, distributions error_type, mean_col, std_col, unit, note)
ERROR_TYPES = [
    ("radial",     "distal",        "distal_mean",     "distal_std",     "meters",    "Range error"),
    ("lateral",    "perpendicular", "perp_mean",        "perp_std",       "meters",    "Cross-range error"),
    ("vertical",   "height",        "z_mean",           "z_std",          "meters",    "Elevation error"),
    ("yaw",        "yaw",           "yaw_mean",         "yaw_std",        "radians",   "Heading angle"),
    ("width",      "width",         "width_mean",       "width_std",      "meters",    "Box width"),
    ("length",     "length",        "length_mean",      "length_std",     "meters",    "Box length"),
    ("box_height", "box_height",    "box_height_mean",  "box_height_std", "meters",    "Box height"),
]

# 10m slabs for distributions.csv; slabs with no measured data are skipped
SLABS_10M = [(lo, lo + 10) for lo in range(0, 100, 10)]
ALL_SLAB   = (0, 150)

REGRESSION_HEADER = [
    "error_type",
    "intercept", "slope",
    "quad_a", "quad_b", "quad_c",
    "bias_intercept", "bias_slope",
    "bias_quad_a", "bias_quad_b", "bias_quad_c",
    "std_intercept", "std_slope",
    "std_quad_a", "std_quad_b", "std_quad_c",
    "var_intercept", "var_slope",
    "var_quad_a", "var_quad_b", "var_quad_c",
    "unit", "notes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _weighted_stats(rows: list[dict], mean_col: str, std_col: str,
                    weight_col: str = "gt_count") -> tuple[float, float, float, float]:
    """Return (wmean, wstd, wabs_mean, range_center) weighted by gt_count.

    Returns (wmean, wstd, wabs_mean, 0.0); range_center handled outside.
    """
    weights = [_f(r.get(weight_col, 0)) for r in rows]
    means   = [_f(r.get(mean_col)) for r in rows]
    stds    = [_f(r.get(std_col))  for r in rows]

    tot = sum(weights)
    if tot <= 0:
        wmean = sum(means) / len(means) if means else 0.0
        wstd  = (sum(s**2 for s in stds) / len(stds)) ** 0.5 if stds else 0.0
        return wmean, wstd, sum(abs(m) for m in means) / len(means) if means else 0.0

    wmean    = sum(w * m for w, m in zip(weights, means)) / tot
    wabs     = sum(w * abs(m) for w, m in zip(weights, means)) / tot
    # Pooled variance: E[X²] - (E[X])²
    ex2      = sum(w * (s**2 + m**2) for w, s, m in zip(weights, stds, means)) / tot
    wstd     = math.sqrt(max(0.0, ex2 - wmean**2))
    return wmean, wstd, wabs


def _group_by_range_bin(measured: list[dict]) -> dict[tuple, list[dict]]:
    """Group measured rows by (range_lo, range_hi) collapsing angle dimension."""
    groups: dict[tuple, list[dict]] = {}
    for r in measured:
        key = (_f(r["range_lo"]), _f(r["range_hi"]))
        groups.setdefault(key, []).append(r)
    return dict(sorted(groups.items()))


def _fit_poly(xs: list[float], ys: list[float], deg: int) -> np.ndarray:
    """Fit polynomial; fall back to zeros if too few points."""
    if len(xs) < deg + 1:
        return np.zeros(deg + 1)
    return np.polyfit(xs, ys, deg)


# ---------------------------------------------------------------------------
# Distributions CSV
# ---------------------------------------------------------------------------

def _write_distributions(stem: str, measured: list[dict], mm_models: Path) -> None:
    out_rows = []
    for slab_lo, slab_hi in SLABS_10M + [ALL_SLAB]:
        if slab_hi == 150:
            slab_rows = measured
        else:
            slab_rows = [r for r in measured
                         if _f(r["range_lo"]) >= slab_lo
                         and _f(r["range_hi"]) <= slab_hi]
        if not slab_rows:
            continue
        for _, dist_et, mean_col, std_col, _, _ in ERROR_TYPES:
            wm, ws, _ = _weighted_stats(slab_rows, mean_col, std_col)
            out_rows.append({
                "error_type":   dist_et,
                "dist_min":     int(slab_lo),
                "dist_max":     int(slab_hi),
                "distribution": "normal",
                "param1":       f"{wm:.6f}",
                "param2":       f"{ws:.6f}",
                "param3":       "",
            })

    out_path = mm_models / f"{stem}_distributions.csv"
    fieldnames = ["error_type", "dist_min", "dist_max", "distribution",
                  "param1", "param2", "param3"]
    with open(out_path, "w", newline="") as f:
        f.write(f"# {stem} - Binned error distributions (normal, from polar calibration)\n\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_slabs = len({(r["dist_min"], r["dist_max"]) for r in out_rows})
    print(f"    distributions: {n_slabs} slabs × {len(ERROR_TYPES)} error types")


# ---------------------------------------------------------------------------
# Main regression CSV
# ---------------------------------------------------------------------------

def _write_regression(stem: str, measured: list[dict], mm_models: Path) -> None:
    # Collapse angle dimension: for each (range_lo, range_hi) group,
    # compute gt_count-weighted stats across all angles.
    range_groups = _group_by_range_bin(measured)

    # Build per-range-bin arrays
    range_centers: list[float] = []
    stats_by_et: dict[str, dict] = {et: {"bias": [], "std": [], "abs": [], "var": []}
                                     for et, *_ in ERROR_TYPES}
    miss_rates: list[float] = []
    miss_weights: list[float] = []

    for (rlo, rhi), rows in range_groups.items():
        rc = (rlo + rhi) / 2.0
        range_centers.append(rc)
        w_tot = sum(_f(r.get("gt_count", 0)) for r in rows)

        for et, _, mean_col, std_col, _, _ in ERROR_TYPES:
            wm, ws, wa = _weighted_stats(rows, mean_col, std_col)
            stats_by_et[et]["bias"].append(wm)
            stats_by_et[et]["std"].append(ws)
            stats_by_et[et]["abs"].append(wa)
            stats_by_et[et]["var"].append(ws**2)

        # miss_rate: weighted average
        w_mr = sum(_f(r.get("gt_count", 0)) * _f(r.get("miss_rate", 0)) for r in rows)
        miss_weights.append(w_tot)
        miss_rates.append(w_mr / w_tot if w_tot > 0 else 0.0)

    xs = range_centers

    out_rows = []
    for et, _, _, _, unit, note in ERROR_TYPES:
        d = stats_by_et[et]
        # Linear fits
        lin_abs  = _fit_poly(xs, d["abs"],  1)  # [slope, intercept]
        lin_bias = _fit_poly(xs, d["bias"], 1)
        lin_std  = _fit_poly(xs, d["std"],  1)
        lin_var  = _fit_poly(xs, d["var"],  1)
        # Quadratic fits
        q_abs    = _fit_poly(xs, d["abs"],  2)  # [a, b, c]
        q_bias   = _fit_poly(xs, d["bias"], 2)
        q_std    = _fit_poly(xs, d["std"],  2)
        q_var    = _fit_poly(xs, d["var"],  2)

        out_rows.append({
            "error_type":     et,
            "intercept":      f"{lin_abs[1]:.6e}",
            "slope":          f"{lin_abs[0]:.6e}",
            "quad_a":         f"{q_abs[0]:.6e}",
            "quad_b":         f"{q_abs[1]:.6e}",
            "quad_c":         f"{q_abs[2]:.6e}",
            "bias_intercept": f"{lin_bias[1]:.6e}",
            "bias_slope":     f"{lin_bias[0]:.6e}",
            "bias_quad_a":    f"{q_bias[0]:.6e}",
            "bias_quad_b":    f"{q_bias[1]:.6e}",
            "bias_quad_c":    f"{q_bias[2]:.6e}",
            "std_intercept":  f"{lin_std[1]:.6e}",
            "std_slope":      f"{lin_std[0]:.6e}",
            "std_quad_a":     f"{q_std[0]:.6e}",
            "std_quad_b":     f"{q_std[1]:.6e}",
            "std_quad_c":     f"{q_std[2]:.6e}",
            "var_intercept":  f"{lin_var[1]:.6e}",
            "var_slope":      f"{lin_var[0]:.6e}",
            "var_quad_a":     f"{q_var[0]:.6e}",
            "var_quad_b":     f"{q_var[1]:.6e}",
            "var_quad_c":     f"{q_var[2]:.6e}",
            "unit":           unit,
            "notes":          note,
        })

    # miss_rate row (bias/std columns left empty as in the reference files)
    lin_mr = _fit_poly(xs, miss_rates, 1)
    q_mr   = _fit_poly(xs, miss_rates, 2)
    out_rows.append({
        "error_type":     "miss_rate",
        "intercept":      f"{lin_mr[1]:.6e}",
        "slope":          f"{lin_mr[0]:.6e}",
        "quad_a":         f"{q_mr[0]:.6e}",
        "quad_b":         f"{q_mr[1]:.6e}",
        "quad_c":         f"{q_mr[2]:.6e}",
        "bias_intercept": "", "bias_slope": "",
        "bias_quad_a":    "", "bias_quad_b":    "", "bias_quad_c":    "",
        "std_intercept":  "", "std_slope":      "",
        "std_quad_a":     "", "std_quad_b":     "", "std_quad_c":     "",
        "var_intercept":  "", "var_slope":      "",
        "var_quad_a":     "", "var_quad_b":     "", "var_quad_c":     "",
        "unit":           "probability",
        "notes":          "Miss rate",
    })

    out_path = mm_models / f"{stem}.csv"
    comment_lines = [
        f"# {stem}",
        "# MAE linear: abs_error = intercept + slope * distance",
        "# MAE quadratic: abs_error = quad_a * d^2 + quad_b * d + quad_c",
        "# Std linear: std = std_intercept + std_slope * distance",
        "# Std quadratic: std = std_quad_a * d^2 + std_quad_b * d + std_quad_c",
        "# Var linear: var = var_intercept + var_slope * distance",
        "# Var quadratic: var = var_quad_a * d^2 + var_quad_b * d + var_quad_c",
        "# Bias linear: signed_bias = bias_intercept + bias_slope * distance",
        "# Bias quadratic: signed_bias = bias_quad_a * d^2 + bias_quad_b * d + bias_quad_c",
        "# MSE (for R matrix) = bias^2 + variance (bias-variance decomposition)",
    ]
    with open(out_path, "w", newline="") as f:
        for line in comment_lines:
            f.write(line + "\n")
        writer = csv.DictWriter(f, fieldnames=REGRESSION_HEADER)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"    regression:    {out_path.name}  ({len(range_groups)} range bins fitted)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_sensor_model(stem: str, mm_models: Path) -> None:
    polar_path = mm_models / f"{stem}_polar_calibration.csv"
    if not polar_path.exists():
        print(f"  [skip] not found: {polar_path.name}")
        return

    with open(polar_path, newline="") as f:
        measured = [
            r for r in csv.DictReader(f)
            if r.get("source") == "measured" and _f(r.get("gt_count", 0)) > 0
        ]

    if not measured:
        print(f"  [skip] no measured rows in {polar_path.name}")
        return

    _write_distributions(stem, measured, mm_models)
    _write_regression(stem, measured, mm_models)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm-root", type=Path, default=DEFAULT_MM_ROOT,
                    help=f"mmdetection3d root (default: {DEFAULT_MM_ROOT})")
    args = ap.parse_args()

    mm_models = args.mm_root / "src/data/sensor_models"
    if not mm_models.exists():
        raise SystemExit(f"sensor_models dir not found: {mm_models}")

    print(f"Generating sensor models from {mm_models}")
    for stem in MODELS:
        print(f"\n  {stem}")
        generate_sensor_model(stem, mm_models)
    print("\nDone.")


if __name__ == "__main__":
    main()
