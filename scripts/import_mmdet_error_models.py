#!/usr/bin/env python3
"""
Import error models from mmdetection3d work_dirs into CMR sensor_models format.

Usage:
  python scripts/import_mmdet_error_models.py

Reads from (prefers *_full* when present for overall_averages.csv, else *_mini*):
  ../mmdetection3d/work_dirs/bevfusion_{full|mini}
  ../mmdetection3d/work_dirs/centerpoint_{full|mini}
  ../mmdetection3d/work_dirs/detr3d_{full|mini}

Writes to:
  src/data/sensor_models/bev_fusion.csv, bev_fusion_distributions.csv
  src/data/sensor_models/centerpoint.csv, centerpoint_distributions.csv
  src/data/sensor_models/detr3d.csv, detr3d_distributions.csv

Ignores "mini" in the source dir name (small-dataset smoke test).
"""

import re
import csv
import math
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
MMDET_BASE = REPO_ROOT.parent / "mmdetection3d" / "work_dirs"
CMR_MODELS = REPO_ROOT / "src" / "data" / "sensor_models"

# Map mmdet regression names to CMR CSV error_type (first CSV: regression).
# Each tuple: (linear_names, quad_names, bias_linear_names, bias_quad_names, CMR type, unit, notes).
# New format uses x/y/z.
REGRESSION_MAP = [
    (["distal_linear", "x_linear"], ["distal_quadratic", "x_quadratic"],
     ["distal_bias_linear", "x_bias_linear"], ["distal_bias_quadratic", "x_bias_quadratic"], "radial", "meters", "Range error"),
    (["perp_linear", "y_linear"], ["perp_quadratic", "y_quadratic"],
     ["perp_bias_linear", "y_bias_linear"], ["perp_bias_quadratic", "y_bias_quadratic"], "lateral", "meters", "Cross-range error"),
    (["height_linear", "z_linear"], ["height_quadratic", "z_quadratic"],
     ["z_bias_linear"], ["z_bias_quadratic"], "vertical", "meters", "Elevation error"),
    (["yaw_linear"], ["yaw_quadratic"],
     ["yaw_bias_linear"], ["yaw_bias_quadratic"], "yaw", "radians", "Heading angle"),
    (["width_linear"], ["width_quadratic"],
     ["width_bias_linear"], ["width_bias_quadratic"], "width", "meters", "Box width"),
    (["length_linear"], ["length_quadratic"],
     ["length_bias_linear"], ["length_bias_quadratic"], "length", "meters", "Box length"),
    (["box_height_linear"], ["box_height_quadratic"],
     ["box_height_bias_linear"], ["box_height_bias_quadratic"], "box_height", "meters", "Box height"),
    (["missed_rate_linear"], ["missed_rate_quadratic"],
     [], [], "miss_rate", "probability", "Miss rate"),
]

# Map binned_errors columns to CMR distribution error_type.
# (bias_cols, bias_std_cols, CMR type) — use signed error stats for proper distributions.
# bias = mean of signed errors (the systematic offset), bias_std = std of signed errors (the noise).
# This gives normal(bias, bias_std) which is the actual error distribution, not the folded one.
BINNED_MAP = [
    (["distal_bias", "x_bias"], ["distal_bias_std", "x_bias_std"], "distal"),
    (["perp_bias", "y_bias"], ["perp_bias_std", "y_bias_std"], "perpendicular"),
    (["z_bias"], ["z_bias_std"], "height"),
    (["yaw_bias"], ["yaw_bias_std"], "yaw"),
    (["width_bias"], ["width_bias_std"], "width"),
    (["length_bias"], ["length_bias_std"], "length"),
    (["box_height_bias"], ["box_height_bias_std"], "box_height"),
]

# Fallback: unsigned error columns (for backwards compat with old binned_errors.csv without bias)
BINNED_MAP_UNSIGNED = [
    (["distal_mean", "x_mean"], ["distal_std", "x_std"], "distal"),
    (["perp_mean", "y_mean"], ["perp_std", "y_std"], "perpendicular"),
    (["height_mean", "z_mean"], ["height_std", "z_std"], "height"),
    (["yaw_mean"], ["yaw_std"], "yaw"),
    (["width_mean"], ["width_std"], "width"),
    (["length_mean"], ["length_std"], "length"),
    (["box_height_mean"], ["box_height_std"], "box_height"),
]

# overall_averages.csv metric name -> CMR distribution error_type
OVERALL_AVERAGES_MAP = [
    ("x", "distal"),
    ("y", "perpendicular"),
    ("z", "height"),
    ("yaw", "yaw"),
    ("width", "width"),
    ("length", "length"),
    ("box_height", "box_height"),
]


def parse_regression_txt(path: Path) -> dict:
    """Parse regression_models.txt; return dict keyed by mmdet name -> regression params.
    
    Returns:
        dict: {
            'x_linear': (intercept, slope),
            'x_quadratic': (a, b, c),  # error = a*d^2 + b*d + c
            ...
        }
    """
    text = path.read_text()
    out = {}
    
    # Linear: "x_linear:  0.003281 * d + 0.031255"
    linear_pattern = re.compile(
        r"(\w+_linear):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*d\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )
    for m in linear_pattern.finditer(text):
        name, slope, intercept = m.group(1), float(m.group(2)), float(m.group(3))
        out[name] = (intercept, slope)
    
    # Quadratic: "x_quadratic: -2.126893e-06*d^2 + 3.401052e-03*d + 2.996122e-02"
    quad_pattern = re.compile(
        r"(\w+_quadratic):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*d\^2\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*d\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )
    for m in quad_pattern.finditer(text):
        name = m.group(1)
        a, b, c = float(m.group(2)), float(m.group(3)), float(m.group(4))
        out[name] = (a, b, c)  # error = a*d^2 + b*d + c
    
    return out


def write_regression_csv(model_name: str, reg: dict, description: str):
    """Write model_name.csv in CMR format with linear, quadratic, and bias coefficients.

    CSV format:
        error_type, intercept, slope, quad_a, quad_b, quad_c, bias_intercept, bias_slope,
        bias_quad_a, bias_quad_b, bias_quad_c, unit, notes

    Linear: abs_error = intercept + slope * d
    Quadratic: abs_error = quad_a * d^2 + quad_b * d + quad_c
    Bias linear: signed_bias = bias_intercept + bias_slope * d
    Bias quadratic: signed_bias = bias_quad_a * d^2 + bias_quad_b * d + bias_quad_c
    """
    path = CMR_MODELS / f"{model_name}.csv"
    rows = []
    for linear_names, quad_names, bias_lin_names, bias_quad_names, cmr_type, unit, notes in REGRESSION_MAP:
        # Get linear coefficients
        intercept, slope = None, None
        for name in linear_names:
            if name in reg:
                intercept, slope = reg[name]
                break
        if intercept is None:
            continue

        # Get quadratic coefficients (optional)
        quad_a, quad_b, quad_c = "", "", ""
        for name in quad_names:
            if name in reg:
                quad_a, quad_b, quad_c = reg[name]
                break

        # Get bias linear coefficients (optional)
        bias_intercept, bias_slope = "", ""
        for name in bias_lin_names:
            if name in reg:
                bias_intercept, bias_slope = reg[name]
                break

        # Get bias quadratic coefficients (optional)
        bias_quad_a, bias_quad_b, bias_quad_c = "", "", ""
        for name in bias_quad_names:
            if name in reg:
                bias_quad_a, bias_quad_b, bias_quad_c = reg[name]
                break

        rows.append({
            "error_type": cmr_type,
            "intercept": intercept,
            "slope": slope,
            "quad_a": quad_a,
            "quad_b": quad_b,
            "quad_c": quad_c,
            "bias_intercept": bias_intercept,
            "bias_slope": bias_slope,
            "bias_quad_a": bias_quad_a,
            "bias_quad_b": bias_quad_b,
            "bias_quad_c": bias_quad_c,
            "unit": unit,
            "notes": notes
        })

    fieldnames = ["error_type", "intercept", "slope", "quad_a", "quad_b", "quad_c",
                  "bias_intercept", "bias_slope", "bias_quad_a", "bias_quad_b", "bias_quad_c",
                  "unit", "notes"]
    with open(path, "w", newline="") as f:
        f.write(f"# {description}\n")
        f.write("# Linear model: abs_error = intercept + slope * distance\n")
        f.write("# Quadratic model: abs_error = quad_a * d^2 + quad_b * d + quad_c\n")
        f.write("# Bias linear: signed_bias = bias_intercept + bias_slope * distance\n")
        f.write("# Bias quadratic: signed_bias = bias_quad_a * d^2 + bias_quad_b * d + bias_quad_c\n")
        f.write("# For covariance: std = 1.253 * abs_error (half-normal to std conversion)\n\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path}")


def parse_binned_csv(path: Path) -> tuple:
    """Parse binned_errors.csv; return (rows, has_bias) where has_bias indicates signed error columns exist."""
    rows = []
    has_bias = False
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                bin_lo = float(row["bin_lo"])
                bin_hi = float(row["bin_hi"])
            except (KeyError, ValueError):
                continue
            # Check if bias columns exist
            if not has_bias and "x_bias" in row:
                try:
                    float(row["x_bias"])
                    has_bias = True
                except (ValueError, TypeError):
                    pass
            # Accept either bias or unsigned columns
            binned_map = BINNED_MAP if has_bias else BINNED_MAP_UNSIGNED
            has_any = False
            for mean_cols, _std_cols, _ in binned_map:
                for col in mean_cols:
                    if col in row:
                        try:
                            float(row[col])
                            has_any = True
                            break
                        except (ValueError, TypeError):
                            pass
                if has_any:
                    break
            if not has_any:
                continue
            rows.append(row)
    return rows, has_bias


def parse_overall_averages_csv(path: Path) -> list:
    """Parse overall_averages.csv; return list of dicts with error_type, mean, std for distribution use."""
    out = []
    with open(path) as f:
        r = csv.DictReader(f)
        metric_to_cmr = dict(OVERALL_AVERAGES_MAP)
        for row in r:
            metric = row.get("metric", "").strip()
            if metric not in metric_to_cmr:
                continue
            try:
                mu = float(row["mean"])
                sig = float(row["std"])
            except (ValueError, KeyError):
                continue
            if math.isnan(mu) or math.isnan(sig):
                continue
            out.append({
                "error_type": metric_to_cmr[metric],
                "mean": mu,
                "std": max(sig, 1e-6),
            })
    return out


def write_distributions_csv(
    model_name: str, binned_rows: list, description: str,
    overall_bin: list = None, overall_dist_max: int = 150, has_bias: bool = False
):
    """Write model_name_distributions.csv from binned error stats.

    When has_bias=True, uses signed error stats: normal(bias, bias_std) — the actual error distribution.
    When has_bias=False, falls back to unsigned: normal(mean_abs, std_abs) — the folded distribution.
    If overall_bin is provided (from overall_averages.csv), add one 0..overall_dist_max bin per type."""
    path = CMR_MODELS / f"{model_name}_distributions.csv"
    binned_map = BINNED_MAP if has_bias else BINNED_MAP_UNSIGNED
    out_rows = []
    for row in binned_rows:
        bin_lo = float(row["bin_lo"])
        bin_hi = float(row["bin_hi"])
        for mean_cols, std_cols, error_type in binned_map:
            mu = sig = None
            for mean_col, std_col in zip(mean_cols, std_cols):
                try:
                    if mean_col in row and std_col in row:
                        mu = float(row[mean_col])
                        sig = float(row[std_col])
                        break
                except (ValueError, KeyError):
                    continue
            if mu is None or math.isnan(mu) or math.isnan(sig):
                continue
            sig = max(sig, 1e-6)
            out_rows.append({
                "error_type": error_type,
                "dist_min": int(bin_lo),
                "dist_max": int(bin_hi),
                "distribution": "normal",
                "param1": round(mu, 6),
                "param2": round(sig, 6),
                "param3": "",
            })
    if overall_bin:
        for entry in overall_bin:
            out_rows.append({
                "error_type": entry["error_type"],
                "dist_min": 0,
                "dist_max": overall_dist_max,
                "distribution": "normal",
                "param1": round(entry["mean"], 6),
                "param2": round(entry["std"], 6),
                "param3": "",
            })
    # CMR expects one row per (error_type, dist_min, dist_max)
    with open(path, "w", newline="") as f:
        f.write(f"# {description} - Binned error distributions (normal from mean/std)\n\n")
        w = csv.DictWriter(f, fieldnames=["error_type", "dist_min", "dist_max", "distribution", "param1", "param2", "param3"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Wrote {path}")


def import_one(source_dir: Path, model_name: str, description: str):
    """Import one mmdet work_dir into CMR sensor_models."""
    reg_path = source_dir / "error_analysis" / "regression_models.txt"
    binned_path = source_dir / "error_analysis" / "binned_errors.csv"
    overall_path = source_dir / "error_analysis" / "overall_averages.csv"
    if not reg_path.exists():
        print(f"  Skip {model_name}: missing {reg_path}")
        return
    if not binned_path.exists():
        print(f"  Skip {model_name}: missing {binned_path}")
        return
    reg = parse_regression_txt(reg_path)
    write_regression_csv(model_name, reg, description)
    binned_rows, has_bias = parse_binned_csv(binned_path)
    if has_bias:
        print(f"  Using signed error stats (bias/bias_std) for distributions")
    overall_bin = None
    if overall_path.exists():
        overall_bin = parse_overall_averages_csv(overall_path)
        print(f"  Using overall_averages.csv ({len(overall_bin)} metrics) for 0–150 m bin")
    write_distributions_csv(model_name, binned_rows, description, overall_bin=overall_bin, has_bias=has_bias)


def main():
    CMR_MODELS.mkdir(parents=True, exist_ok=True)
    # Prefer *_full* work_dirs when present (they include overall_averages.csv); else use *_mini*.
    candidates = [
        ("bev_fusion", "BEV Fusion (lidar+camera) - from mmdetection3d", "bevfusion_full", "bevfusion_mini"),
        ("centerpoint", "CenterPoint (lidar only) - from mmdetection3d", "centerpoint_full", "centerpoint_mini"),
        ("detr3d", "DETR3D (camera only) - from mmdetection3d", "detr3d_full", "detr3d_mini"),
    ]
    for model_name, description, full_dir, mini_dir in candidates:
        source_dir = MMDET_BASE / full_dir
        if not source_dir.exists():
            source_dir = MMDET_BASE / mini_dir
        print(f"Importing {model_name} from {source_dir}...")
        import_one(source_dir, model_name, description)
    print("Done.")


if __name__ == "__main__":
    main()
