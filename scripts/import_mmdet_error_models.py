#!/usr/bin/env python3
"""
Import error models from mmdetection3d work_dirs into CMR sensor_models format.

Usage:
  python scripts/import_mmdet_error_models.py

Reads from (prefers *_100m_full* for polar data, else *_full*, else *_mini*):
  ../mmdetection3d/work_dirs/bevfusion_100m_full
  ../mmdetection3d/work_dirs/centerpoint_100m_full
  ../mmdetection3d/work_dirs/detr3d_100m_full

Writes to:
  src/data/sensor_models/bev_fusion.csv, bev_fusion_distributions.csv, bev_fusion_polar_distributions.csv
  src/data/sensor_models/centerpoint.csv, centerpoint_distributions.csv, centerpoint_polar_distributions.csv
  src/data/sensor_models/detr3d.csv, detr3d_distributions.csv, detr3d_polar_distributions.csv
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
# Each tuple: (mae_lin, mae_quad, bias_lin, bias_quad, std_lin, std_quad, var_lin, var_quad, CMR type, unit, notes).
REGRESSION_MAP = [
    (["distal_linear", "x_linear"], ["distal_quadratic", "x_quadratic"],
     ["distal_bias_linear", "x_bias_linear"], ["distal_bias_quadratic", "x_bias_quadratic"],
     ["distal_std_linear", "x_std_linear"], ["distal_std_quadratic", "x_std_quadratic"],
     ["distal_var_linear", "x_var_linear"], ["distal_var_quadratic", "x_var_quadratic"],
     "radial", "meters", "Range error"),
    (["perp_linear", "y_linear"], ["perp_quadratic", "y_quadratic"],
     ["perp_bias_linear", "y_bias_linear"], ["perp_bias_quadratic", "y_bias_quadratic"],
     ["perp_std_linear", "y_std_linear"], ["perp_std_quadratic", "y_std_quadratic"],
     ["perp_var_linear", "y_var_linear"], ["perp_var_quadratic", "y_var_quadratic"],
     "lateral", "meters", "Cross-range error"),
    (["height_linear", "z_linear"], ["height_quadratic", "z_quadratic"],
     ["z_bias_linear"], ["z_bias_quadratic"],
     ["z_std_linear"], ["z_std_quadratic"],
     ["z_var_linear"], ["z_var_quadratic"],
     "vertical", "meters", "Elevation error"),
    (["yaw_linear"], ["yaw_quadratic"],
     ["yaw_bias_linear"], ["yaw_bias_quadratic"],
     ["yaw_std_linear"], ["yaw_std_quadratic"],
     ["yaw_var_linear"], ["yaw_var_quadratic"],
     "yaw", "radians", "Heading angle"),
    (["width_linear"], ["width_quadratic"],
     ["width_bias_linear"], ["width_bias_quadratic"],
     ["width_std_linear"], ["width_std_quadratic"],
     ["width_var_linear"], ["width_var_quadratic"],
     "width", "meters", "Box width"),
    (["length_linear"], ["length_quadratic"],
     ["length_bias_linear"], ["length_bias_quadratic"],
     ["length_std_linear"], ["length_std_quadratic"],
     ["length_var_linear"], ["length_var_quadratic"],
     "length", "meters", "Box length"),
    (["box_height_linear"], ["box_height_quadratic"],
     ["box_height_bias_linear"], ["box_height_bias_quadratic"],
     ["box_height_std_linear"], ["box_height_std_quadratic"],
     ["box_height_var_linear"], ["box_height_var_quadratic"],
     "box_height", "meters", "Box height"),
    (["missed_rate_linear"], ["missed_rate_quadratic"],
     [], [], [], [], [], [],
     "miss_rate", "probability", "Miss rate"),
]

# Map binned_errors columns to CMR distribution error_type (signed error stats).
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
    ("distal", "distal"),
    ("perp", "perpendicular"),
    ("z", "height"),
    ("yaw", "yaw"),
    ("width", "width"),
    ("length", "length"),
    ("box_height", "box_height"),
]

# Map polar_binned_errors.csv columns to CMR polar distribution error_type.
# (mean_col, std_col, CMR_type) — polar CSV has both _mean (MAE) and _std columns.
POLAR_MAP = [
    ("distal_mean", "distal_std", "distal"),
    ("perp_mean", "perp_std", "perpendicular"),
    ("z_mean", "z_std", "height"),
    ("yaw_mean", "yaw_std", "yaw"),
    ("width_mean", "width_std", "width"),
    ("length_mean", "length_std", "length"),
    ("box_height_mean", "box_height_std", "box_height"),
]


def parse_regression_txt(path: Path) -> dict:
    """Parse regression_models.txt; return dict keyed by mmdet name -> regression params."""
    text = path.read_text()
    out = {}

    linear_pattern = re.compile(
        r"(\w+_linear):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*d\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )
    for m in linear_pattern.finditer(text):
        name, slope, intercept = m.group(1), float(m.group(2)), float(m.group(3))
        out[name] = (intercept, slope)

    quad_pattern = re.compile(
        r"(\w+_quadratic):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*d\^2\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*d\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )
    for m in quad_pattern.finditer(text):
        name = m.group(1)
        a, b, c = float(m.group(2)), float(m.group(3)), float(m.group(4))
        out[name] = (a, b, c)

    return out


def _lookup_reg(reg, names):
    """Look up regression coefficients by trying multiple key names."""
    for name in names:
        if name in reg:
            return reg[name]
    return None


def write_regression_csv(model_name: str, reg: dict, description: str):
    """Write model_name.csv with MAE, std, bias regression coefficients."""
    path = CMR_MODELS / f"{model_name}.csv"
    rows = []
    for entry in REGRESSION_MAP:
        (mae_lin_names, mae_quad_names, bias_lin_names, bias_quad_names,
         std_lin_names, std_quad_names, var_lin_names, var_quad_names,
         cmr_type, unit, notes) = entry

        # MAE linear (required)
        mae_lin = _lookup_reg(reg, mae_lin_names)
        if mae_lin is None:
            continue
        intercept, slope = mae_lin

        # MAE quadratic (optional)
        mae_quad = _lookup_reg(reg, mae_quad_names)
        quad_a, quad_b, quad_c = mae_quad if mae_quad else ("", "", "")

        # Bias linear (optional)
        bias_lin = _lookup_reg(reg, bias_lin_names)
        bias_intercept, bias_slope = bias_lin if bias_lin else ("", "")

        # Bias quadratic (optional)
        bias_quad = _lookup_reg(reg, bias_quad_names)
        bias_quad_a, bias_quad_b, bias_quad_c = bias_quad if bias_quad else ("", "", "")

        # Std linear (optional)
        std_lin = _lookup_reg(reg, std_lin_names)
        std_intercept, std_slope = std_lin if std_lin else ("", "")

        # Std quadratic (optional)
        std_quad = _lookup_reg(reg, std_quad_names)
        std_quad_a, std_quad_b, std_quad_c = std_quad if std_quad else ("", "", "")

        # Variance linear (optional — for MSE-based R matrix)
        var_lin = _lookup_reg(reg, var_lin_names)
        var_intercept, var_slope = var_lin if var_lin else ("", "")

        # Variance quadratic (optional)
        var_quad = _lookup_reg(reg, var_quad_names)
        var_quad_a, var_quad_b, var_quad_c = var_quad if var_quad else ("", "", "")

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
            "std_intercept": std_intercept,
            "std_slope": std_slope,
            "std_quad_a": std_quad_a,
            "std_quad_b": std_quad_b,
            "std_quad_c": std_quad_c,
            "var_intercept": var_intercept,
            "var_slope": var_slope,
            "var_quad_a": var_quad_a,
            "var_quad_b": var_quad_b,
            "var_quad_c": var_quad_c,
            "unit": unit,
            "notes": notes
        })

    fieldnames = ["error_type", "intercept", "slope", "quad_a", "quad_b", "quad_c",
                  "bias_intercept", "bias_slope", "bias_quad_a", "bias_quad_b", "bias_quad_c",
                  "std_intercept", "std_slope", "std_quad_a", "std_quad_b", "std_quad_c",
                  "var_intercept", "var_slope", "var_quad_a", "var_quad_b", "var_quad_c",
                  "unit", "notes"]
    with open(path, "w", newline="") as f:
        f.write(f"# {description}\n")
        f.write("# MAE linear: abs_error = intercept + slope * distance\n")
        f.write("# MAE quadratic: abs_error = quad_a * d^2 + quad_b * d + quad_c\n")
        f.write("# Std linear: std = std_intercept + std_slope * distance\n")
        f.write("# Std quadratic: std = std_quad_a * d^2 + std_quad_b * d + std_quad_c\n")
        f.write("# Var linear: var = var_intercept + var_slope * distance\n")
        f.write("# Var quadratic: var = var_quad_a * d^2 + var_quad_b * d + var_quad_c\n")
        f.write("# Bias linear: signed_bias = bias_intercept + bias_slope * distance\n")
        f.write("# Bias quadratic: signed_bias = bias_quad_a * d^2 + bias_quad_b * d + bias_quad_c\n")
        f.write("# MSE (for R matrix) = bias^2 + variance (bias-variance decomposition)\n\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path}")


def parse_binned_csv(path: Path) -> tuple:
    """Parse binned_errors.csv; return (rows, has_bias)."""
    rows = []
    has_bias = False
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                float(row["bin_lo"])
                float(row["bin_hi"])
            except (KeyError, ValueError):
                continue
            if not has_bias and "x_bias" in row:
                try:
                    float(row["x_bias"])
                    has_bias = True
                except (ValueError, TypeError):
                    pass
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
    """Parse overall_averages.csv; return list of dicts with error_type, mean, std."""
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
    """Write model_name_distributions.csv from distance-binned error stats."""
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
    # Compute overall bin from per-bin data (count-weighted signed stats)
    # This ensures the overall bin uses the same bias/bias_std as per-bin rows,
    # rather than the unsigned stats from overall_averages.csv.
    for mean_cols, std_cols, error_type in binned_map:
        weighted_mu = 0.0
        weighted_var = 0.0
        total_count = 0
        for row in binned_rows:
            try:
                count = int(row.get("count", 0)) - int(row.get("missed", 0))
            except (ValueError, KeyError):
                continue
            if count <= 0:
                continue
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
            weighted_mu += count * mu
            weighted_var += count * (sig**2 + mu**2)
            total_count += count
        if total_count > 0:
            overall_mu = weighted_mu / total_count
            overall_var = weighted_var / total_count - overall_mu**2
            overall_sig = max(math.sqrt(max(0, overall_var)), 1e-6)
            out_rows.append({
                "error_type": error_type,
                "dist_min": 0,
                "dist_max": overall_dist_max,
                "distribution": "normal",
                "param1": round(overall_mu, 6),
                "param2": round(overall_sig, 6),
                "param3": "",
            })
    with open(path, "w", newline="") as f:
        f.write(f"# {description} - Binned error distributions (normal from mean/std)\n\n")
        w = csv.DictWriter(f, fieldnames=["error_type", "dist_min", "dist_max", "distribution", "param1", "param2", "param3"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Wrote {path}")


def parse_polar_binned_csv(path: Path) -> list:
    """Parse polar_binned_errors.csv; return list of row dicts.

    Columns: range_lo, range_hi, angle_lo, angle_hi, gt_count, missed,
             miss_rate, source, x_mean, y_mean, z_mean, distal_mean, perp_mean,
             yaw_mean, width_mean, length_mean, box_height_mean
    """
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                float(row["range_lo"])
                float(row["range_hi"])
                float(row["angle_lo"])
                float(row["angle_hi"])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def write_polar_distributions_csv(model_name: str, polar_rows: list, description: str):
    """Write model_name_polar_distributions.csv from polar-binned error stats.

    Output format includes angle_min/angle_max columns for 2D (distance+angle) lookup.
    Uses _mean (MAE) and _std columns from the polar CSV.
    Bins with source='fallback', gt_count=0, or miss_rate>=1.0 produce miss_detection rows.
    """
    path = CMR_MODELS / f"{model_name}_polar_distributions.csv"
    out_rows = []

    for row in polar_rows:
        range_lo = float(row["range_lo"])
        range_hi = float(row["range_hi"])
        angle_lo = float(row["angle_lo"])
        angle_hi = float(row["angle_hi"])
        gt_count = int(row["gt_count"])
        miss_rate = float(row["miss_rate"])
        source = row.get("source", "measured")

        is_dead_zone = (source == "fallback" or gt_count == 0 or miss_rate >= 1.0)

        # Write miss_detection entry for this bin
        out_rows.append({
            "error_type": "miss_detection",
            "dist_min": range_lo,
            "dist_max": range_hi,
            "angle_min": angle_lo,
            "angle_max": angle_hi,
            "distribution": "constant",
            "param1": round(1.0 if is_dead_zone else miss_rate, 6),
            "param2": "",
            "param3": "",
        })

        # Skip error distributions for dead zones
        if is_dead_zone:
            continue

        # Write error distributions for each error type
        for mean_col, std_col, error_type in POLAR_MAP:
            try:
                mae = float(row[mean_col])
                std = float(row[std_col])
            except (KeyError, ValueError, TypeError):
                continue
            if math.isnan(mae) or math.isnan(std):
                continue
            # Store as normal distribution with mu=0, sigma=std from the evaluation.
            # MAE stored in param3 for reference (GPEM covariance uses it via MAE_TO_STD).
            std = max(std, 1e-6)
            mae = max(mae, 1e-6)
            out_rows.append({
                "error_type": error_type,
                "dist_min": range_lo,
                "dist_max": range_hi,
                "angle_min": angle_lo,
                "angle_max": angle_hi,
                "distribution": "normal",
                "param1": 0.0,
                "param2": round(std, 6),
                "param3": round(mae, 6),
            })

    fieldnames = ["error_type", "dist_min", "dist_max", "angle_min", "angle_max",
                  "distribution", "param1", "param2", "param3"]
    with open(path, "w", newline="") as f:
        f.write(f"# {description} - Polar binned error distributions (angle+distance)\n")
        f.write("# normal: param1=mu (0), param2=std, param3=mae (for GPEM covariance)\n")
        f.write("# constant for miss_detection: param1=miss_rate\n\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Wrote {path} ({len(out_rows)} rows)")


def import_one(source_dir: Path, model_name: str, description: str):
    """Import one mmdet work_dir into CMR sensor_models."""
    reg_path = source_dir / "error_analysis" / "regression_models.txt"
    binned_path = source_dir / "error_analysis" / "binned_errors.csv"
    overall_path = source_dir / "error_analysis" / "overall_averages.csv"
    polar_path = source_dir / "error_analysis" / "polar_binned_errors.csv"

    if not reg_path.exists():
        print(f"  Skip {model_name}: missing {reg_path}")
        return
    if not binned_path.exists():
        print(f"  Skip {model_name}: missing {binned_path}")
        return

    # Regression CSV (linear/quadratic coefficients)
    reg = parse_regression_txt(reg_path)
    write_regression_csv(model_name, reg, description)

    # Distance-only binned distributions
    binned_rows, has_bias = parse_binned_csv(binned_path)
    if has_bias:
        print(f"  Using signed error stats (bias/bias_std) for distributions")
    overall_bin = None
    if overall_path.exists():
        overall_bin = parse_overall_averages_csv(overall_path)
        print(f"  Using overall_averages.csv ({len(overall_bin)} metrics) for 0-150m bin")
    write_distributions_csv(model_name, binned_rows, description,
                            overall_bin=overall_bin, has_bias=has_bias)

    # Polar (angle+distance) distributions
    if polar_path.exists():
        polar_rows = parse_polar_binned_csv(polar_path)
        print(f"  Found polar_binned_errors.csv ({len(polar_rows)} bins)")
        write_polar_distributions_csv(model_name, polar_rows, description)
    else:
        print(f"  No polar_binned_errors.csv found, skipping polar distributions")


def main():
    CMR_MODELS.mkdir(parents=True, exist_ok=True)
    # Prefer *_100m_full* (polar + extended range), else *_full*, else *_mini*.
    candidates = [
        ("bev_fusion", "BEV Fusion (lidar+camera) 100m - from mmdetection3d",
         "bevfusion_100m_full", "bevfusion_full", "bevfusion_mini"),
        ("centerpoint", "CenterPoint (lidar only) 100m - from mmdetection3d",
         "centerpoint_100m_full", "centerpoint_full", "centerpoint_mini"),
        ("detr3d", "DETR3D (camera only) 100m - from mmdetection3d",
         "detr3d_100m_full", "detr3d_full", "detr3d_mini"),
    ]
    for entry in candidates:
        model_name, description = entry[0], entry[1]
        source_dirs = entry[2:]
        source_dir = None
        for d in source_dirs:
            candidate = MMDET_BASE / d
            if candidate.exists():
                source_dir = candidate
                break
        if source_dir is None:
            print(f"Skipping {model_name}: no source directory found")
            continue
        print(f"Importing {model_name} from {source_dir}...")
        import_one(source_dir, model_name, description)
    print("Done.")


if __name__ == "__main__":
    main()
