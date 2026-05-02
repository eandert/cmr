#!/usr/bin/env python3
"""
Import localizer evaluation data from pre-computed binned CSVs.

Reads velocity-binned error statistics and regression models, producing:
1. Regression coefficients CSV — MAE, std, variance, bias regressions
2. Velocity-binned distributions CSV — for error sampling

Usage:
    python import_localizer_binned.py <binned_csv> <regression_txt> <output_name>

Example:
    python import_localizer_binned.py \
        path/to/hd_map_kiss_binned_errors.csv \
        path/to/hd_map_kiss_regression_models.txt \
        kiss_icp
"""

import re
import sys
import csv
import math
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "src" / "data" / "sensor_models"

# Error types to process
ERROR_TYPES = {
    'longitudinal': ('longitudinal', 'meters', 'Longitudinal (forward) position error'),
    'lateral': ('lateral', 'meters', 'Lateral (side) position error'),
    'vertical': ('vertical', 'meters', 'Vertical (up/down) position error'),
    'roll': ('roll', 'radians', 'Roll angle error'),
    'pitch': ('pitch', 'radians', 'Pitch angle error'),
    'yaw': ('yaw', 'radians', 'Yaw angle error'),
}


def parse_regression_txt(path: Path) -> dict:
    """Parse regression_models.txt; return dict keyed by name -> coefficients.

    Handles:
        name_linear:  slope * v + intercept
        name_quadratic:  a*v^2 + b*v + c
    """
    text = path.read_text()
    out = {}

    # Linear: "name_linear:  slope * v + intercept"
    linear_pattern = re.compile(
        r"(\w+_linear):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*v\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )
    for m in linear_pattern.finditer(text):
        name, slope, intercept = m.group(1), float(m.group(2)), float(m.group(3))
        out[name] = (intercept, slope)

    # Quadratic: "name_quadratic:  a*v^2 + b*v + c"
    quad_pattern = re.compile(
        r"(\w+_quadratic):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*v\^2\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*\s*v\s*\+\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )
    for m in quad_pattern.finditer(text):
        name = m.group(1)
        a, b, c = float(m.group(2)), float(m.group(3)), float(m.group(4))
        out[name] = (a, b, c)

    return out


def process_binned_data(binned_csv: str, regression_txt: str, output_name: str):
    """Process binned localizer data and regression models into CMR format."""
    print(f"Reading binned data: {binned_csv}")
    rows = []
    with open(binned_csv) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            try:
                if int(row['count']) > 0 and not math.isnan(float(row['longitudinal_std'])):
                    rows.append(row)
            except (ValueError, KeyError):
                continue
    print(f"  Valid bins: {len(rows)}")
    vel_mins = [float(r['velocity_min']) for r in rows]
    vel_maxs = [float(r['velocity_max']) for r in rows]
    print(f"  Velocity range: {min(vel_mins):.0f} - {max(vel_maxs):.0f} m/s")

    has_bias = 'longitudinal_bias' in columns
    print(f"  Has bias columns: {has_bias}")

    # Convert to arrays for convenience
    df = rows  # list of dicts

    # Parse regression models if available
    reg = {}
    reg_path = Path(regression_txt)
    if reg_path.exists():
        print(f"Reading regressions: {regression_txt}")
        reg = parse_regression_txt(reg_path)
        print(f"  Found {len(reg)} regression entries")

    # --- Regression CSV ---
    regression_rows = []
    for error_key, (out_name, unit, notes) in ERROR_TYPES.items():
        row = {'error_type': out_name, 'unit': unit, 'notes': notes}

        # MAE regression (intercept + slope * v)
        mae_lin = reg.get(f'{error_key}_linear')
        if mae_lin:
            row['intercept'], row['slope'] = mae_lin
        else:
            # Fallback: fit from bin means
            stds = df[f'{error_key}_mean'].values
            vel = df['velocity_center'].values
            if len(vel) >= 2:
                lin = np.polyfit(vel, stds, 1)
                row['intercept'], row['slope'] = float(lin[1]), float(lin[0])
            else:
                row['intercept'], row['slope'] = float(np.mean(stds)), 0.0

        # MAE quadratic
        mae_quad = reg.get(f'{error_key}_quadratic')
        if mae_quad:
            row['quad_a'], row['quad_b'], row['quad_c'] = mae_quad
        else:
            row['quad_a'] = row['quad_b'] = row['quad_c'] = ''

        # Std regression
        std_lin = reg.get(f'{error_key}_std_linear')
        if std_lin:
            row['std_intercept'], row['std_slope'] = std_lin
        else:
            row['std_intercept'] = row['std_slope'] = ''

        std_quad = reg.get(f'{error_key}_std_quadratic')
        if std_quad:
            row['std_quad_a'], row['std_quad_b'], row['std_quad_c'] = std_quad
        else:
            row['std_quad_a'] = row['std_quad_b'] = row['std_quad_c'] = ''

        # Variance regression
        var_lin = reg.get(f'{error_key}_var_linear')
        if var_lin:
            row['var_intercept'], row['var_slope'] = var_lin
        else:
            row['var_intercept'] = row['var_slope'] = ''

        var_quad = reg.get(f'{error_key}_var_quadratic')
        if var_quad:
            row['var_quad_a'], row['var_quad_b'], row['var_quad_c'] = var_quad
        else:
            row['var_quad_a'] = row['var_quad_b'] = row['var_quad_c'] = ''

        # Bias regression
        bias_lin = reg.get(f'{error_key}_bias_linear')
        if bias_lin:
            row['bias_intercept'], row['bias_slope'] = bias_lin
        else:
            row['bias_intercept'] = row['bias_slope'] = ''

        bias_quad = reg.get(f'{error_key}_bias_quadratic')
        if bias_quad:
            row['bias_quad_a'], row['bias_quad_b'], row['bias_quad_c'] = bias_quad
        else:
            row['bias_quad_a'] = row['bias_quad_b'] = row['bias_quad_c'] = ''

        regression_rows.append(row)

        # Print summary
        mae_i = row.get('intercept', 0)
        mae_s = row.get('slope', 0)
        std_i = row.get('std_intercept', '')
        var_i = row.get('var_intercept', '')
        bias_i = row.get('bias_intercept', '')
        print(f"  {out_name}: MAE={mae_i:.6f}+{mae_s:.2e}*v"
              f"  std={std_i}"
              f"  var={var_i}"
              f"  bias={bias_i}")

    fieldnames = ["error_type", "intercept", "slope", "quad_a", "quad_b", "quad_c",
                  "bias_intercept", "bias_slope", "bias_quad_a", "bias_quad_b", "bias_quad_c",
                  "std_intercept", "std_slope", "std_quad_a", "std_quad_b", "std_quad_c",
                  "var_intercept", "var_slope", "var_quad_a", "var_quad_b", "var_quad_c",
                  "unit", "notes"]

    regression_path = OUTPUT_DIR / f"{output_name}.csv"
    with open(regression_path, 'w', newline='') as f:
        f.write(f"# {output_name} Localization - Imported from binned evaluation results\n")
        f.write("# MAE linear: mae = intercept + slope * velocity\n")
        f.write("# Std linear: std = std_intercept + std_slope * velocity\n")
        f.write("# Var linear: var = var_intercept + var_slope * velocity\n")
        f.write("# Bias linear: bias = bias_intercept + bias_slope * velocity\n")
        f.write("# MSE (for R matrix) = bias^2 + variance\n")
        f.write("# velocity in m/s, position errors in meters, orientation errors in radians\n\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(regression_rows)
    print(f"\nWrote regression CSV: {regression_path}")

    # --- Distributions CSV (velocity-binned, for error sampling) ---
    all_bins = []
    for error_key, (out_name, unit, notes) in ERROR_TYPES.items():
        if has_bias:
            mu_col = f'{error_key}_bias'
            sig_col = f'{error_key}_bias_std'
        else:
            mu_col = f'{error_key}_mean'
            sig_col = f'{error_key}_std'

        for row in df:
            try:
                mu = float(row[mu_col])
                sig = float(row[sig_col])
            except (ValueError, KeyError):
                continue
            if math.isnan(mu) or math.isnan(sig):
                continue
            sig = max(sig, 1e-6)
            all_bins.append({
                'error_type': out_name,
                'dist_min': float(row['velocity_min']),
                'dist_max': float(row['velocity_max']),
                'distribution': 'normal',
                'param1': round(mu, 6),
                'param2': round(sig, 6),
                'param3': '',
            })

        # Overall distribution
        counts = np.array([float(r['count']) for r in df])
        mus = np.array([float(r.get(mu_col, 'nan')) for r in df])
        sigs = np.array([float(r.get(sig_col, 'nan')) for r in df])
        valid = ~(np.isnan(mus) | np.isnan(sigs))
        if valid.sum() > 0:
            overall_mu = np.average(mus[valid], weights=counts[valid])
            overall_var = np.average(sigs[valid]**2 + mus[valid]**2, weights=counts[valid]) - overall_mu**2
            overall_sig = np.sqrt(max(0, overall_var))
        else:
            overall_mu, overall_sig = 0.0, 0.1

        max_vel = max(float(r['velocity_max']) for r in df)
        all_bins.append({
            'error_type': out_name,
            'dist_min': 0,
            'dist_max': max_vel,
            'distribution': 'normal',
            'param1': round(overall_mu, 6),
            'param2': round(max(overall_sig, 1e-6), 6),
            'param3': '',
        })

    bins_path = OUTPUT_DIR / f"{output_name}_distributions.csv"
    with open(bins_path, 'w', newline='') as f:
        f.write(f"# {output_name} Localization - Velocity-binned error distributions\n")
        if has_bias:
            f.write("# param1=bias (signed mean), param2=bias_std (spread around bias)\n")
        else:
            f.write("# param1=mean (absolute error), param2=std\n")
        f.write("# velocity bins in m/s\n\n")
        w = csv.DictWriter(f, fieldnames=['error_type', 'dist_min', 'dist_max',
                                           'distribution', 'param1', 'param2', 'param3'])
        w.writeheader()
        w.writerows(all_bins)
    print(f"Wrote distributions CSV: {bins_path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python import_localizer_binned.py <binned_csv> <regression_txt> <output_name>")
        print("  or:  python import_localizer_binned.py <binned_csv> <output_name>  (no regression file)")
        sys.exit(1)

    if len(sys.argv) == 3:
        # Old format: just binned CSV + output name
        binned_csv = sys.argv[1]
        regression_txt = None
        output_name = sys.argv[2]
    else:
        binned_csv = sys.argv[1]
        regression_txt = sys.argv[2]
        output_name = sys.argv[3]

    if not Path(binned_csv).exists():
        print(f"Error: Input file not found: {binned_csv}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if regression_txt and Path(regression_txt).exists():
        process_binned_data(binned_csv, regression_txt, output_name)
    else:
        # Fallback: no regression file, fit from bins only
        if regression_txt:
            print(f"Warning: Regression file not found: {regression_txt}, fitting from bins")
        process_binned_data(binned_csv, "/dev/null", output_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
