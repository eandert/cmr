#!/usr/bin/env python3
"""
Import localizer evaluation data from pre-computed binned CSVs.

Reads velocity-binned error statistics (mean, std per 2 m/s bin) and produces:
1. Regression coefficients CSV — regressions fit on bin stds directly (not MAE)
2. Velocity-binned distributions CSV — for error sampling and std-from-bins

Usage:
    python import_localizer_binned.py <binned_csv> <output_name>

Example:
    python import_localizer_binned.py \
        path/to/hd_map_kiss_binned_errors.csv \
        kiss_icp
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "src" / "data" / "sensor_models"

# Error types to process (column prefix in binned CSV -> output name)
ERROR_TYPES = {
    'longitudinal': 'longitudinal',
    'lateral': 'lateral',
    'vertical': 'vertical',
    'roll': 'roll',
    'pitch': 'pitch',
    'yaw': 'yaw',
}

UNITS = {
    'longitudinal': 'meters',
    'lateral': 'meters',
    'vertical': 'meters',
    'roll': 'radians',
    'pitch': 'radians',
    'yaw': 'radians',
}

NOTES = {
    'longitudinal': 'Longitudinal (forward) position error',
    'lateral': 'Lateral (side) position error',
    'vertical': 'Vertical (up/down) position error',
    'roll': 'Roll angle error',
    'pitch': 'Pitch angle error',
    'yaw': 'Yaw angle error',
}


def process_binned_data(input_csv: str, output_name: str):
    """Process pre-computed binned localizer data and generate output CSVs."""
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Filter out rows with count=0 or NaN
    df = df[df['count'] > 0].copy()
    df = df.dropna(subset=['longitudinal_std', 'lateral_std'])
    print(f"  Valid bins: {len(df)}")
    print(f"  Velocity range: {df['velocity_min'].min():.0f} - {df['velocity_max'].max():.0f} m/s")

    vel_centers = df['velocity_center'].values

    # --- Regression CSV (fit on bin stds directly) ---
    regression_rows = []
    for error_type, out_name in ERROR_TYPES.items():
        std_col = f'{error_type}_std'
        stds = df[std_col].values

        # Linear regression: std = intercept + slope * velocity
        if len(vel_centers) >= 2:
            lin = np.polyfit(vel_centers, stds, 1)  # [slope, intercept]
            intercept, slope = float(lin[1]), float(lin[0])
        else:
            intercept, slope = float(np.mean(stds)), 0.0

        # Quadratic regression: std = quad_a*v^2 + quad_b*v + quad_c
        if len(vel_centers) >= 3:
            quad = np.polyfit(vel_centers, stds, 2)  # [a, b, c]
            quad_a, quad_b, quad_c = float(quad[0]), float(quad[1]), float(quad[2])
        else:
            quad_a, quad_b, quad_c = 0.0, slope, intercept

        # Compute R² for reporting
        pred_lin = intercept + slope * vel_centers
        ss_res = np.sum((stds - pred_lin) ** 2)
        ss_tot = np.sum((stds - np.mean(stds)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  {out_name}: intercept={intercept:.6f}, slope={slope:.6e}, "
              f"quad=[{quad_a:.6e}, {quad_b:.6e}, {quad_c:.6f}], R²={r2:.4f}")

        regression_rows.append({
            'error_type': out_name,
            'intercept': intercept,
            'slope': slope,
            'quad_a': quad_a,
            'quad_b': quad_b,
            'quad_c': quad_c,
            'unit': UNITS[error_type],
            'notes': NOTES[error_type],
        })

    regression_df = pd.DataFrame(regression_rows)
    regression_path = OUTPUT_DIR / f"{output_name}.csv"

    with open(regression_path, 'w') as f:
        f.write(f"# {output_name} Localization - Imported from binned evaluation results\n")
        f.write("# Regressions fit on bin stds directly (std-from-bins method)\n")
        f.write("# Linear model: std = intercept + slope * velocity\n")
        f.write("# Quadratic model: std = quad_c + quad_b*v + quad_a*v^2\n")
        f.write("# velocity in m/s, position errors in meters, orientation errors in radians\n")
        f.write("\n")
        regression_df.to_csv(f, index=False)
    print(f"\nWrote regression coefficients to: {regression_path}")

    # --- Distributions CSV (velocity-binned) ---
    all_bins = []
    for error_type, out_name in ERROR_TYPES.items():
        mean_col = f'{error_type}_mean'
        std_col = f'{error_type}_std'

        for _, row in df.iterrows():
            all_bins.append({
                'error_type': out_name,
                'dist_min': row['velocity_min'],
                'dist_max': row['velocity_max'],
                'distribution': 'normal',
                'param1': row[mean_col],  # mean (absolute error magnitude)
                'param2': row[std_col],   # std
                'param3': '',
            })

        # Overall distribution (across all velocities, weighted by count)
        counts = df['count'].values
        means = df[mean_col].values
        stds_arr = df[std_col].values
        total_count = counts.sum()
        if total_count > 0:
            overall_mean = np.average(means, weights=counts)
            # Pool variances: avg_var = weighted_avg(std^2 + mean^2) - overall_mean^2
            overall_var = np.average(stds_arr**2 + means**2, weights=counts) - overall_mean**2
            overall_std = np.sqrt(max(0, overall_var))
        else:
            overall_mean = 0.0
            overall_std = 0.1

        all_bins.append({
            'error_type': out_name,
            'dist_min': 0,
            'dist_max': df['velocity_max'].max(),
            'distribution': 'normal',
            'param1': overall_mean,
            'param2': overall_std,
            'param3': '',
        })
        print(f"  {out_name}: {len(df)} bins, overall mean={overall_mean:.6f}, std={overall_std:.6f}")

    bins_df = pd.DataFrame(all_bins)
    bins_path = OUTPUT_DIR / f"{output_name}_distributions.csv"

    with open(bins_path, 'w') as f:
        f.write(f"# {output_name} Localization - Binned error distributions\n")
        f.write("# Velocity bins in m/s, position errors in meters, orientation errors in radians\n")
        f.write("# param1=mean (absolute error), param2=std, distribution=normal\n")
        f.write("\n")
        bins_df.to_csv(f, index=False)
    print(f"Wrote velocity-binned distributions to: {bins_path}")

    return regression_path, bins_path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_csv = sys.argv[1]
    output_name = sys.argv[2]

    if not Path(input_csv).exists():
        print(f"Error: Input file not found: {input_csv}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_binned_data(input_csv, output_name)
    print("\nDone!")


if __name__ == "__main__":
    main()
