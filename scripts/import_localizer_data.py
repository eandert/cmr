#!/usr/bin/env python3
"""
Import localizer evaluation data and generate CSV files for the CMR localizer module.

Reads raw 6DOF error data from localization evaluation results and produces:
1. Regression coefficients CSV (linear and quadratic models for error vs velocity)
2. Velocity-binned distributions CSV (for error sampling)

Usage:
    python import_localizer_data.py <input_csv> <output_name>
    
Example:
    python import_localizer_data.py \
        ~/test/localization_project/evaluation_results/kiss_icp_best/kiss_icp_best_all_raw_errors_6dof.csv \
        kiss_icp
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "src" / "data" / "sensor_models"

# Velocity bins in m/s (0-27 m/s covers ~0-60 mph)
VELOCITY_BINS = [(i, i+1) for i in range(27)]


def compute_regression(velocities: np.ndarray, errors: np.ndarray):
    """Compute linear and quadratic regression for absolute errors vs velocity."""
    abs_errors = np.abs(errors)
    vel_col = velocities.reshape(-1, 1)
    
    # Linear regression: error = intercept + slope * velocity
    lin_reg = LinearRegression()
    lin_reg.fit(vel_col, abs_errors)
    intercept = lin_reg.intercept_
    slope = lin_reg.coef_[0]
    
    # Quadratic regression: error = a*v^2 + b*v + c
    poly = PolynomialFeatures(degree=2)
    vel_poly = poly.fit_transform(vel_col)
    quad_reg = LinearRegression()
    quad_reg.fit(vel_poly, abs_errors)
    # Coefficients are [1, v, v^2] -> [c, b, a] in our convention
    quad_c = quad_reg.intercept_
    quad_b = quad_reg.coef_[1]
    quad_a = quad_reg.coef_[2]
    
    return {
        'intercept': intercept,
        'slope': slope,
        'quad_a': quad_a,
        'quad_b': quad_b,
        'quad_c': quad_c
    }


def compute_velocity_bins(velocities: np.ndarray, errors: np.ndarray, error_type: str):
    """Compute velocity-binned distribution statistics."""
    bins_data = []
    
    for vel_min, vel_max in VELOCITY_BINS:
        mask = (velocities >= vel_min) & (velocities < vel_max)
        if mask.sum() < 5:  # Skip bins with too few samples
            continue
        
        bin_errors = errors[mask]
        
        # For localization we use signed errors (can be + or -)
        # Distribution is normal with mean and std
        mean = np.mean(bin_errors)
        std = np.std(bin_errors)
        
        bins_data.append({
            'error_type': error_type,
            'dist_min': vel_min,
            'dist_max': vel_max,
            'distribution': 'normal',
            'param1': mean,  # mean
            'param2': std,   # std
            'param3': ''
        })
    
    return bins_data


def process_localizer_data(input_csv: str, output_name: str):
    """Process localizer evaluation data and generate output CSVs."""
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Velocity range: {df['velocity'].min():.2f} - {df['velocity'].max():.2f} m/s")
    
    velocities = df['velocity'].values
    
    # Error components mapping (CSV column -> our naming)
    error_mapping = {
        'x': 'radial',       # Forward/backward = radial/longitudinal
        'y': 'lateral',      # Left/right = lateral
        'z': 'vertical',     # Up/down = vertical
        'roll': 'roll',
        'pitch': 'pitch', 
        'yaw': 'yaw'
    }
    
    # Compute regressions for each component
    regressions = {}
    for col, name in error_mapping.items():
        errors = df[col].values
        reg = compute_regression(velocities, errors)
        regressions[name] = reg
        print(f"  {name}: intercept={reg['intercept']:.4f}, slope={reg['slope']:.6f}, "
              f"quad=[{reg['quad_a']:.6e}, {reg['quad_b']:.6f}, {reg['quad_c']:.4f}]")
    
    # Generate regression CSV
    regression_rows = []
    for name in ['radial', 'lateral', 'vertical', 'roll', 'pitch', 'yaw']:
        reg = regressions[name]
        unit = 'meters' if name in ['radial', 'lateral', 'vertical'] else 'radians'
        notes_map = {
            'radial': 'X (forward) position error',
            'lateral': 'Y (lateral) position error',
            'vertical': 'Z (vertical) position error',
            'roll': 'Roll angle error',
            'pitch': 'Pitch angle error',
            'yaw': 'Yaw angle error'
        }
        regression_rows.append({
            'error_type': name,
            'intercept': reg['intercept'],
            'slope': reg['slope'],
            'quad_a': reg['quad_a'],
            'quad_b': reg['quad_b'],
            'quad_c': reg['quad_c'],
            'unit': unit,
            'notes': notes_map[name]
        })
    
    regression_df = pd.DataFrame(regression_rows)
    regression_path = OUTPUT_DIR / f"{output_name}.csv"
    
    # Write with header comment
    with open(regression_path, 'w') as f:
        f.write(f"# {output_name} Localization - Imported from evaluation results\n")
        f.write("# Linear model: abs_error = intercept + slope * velocity\n")
        f.write("# Quadratic model: abs_error = quad_c + quad_b*v + quad_a*v^2\n")
        f.write("# velocity in m/s, position errors in meters, orientation errors in radians\n")
        f.write("\n")
        regression_df.to_csv(f, index=False)
    print(f"\nWrote regression coefficients to: {regression_path}")
    
    # Generate velocity-binned distributions CSV
    all_bins = []
    for col, name in error_mapping.items():
        errors = df[col].values
        bins = compute_velocity_bins(velocities, errors, name)
        all_bins.extend(bins)
        print(f"  {name}: {len(bins)} velocity bins")
    
    # Add overall distribution (all velocities)
    print("\n  Overall distributions:")
    for col, name in error_mapping.items():
        errors = df[col].values
        mean = np.mean(errors)
        std = np.std(errors)
        all_bins.append({
            'error_type': name,
            'dist_min': 0,
            'dist_max': 27,
            'distribution': 'normal',
            'param1': mean,
            'param2': std,
            'param3': ''
        })
        print(f"    {name}: mean={mean:.4f}, std={std:.4f}")
    
    bins_df = pd.DataFrame(all_bins)
    bins_path = OUTPUT_DIR / f"{output_name}_distributions.csv"
    
    with open(bins_path, 'w') as f:
        f.write(f"# {output_name} Localization - Binned error distributions\n")
        f.write("# Velocity bins in m/s, position errors in meters, orientation errors in radians\n")
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
    
    process_localizer_data(input_csv, output_name)
    print("\nDone!")


if __name__ == "__main__":
    main()
