#!/usr/bin/env python3
"""
Import localizer evaluation results into CMR sensor model format.

Reads 6DOF raw error data and generates:
1. {localizer}.csv - Regression coefficients for lateral/radial errors
2. {localizer}_distributions.csv - Velocity-binned normal distributions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

def compute_regressions(df, col):
    """Compute linear and quadratic regressions for an error column."""
    velocity = df['velocity'].values.reshape(-1, 1)
    errors = np.abs(df[col].values)
    
    # Linear regression
    lin_model = LinearRegression()
    lin_model.fit(velocity, errors)
    
    # Quadratic regression
    poly = PolynomialFeatures(degree=2, include_bias=False)
    vel_poly = poly.fit_transform(velocity)
    quad_model = LinearRegression()
    quad_model.fit(vel_poly, errors)
    
    return {
        'intercept': lin_model.intercept_,
        'slope': lin_model.coef_[0],
        'quad_a': quad_model.coef_[1],  # v^2 coefficient
        'quad_b': quad_model.coef_[0],  # v coefficient  
        'quad_c': quad_model.intercept_,  # constant
    }

def compute_velocity_bins(df, col, bin_edges):
    """Compute velocity-binned mean/std for an error column."""
    results = []
    for i in range(len(bin_edges) - 1):
        v_min, v_max = bin_edges[i], bin_edges[i+1]
        mask = (df['velocity'] >= v_min) & (df['velocity'] < v_max)
        subset = df[mask]
        
        if len(subset) > 0:
            abs_errors = np.abs(subset[col].values)
            results.append({
                'dist_min': int(v_min),
                'dist_max': int(v_max),
                'mean': np.mean(abs_errors),
                'std': np.std(abs_errors),
            })
    return results

def process_localizer(name, raw_csv_path, output_dir):
    """Process a localizer's raw 6DOF data and generate CSVs."""
    print(f"\nProcessing {name}...")
    df = pd.read_csv(raw_csv_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Velocity range: {df['velocity'].min():.2f} - {df['velocity'].max():.2f} m/s")
    
    # Map columns: x=radial (forward), y=lateral, z=vertical
    # For 2D localization we primarily care about x (radial) and y (lateral)
    col_mapping = {
        'radial': 'x',      # Forward/longitudinal error
        'lateral': 'y',     # Lateral/sideways error
        'vertical': 'z',    # Vertical error (not used in 2D)
        'roll': 'roll',
        'pitch': 'pitch',
        'yaw': 'yaw',
    }
    
    # Compute regressions for each error type
    regressions = {}
    for error_type, col in col_mapping.items():
        if col in df.columns:
            regressions[error_type] = compute_regressions(df, col)
            r = regressions[error_type]
            print(f"  {error_type}: intercept={r['intercept']:.6f}, slope={r['slope']:.6f}")
    
    # Generate main CSV (regression coefficients)
    csv_lines = [
        f"# {name} Localization - Imported from evaluation results",
        "# Linear model: abs_error = intercept + slope * velocity",
        "# Quadratic model: abs_error = quad_c + quad_b*v + quad_a*v^2",
        "# velocity in m/s, position errors in meters, orientation errors in radians",
        "",
        "error_type,intercept,slope,quad_a,quad_b,quad_c,unit,notes",
    ]
    
    units = {
        'radial': 'meters', 'lateral': 'meters', 'vertical': 'meters',
        'roll': 'radians', 'pitch': 'radians', 'yaw': 'radians'
    }
    notes = {
        'radial': 'X (forward) position error',
        'lateral': 'Y (lateral) position error', 
        'vertical': 'Z (vertical) position error',
        'roll': 'Roll angle error',
        'pitch': 'Pitch angle error',
        'yaw': 'Yaw angle error'
    }
    
    for error_type in ['radial', 'lateral', 'vertical', 'roll', 'pitch', 'yaw']:
        if error_type in regressions:
            r = regressions[error_type]
            csv_lines.append(
                f"{error_type},{r['intercept']},{r['slope']},"
                f"{r['quad_a']},{r['quad_b']},{r['quad_c']},"
                f"{units[error_type]},{notes[error_type]}"
            )
    
    main_csv_path = os.path.join(output_dir, f"{name}.csv")
    with open(main_csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"  Wrote {main_csv_path}")
    
    # Generate distributions CSV (velocity-binned)
    # Use 1 m/s bins from 0 to 15 m/s (matching existing format)
    bin_edges = list(range(0, 28))  # 0, 1, 2, ... 27 m/s
    
    dist_lines = [
        f"# {name} Localization - Binned error distributions",
        "# Velocity bins in m/s, position errors in meters, orientation errors in radians",
        "",
        "error_type,dist_min,dist_max,distribution,param1,param2,param3",
    ]
    
    for error_type, col in col_mapping.items():
        if col in df.columns:
            bins = compute_velocity_bins(df, col, bin_edges)
            for b in bins:
                dist_lines.append(
                    f"{error_type},{b['dist_min']},{b['dist_max']},normal,{b['mean']},{b['std']},"
                )
    
    # Add overall distribution (0 to max velocity)
    dist_lines.append("")
    dist_lines.append("# Overall distribution (all velocities)")
    for error_type, col in col_mapping.items():
        if col in df.columns:
            abs_errors = np.abs(df[col].values)
            mean_err = np.mean(abs_errors)
            std_err = np.std(abs_errors)
            max_vel = int(df['velocity'].max()) + 1
            dist_lines.append(
                f"{error_type},0,{max_vel},normal,{mean_err},{std_err},"
            )
    
    dist_csv_path = os.path.join(output_dir, f"{name}_distributions.csv")
    with open(dist_csv_path, 'w') as f:
        f.write('\n'.join(dist_lines) + '\n')
    print(f"  Wrote {dist_csv_path}")


if __name__ == "__main__":
    output_dir = "src/data/sensor_models"
    
    # Process Kiss ICP
    process_localizer(
        "kiss_icp",
        "~/test/localization_project/evaluation_results/kiss_icp_best/kiss_icp_best_all_raw_errors_6dof.csv",
        output_dir
    )
    
    # Process ORB-SLAM3
    process_localizer(
        "orb_slam3", 
        "~/test/localization_project/evaluation_results/orb_slam3_best/orb_slam3_best_all_raw_errors_6dof.csv",
        output_dir
    )
    
    print("\nDone! New localizer models imported.")
