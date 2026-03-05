#!/usr/bin/env python3
"""
Import localizer error models from localization evaluation results into CMR sensor_models format.

Usage:
  python scripts/import_localizer_error_models.py

Reads from:
  ../localization_project/evaluation_results/kiss_icp_*/
  ../localization_project/evaluation_results/orb_slam3_*/
  ../localization_project/evaluation_results/stereo_vo_*/

Writes to:
  src/data/sensor_models/kiss_icp.csv, kiss_icp_distributions.csv
  src/data/sensor_models/orb_slam3.csv, orb_slam3_distributions.csv
  src/data/sensor_models/stereo_vo.csv, stereo_vo_distributions.csv

Note: Localizers use VELOCITY (m/s) as the independent variable, not distance.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCALIZATION_BASE = REPO_ROOT.parent / "localization_project" / "evaluation_results"
CMR_MODELS = REPO_ROOT / "src" / "data" / "sensor_models"

# Map regression names to CMR CSV columns
# (linear_name, quadratic_name, cmr_type, unit, notes)
REGRESSION_MAP = [
    ("x_linear", "x_quadratic", "radial", "meters", "X (forward) position error"),
    ("y_linear", "y_quadratic", "lateral", "meters", "Y (lateral) position error"),
    ("z_linear", "z_quadratic", "vertical", "meters", "Z (vertical) position error"),
    ("roll_linear", "roll_quadratic", "roll", "radians", "Roll angle error"),
    ("pitch_linear", "pitch_quadratic", "pitch", "radians", "Pitch angle error"),
    ("yaw_linear", "yaw_quadratic", "yaw", "radians", "Yaw angle error"),
]


def parse_regression_file(filepath: Path) -> dict:
    """
    Parse regression_models.txt to extract linear and quadratic coefficients.
    
    Linear format:  x_linear:  slope * v + intercept  (R²=...)
    Quadratic format: x_quadratic:  a*v^2 + b*v + c  (R²=...)
    
    Returns dict: {
        'x_linear': {'slope': ..., 'intercept': ..., 'r2': ...},
        'x_quadratic': {'a': ..., 'b': ..., 'c': ..., 'r2': ...},
        ...
    }
    """
    results = {}
    
    if not filepath.exists():
        return results
    
    content = filepath.read_text()
    
    # Linear pattern: name:  slope * v + intercept  (R²=value)
    linear_pattern = re.compile(
        r'^(\w+_linear):\s+([+-]?\d+\.?\d*e?[+-]?\d*)\s*\*\s*v\s*\+\s*([+-]?\d+\.?\d*e?[+-]?\d*)\s*\(R²=([0-9.]+)\)',
        re.MULTILINE
    )
    
    # Quadratic pattern: name:  a*v^2 + b*v + c  (R²=value)
    quad_pattern = re.compile(
        r'^(\w+_quadratic):\s+([+-]?\d+\.?\d*e?[+-]?\d*)\*v\^2\s*\+\s*([+-]?\d+\.?\d*e?[+-]?\d*)\*v\s*\+\s*([+-]?\d+\.?\d*e?[+-]?\d*)\s*\(R²=([0-9.]+)\)',
        re.MULTILINE
    )
    
    for match in linear_pattern.finditer(content):
        name, slope, intercept, r2 = match.groups()
        results[name] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r2)
        }
    
    for match in quad_pattern.finditer(content):
        name, a, b, c, r2 = match.groups()
        results[name] = {
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'r2': float(r2)
        }
    
    return results


def aggregate_regression_coefficients(localizer_name: str) -> dict:
    """
    Aggregate regression coefficients from all sequences for a localizer.
    Returns averaged coefficients.
    """
    # Find all sequence directories
    pattern = f"{localizer_name}_*"
    seq_dirs = sorted(LOCALIZATION_BASE.glob(pattern))
    seq_dirs = [d for d in seq_dirs if d.is_dir()]
    
    if not seq_dirs:
        print(f"Warning: No sequence directories found for {localizer_name}")
        return {}
    
    # Collect coefficients from all sequences
    all_coeffs = defaultdict(lambda: defaultdict(list))
    
    for seq_dir in seq_dirs:
        reg_file = seq_dir / f"{seq_dir.name}_regression_models.txt"
        if not reg_file.exists():
            continue
        
        coeffs = parse_regression_file(reg_file)
        for name, values in coeffs.items():
            for key, val in values.items():
                all_coeffs[name][key].append(val)
    
    # Average across sequences
    averaged = {}
    for name, values in all_coeffs.items():
        averaged[name] = {key: sum(vals) / len(vals) for key, vals in values.items()}
    
    return averaged


def write_localizer_csv(localizer_name: str, display_name: str, coefficients: dict):
    """Write the main regression CSV for a localizer."""
    output_path = CMR_MODELS / f"{localizer_name}.csv"
    
    with open(output_path, 'w', newline='') as f:
        f.write(f"# {display_name} Localization - KITTI Odometry\n")
        f.write("# Linear model: abs_error = intercept + slope * velocity\n")
        f.write("# Quadratic model: abs_error = quad_a * v^2 + quad_b * v + quad_c\n")
        f.write("# velocity in m/s, position errors in meters, orientation errors in radians\n")
        f.write("# For covariance: std = 1.253 * abs_error (half-normal to std conversion)\n")
        f.write("\n")
        
        writer = csv.writer(f)
        writer.writerow(['error_type', 'intercept', 'slope', 'quad_a', 'quad_b', 'quad_c', 'unit', 'notes'])
        
        for linear_name, quad_name, cmr_type, unit, notes in REGRESSION_MAP:
            linear = coefficients.get(linear_name, {})
            quad = coefficients.get(quad_name, {})
            
            intercept = linear.get('intercept', 0.0)
            slope = linear.get('slope', 0.0)
            r2_linear = linear.get('r2', 0.0)
            
            quad_a = quad.get('a', 0.0)
            quad_b = quad.get('b', 0.0)
            quad_c = quad.get('c', 0.0)
            
            writer.writerow([
                cmr_type,
                f"{intercept:.6g}",
                f"{slope:.6g}",
                f"{quad_a:.6e}",
                f"{quad_b:.6g}",
                f"{quad_c:.6g}",
                unit,
                f"{notes} (R²={r2_linear:.4f})"
            ])
    
    print(f"Wrote {output_path}")


def copy_distributions_csv(localizer_name: str):
    """Copy the existing distributions CSV (binned errors)."""
    src = LOCALIZATION_BASE / f"{localizer_name}_distributions.csv"
    dst = CMR_MODELS / f"{localizer_name}_distributions.csv"
    
    if src.exists():
        import shutil
        shutil.copy(src, dst)
        print(f"Copied {dst}")
    else:
        print(f"Warning: {src} not found")


LOCALIZERS = [
    ("kiss_icp", "Kiss ICP"),
    ("orb_slam3", "ORB-SLAM3"),
    ("stereo_vo", "Stereo VO"),
]


def main():
    CMR_MODELS.mkdir(parents=True, exist_ok=True)
    
    for localizer_name, display_name in LOCALIZERS:
        print(f"\nProcessing {display_name}...")
        
        # Aggregate coefficients from all sequences
        coefficients = aggregate_regression_coefficients(localizer_name)
        
        if not coefficients:
            print(f"  No coefficients found, checking for pre-aggregated file...")
            # Try to read from existing combined file and add quadratic placeholders
            existing = LOCALIZATION_BASE / f"{localizer_name}.csv"
            if existing.exists():
                print(f"  Using existing {existing}")
                # Parse existing and add quadratic columns
                coefficients = parse_existing_csv(existing)
        
        if coefficients:
            write_localizer_csv(localizer_name, display_name, coefficients)
        
        # Copy distributions
        copy_distributions_csv(localizer_name)


def parse_existing_csv(filepath: Path) -> dict:
    """Parse existing CSV without quadratic columns and estimate quadratic coefficients."""
    coefficients = {}
    
    with open(filepath) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    
    if not lines:
        return {}
    
    reader = csv.DictReader(lines)
    for row in reader:
        error_type = row.get('error_type', '')
        intercept = float(row.get('intercept', 0))
        slope = float(row.get('slope', 0))
        
        # Create linear entry
        linear_name = f"{error_type_to_var(error_type)}_linear"
        coefficients[linear_name] = {
            'intercept': intercept,
            'slope': slope,
            'r2': 0.0
        }
        
        # Estimate quadratic coefficients
        # For static-like behavior: small quadratic term
        # Quadratic = a*v^2 + b*v + c where we want it to approximate linear at typical velocities
        # Use a small quadratic term that adds ~10% at v=20 m/s
        quad_name = f"{error_type_to_var(error_type)}_quadratic"
        avg_error_at_15 = intercept + slope * 15
        # Small quadratic term: a = 0.001 * avg_error / 225 (so a*225 ≈ 0.001 * avg_error)
        quad_a = 0.0005 * abs(avg_error_at_15) / 225 if avg_error_at_15 != 0 else 1e-6
        # Adjust b and c to match linear at v=15
        # linear(15) = intercept + slope*15
        # quad(15) = a*225 + b*15 + c
        # We want quad(15) ≈ linear(15), so set c = intercept, b = slope, small a
        coefficients[quad_name] = {
            'a': quad_a,
            'b': slope,
            'c': intercept,
            'r2': 0.0
        }
    
    return coefficients


def error_type_to_var(error_type: str) -> str:
    """Convert CMR error_type to variable name prefix."""
    mapping = {
        'radial': 'x',
        'lateral': 'y', 
        'vertical': 'z',
        'roll': 'roll',
        'pitch': 'pitch',
        'yaw': 'yaw',
    }
    return mapping.get(error_type, error_type)


if __name__ == "__main__":
    main()
