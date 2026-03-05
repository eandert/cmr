"""
Sensor Model Loader

Loads sensor error models from CSV files for easy configuration and updates.

Usage:
    from sensor_model_loader import load_sensor_model, list_available_models
    
    # List available sensor models
    models = list_available_models()
    
    # Load a specific model
    model = load_sensor_model("pointpillars_os1_128")
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Default path for sensor model data files
DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "sensor_models"


@dataclass
class RegressionParams:
    """Regression parameters for an error type (linear and optional quadratic)."""
    error_type: str
    intercept: float
    slope: float
    unit: str
    notes: str = ""
    # Quadratic coefficients: error = quad_a * d^2 + quad_b * d + quad_c
    quad_a: Optional[float] = None
    quad_b: Optional[float] = None
    quad_c: Optional[float] = None
    
    @property
    def has_quadratic(self) -> bool:
        """Check if quadratic coefficients are available."""
        return self.quad_a is not None and self.quad_b is not None and self.quad_c is not None
    
    def predict_linear(self, distance: float) -> float:
        """Predict absolute error using linear model."""
        return max(0.001, self.intercept + self.slope * distance)
    
    def predict_quadratic(self, distance: float) -> float:
        """Predict absolute error using quadratic model (if available)."""
        if not self.has_quadratic:
            return self.predict_linear(distance)
        return max(0.001, self.quad_a * distance**2 + self.quad_b * distance + self.quad_c)
    
    def predict(self, distance: float, use_quadratic: bool = False) -> float:
        """Predict absolute error at the given distance.
        
        Args:
            distance: Distance in meters
            use_quadratic: If True and quadratic coeffs available, use quadratic model
        """
        if use_quadratic and self.has_quadratic:
            return self.predict_quadratic(distance)
        return self.predict_linear(distance)
    
    def predict_std(self, distance: float, mae_to_std: float = 1.2533141373, use_quadratic: bool = False) -> float:
        """Predict standard deviation at the given distance."""
        return self.predict(distance, use_quadratic=use_quadratic) * mae_to_std


@dataclass
class DistributionBinData:
    """A single distribution bin for error sampling."""
    error_type: str
    dist_min: float
    dist_max: float
    distribution: str  # normal, laplace, logistic, student_t, cauchy
    params: Dict[str, float]  # distribution-specific parameters
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type,
            'dist_min': self.dist_min,
            'dist_max': self.dist_max,
            'distribution': self.distribution,
            'params': self.params
        }


@dataclass 
class SensorModelData:
    """Complete sensor model data loaded from CSV."""
    name: str
    file_path: str
    params: Dict[str, RegressionParams]
    # Distribution bins for sampling (loaded from *_distributions.csv)
    distribution_bins: Dict[str, List[DistributionBinData]] = None
    
    def __post_init__(self):
        if self.distribution_bins is None:
            self.distribution_bins = {}
    
    # Convenience properties
    @property
    def radial(self) -> Optional[RegressionParams]:
        return self.params.get("radial")
    
    @property
    def lateral(self) -> Optional[RegressionParams]:
        return self.params.get("lateral")
    
    @property
    def vertical(self) -> Optional[RegressionParams]:
        return self.params.get("vertical")
    
    @property
    def yaw(self) -> Optional[RegressionParams]:
        return self.params.get("yaw")
    
    @property
    def width(self) -> Optional[RegressionParams]:
        return self.params.get("width")
    
    @property
    def length(self) -> Optional[RegressionParams]:
        return self.params.get("length")
    
    @property
    def box_height(self) -> Optional[RegressionParams]:
        return self.params.get("box_height")
    
    @property
    def miss_rate(self) -> Optional[RegressionParams]:
        return self.params.get("miss_rate")


def load_sensor_model_csv(file_path: str) -> SensorModelData:
    """
    Load a sensor model from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        SensorModelData with all regression parameters
    """
    params = {}
    
    with open(file_path, 'r') as f:
        # Filter out comment lines (starting with #) and empty lines
        lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            error_type = row['error_type'].strip()
            
            # Parse quadratic coefficients (optional)
            quad_a = None
            quad_b = None
            quad_c = None
            if 'quad_a' in row and row['quad_a'].strip():
                try:
                    quad_a = float(row['quad_a'])
                    quad_b = float(row['quad_b'])
                    quad_c = float(row['quad_c'])
                except (ValueError, KeyError):
                    pass
            
            params[error_type] = RegressionParams(
                error_type=error_type,
                intercept=float(row['intercept']),
                slope=float(row['slope']),
                unit=row['unit'].strip(),
                notes=row.get('notes', '').strip(),
                quad_a=quad_a,
                quad_b=quad_b,
                quad_c=quad_c,
            )
    
    name = Path(file_path).stem
    model_data = SensorModelData(name=name, file_path=file_path, params=params)
    
    # Try to load distribution bins from companion file
    dist_file = Path(file_path).parent / f"{name}_distributions.csv"
    if dist_file.exists():
        model_data.distribution_bins = load_distribution_bins_csv(str(dist_file))
    
    return model_data


def load_distribution_bins_csv(file_path: str) -> Dict[str, List[DistributionBinData]]:
    """
    Load distribution bins from a CSV file.
    
    Args:
        file_path: Path to the distributions CSV file
        
    Returns:
        Dictionary mapping error_type to list of DistributionBinData
    """
    bins = {}
    
    with open(file_path, 'r') as f:
        # Filter out comment lines (starting with #) and empty lines
        lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        
        for row in reader:
            error_type = row['error_type'].strip()
            distribution = row['distribution'].strip()
            
            # Parse parameters based on distribution type
            param1 = float(row['param1']) if row['param1'].strip() else None
            param2 = float(row['param2']) if row['param2'].strip() else None
            param3 = float(row['param3']) if row.get('param3', '').strip() else None
            
            # Build params dict based on distribution type
            if distribution == 'normal':
                params = {'mu': param1, 'sigma': param2}
            elif distribution == 'laplace':
                params = {'mu': param1, 'b': param2}
            elif distribution == 'logistic':
                params = {'mu': param1, 's': param2}
            elif distribution == 'student_t':
                params = {'nu': param1, 'mu': param2, 'sigma': param3}
            elif distribution == 'cauchy':
                params = {'x0': param1, 'gamma': param2}
            else:
                raise ValueError(f"Unknown distribution type: {distribution}")
            
            bin_data = DistributionBinData(
                error_type=error_type,
                dist_min=float(row['dist_min']),
                dist_max=float(row['dist_max']),
                distribution=distribution,
                params=params
            )
            
            if error_type not in bins:
                bins[error_type] = []
            bins[error_type].append(bin_data)
    
    # Sort bins by distance
    for error_type in bins:
        bins[error_type].sort(key=lambda b: b.dist_min)
    
    return bins


def list_available_models(data_dir: str = None) -> List[str]:
    """
    List all available sensor models in the data directory.
    
    Args:
        data_dir: Directory to search (default: src/data/sensor_models)
        
    Returns:
        List of model names (without .csv extension)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    models = []
    for f in data_path.glob("*.csv"):
        models.append(f.stem)
    
    return sorted(models)


def load_sensor_model(name: str, data_dir: str = None) -> SensorModelData:
    """
    Load a sensor model by name.
    
    Args:
        name: Name of the sensor model (e.g., "pointpillars_os1_128")
        data_dir: Directory to search (default: src/data/sensor_models)
        
    Returns:
        SensorModelData with all regression parameters
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    file_path = Path(data_dir) / f"{name}.csv"
    
    if not file_path.exists():
        available = list_available_models(data_dir)
        raise FileNotFoundError(
            f"Sensor model '{name}' not found at {file_path}. "
            f"Available models: {available}"
        )
    
    return load_sensor_model_csv(str(file_path))


def get_model_summary(model: SensorModelData) -> str:
    """Get a formatted summary of a sensor model."""
    lines = [
        f"=== Sensor Model: {model.name} ===",
        f"File: {model.file_path}",
        "",
        "Regression Parameters (abs_error = intercept + slope * distance):",
        "-" * 60,
    ]
    
    for error_type, params in sorted(model.params.items()):
        lines.append(
            f"  {error_type:12s}: {params.intercept:+.6f} + {params.slope:+.6f} * d  [{params.unit}]"
        )
    
    return "\n".join(lines)


# Cache for loaded models
_model_cache: Dict[str, SensorModelData] = {}


def get_cached_model(name: str, data_dir: str = None) -> SensorModelData:
    """Get a sensor model, using cache if available."""
    cache_key = f"{data_dir or 'default'}:{name}"
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = load_sensor_model(name, data_dir)
    
    return _model_cache[cache_key]


def clear_model_cache():
    """Clear the model cache (useful for reloading after edits)."""
    global _model_cache
    _model_cache = {}


if __name__ == "__main__":
    # Demo: list and load available models
    print("Available sensor models:")
    for name in list_available_models():
        print(f"  - {name}")
    
    print()
    
    # Load and display the pointpillars model
    try:
        model = load_sensor_model("pointpillars_os1_128")
        print(get_model_summary(model))
        
        print()
        print("Example predictions at 30m:")
        print(f"  Radial std:  {model.radial.predict_std(30):.4f} m")
        print(f"  Lateral std: {model.lateral.predict_std(30):.4f} m")
        print(f"  Yaw std:     {model.yaw.predict_std(30):.4f} rad ({model.yaw.predict_std(30)*57.3:.2f} deg)")
        print(f"  Width std:   {model.width.predict_std(30):.4f} m")
        print(f"  Length std:  {model.length.predict_std(30):.4f} m")
        print(f"  Miss rate:   {model.miss_rate.predict(30):.1%}")
        print(f"  Detect prob: {1 - model.miss_rate.predict(30):.1%}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
