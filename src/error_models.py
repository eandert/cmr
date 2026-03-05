"""
Error Models from Regression Testing

This module implements error models derived from actual sensor regression testing.
Error distributions are distance-binned and support different vehicle types.

Supports loading regression parameters from CSV files for easy configuration.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
from pathlib import Path

# Try to import the sensor model loader
try:
    from sensor_model_loader import load_sensor_model, SensorModelData, DistributionBinData
    HAS_LOADER = True
except ImportError:
    try:
        # Try relative import for when running as module
        from .sensor_model_loader import load_sensor_model, SensorModelData, DistributionBinData
        HAS_LOADER = True
    except ImportError:
        HAS_LOADER = False
        SensorModelData = None
        DistributionBinData = None


# Distribution sampling functions
def sample_normal(mu: float, sigma: float) -> float:
    """Sample from Normal distribution."""
    return np.random.normal(mu, sigma)

def sample_laplace(mu: float, b: float) -> float:
    """Sample from Laplace distribution."""
    return np.random.laplace(mu, b)

def sample_logistic(mu: float, s: float) -> float:
    """Sample from Logistic distribution."""
    return stats.logistic.rvs(loc=mu, scale=s)

def sample_student_t(nu: float, mu: float, sigma: float) -> float:
    """Sample from Student-t distribution."""
    # scipy's t distribution is standard; we need to scale and shift
    return stats.t.rvs(nu, loc=mu, scale=sigma)

def sample_cauchy(x0: float, gamma: float) -> float:
    """Sample from Cauchy distribution."""
    return stats.cauchy.rvs(loc=x0, scale=gamma)


class DistributionBin:
    """Represents a single distance bin with its error distribution."""
    
    def __init__(self, dist_type: str, params: Dict[str, float], 
                 min_dist: float, max_dist: float):
        self.dist_type = dist_type
        self.params = params
        self.min_dist = min_dist
        self.max_dist = max_dist
    
    def sample(self) -> float:
        """Sample an error value from this bin's distribution."""
        if self.dist_type == "normal":
            return sample_normal(self.params["mu"], self.params["sigma"])
        elif self.dist_type == "laplace":
            return sample_laplace(self.params["mu"], self.params["b"])
        elif self.dist_type == "logistic":
            return sample_logistic(self.params["mu"], self.params["s"])
        elif self.dist_type == "student_t":
            return sample_student_t(self.params["nu"], self.params["mu"], self.params["sigma"])
        elif self.dist_type == "cauchy":
            return sample_cauchy(self.params["x0"], self.params["gamma"])
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")
    
    def get_std(self) -> float:
        """Get the standard deviation (or equivalent) for this distribution."""
        if self.dist_type == "normal":
            return self.params["sigma"]
        elif self.dist_type == "laplace":
            # Laplace std = sqrt(2) * b
            return np.sqrt(2) * self.params["b"]
        elif self.dist_type == "logistic":
            # Logistic std = s * pi / sqrt(3)
            return self.params["s"] * np.pi / np.sqrt(3)
        elif self.dist_type == "student_t":
            # For nu > 2, std = sigma * sqrt(nu / (nu - 2))
            nu = self.params["nu"]
            if nu > 2:
                return self.params["sigma"] * np.sqrt(nu / (nu - 2))
            else:
                return self.params["sigma"] * 2  # Approximate for low nu
        elif self.dist_type == "cauchy":
            # Cauchy has undefined std; use gamma as a scale proxy
            # For practical purposes, use ~2*gamma as rough "spread"
            return self.params["gamma"] * 2
        return 0.1  # fallback


class PointPillarsOS1_128ErrorModel:
    """
    Error model for PointPillars detector with OS1_128 LiDAR.
    
    Based on regression testing results with distance-binned error distributions.
    Errors are in distal (radial/depth), perpendicular (lateral), and height dimensions.
    
    The model uses the "All" category which aggregates across Car, Pedestrian, Cyclist.
    
    Two modes:
    - SAMPLING: Uses best-fit distributions (Normal, Laplace, Logistic, Student-t) per bin
    - PREDICTION: Uses linear regression of std vs distance for covariance estimation
    
    Covariance estimation modes (controlled by use_gpem_model flag):
    - Static Covariance (use_gpem_model=False): Uses linear regression of absolute error
    - GPEM Parameterized (use_gpem_model=True): Uses distance-binned distribution standard deviations
    
    Regression parameters can be loaded from CSV file for easy updates.
    """
    
    def __init__(self, model_name: str = "pointpillars_os1_128", load_from_csv: bool = True, 
                 use_gpem_model: bool = False, use_quadratic: bool = False, max_range: float = None):
        self.model_name = model_name
        self.csv_loaded = False
        self.distributions_from_csv = False
        self.use_gpem_model = use_gpem_model  # Flag to use GPEM vs static covariance
        self.use_quadratic = use_quadratic  # Flag to use quadratic vs linear regression
        
        # Load regression parameters from CSV (REQUIRED - no fallback)
        if load_from_csv:
            if not HAS_LOADER:
                raise RuntimeError(f"Cannot load error model '{model_name}': sensor_model_loader not available")
            try:
                self._load_from_csv(model_name)
                self.csv_loaded = True
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Error model '{model_name}' not found in src/data/sensor_models/\n"
                    f"Required files: {model_name}.csv and {model_name}_distributions.csv\n"
                    f"Error: {e}"
                )
        else:
            raise ValueError("load_from_csv=False not supported - models must be loaded from CSV files")
        
        # Verify both regression params and distributions were loaded
        if not self.csv_loaded or not self.distributions_from_csv:
            raise ValueError(
                f"Error model '{model_name}' failed to load completely\n"
                f"csv_loaded={self.csv_loaded}, distributions_from_csv={self.distributions_from_csv}"
            )
        
        # Max detection range - configurable to limit noisy far-range detections
        # Default 70m based on regression training data range (reliable data 0-70m)
        # Beyond this range, regression models extrapolate and become unreliable
        # This also affects ground truth filtering to ensure fair comparison
        self.max_range = max_range if max_range is not None else 70.0  # meters
    
    def _init_hardcoded_distributions(self):
        """Initialize with hardcoded distribution bins (fallback)."""
        # Build distance bins for each error type
        # Using the "All" category from regression testing
        # These use the BEST FIT distribution for actual error sampling
        
        # Distal Error bins (radial/depth direction) - BEST FIT distributions
        self.distal_bins = [
            DistributionBin("logistic", {"mu": 0.0237, "s": 0.0824}, 1, 6),
            DistributionBin("student_t", {"nu": 3.28, "mu": -0.0044, "sigma": 0.0658}, 6, 11),
            DistributionBin("student_t", {"nu": 3.14, "mu": 0.0070, "sigma": 0.0675}, 11, 16),
            DistributionBin("student_t", {"nu": 3.70, "mu": 0.0064, "sigma": 0.0868}, 16, 21),
            DistributionBin("student_t", {"nu": 3.80, "mu": 0.0031, "sigma": 0.1044}, 21, 26),
            DistributionBin("student_t", {"nu": 4.41, "mu": 0.0052, "sigma": 0.1193}, 26, 31),
            DistributionBin("logistic", {"mu": 0.0064, "s": 0.0962}, 31, 36),
            DistributionBin("logistic", {"mu": -0.0120, "s": 0.1064}, 36, 41),
            DistributionBin("normal", {"mu": 0.0074, "sigma": 0.2149}, 41, 46),
            DistributionBin("normal", {"mu": 0.0493, "sigma": 0.2108}, 46, 51),
            DistributionBin("normal", {"mu": 0.0225, "sigma": 0.2517}, 51, 56),
            DistributionBin("normal", {"mu": 0.0134, "sigma": 0.2697}, 56, 61),
            DistributionBin("normal", {"mu": -0.0112, "sigma": 0.2835}, 61, 66),
            DistributionBin("normal", {"mu": 0.0098, "sigma": 0.3110}, 66, 71),
        ]
        
        # Perpendicular Error bins (lateral direction) - BEST FIT distributions
        self.perp_bins = [
            DistributionBin("logistic", {"mu": 0.0394, "s": 0.0668}, 1, 6),
            DistributionBin("student_t", {"nu": 9.14, "mu": 0.0353, "sigma": 0.0780}, 6, 11),
            DistributionBin("student_t", {"nu": 7.34, "mu": 0.0055, "sigma": 0.0720}, 11, 16),
            DistributionBin("student_t", {"nu": 5.10, "mu": 0.0039, "sigma": 0.0713}, 16, 21),
            DistributionBin("student_t", {"nu": 5.35, "mu": -0.0056, "sigma": 0.0694}, 21, 26),
            DistributionBin("student_t", {"nu": 5.26, "mu": -0.0052, "sigma": 0.0774}, 26, 31),
            DistributionBin("student_t", {"nu": 4.85, "mu": 0.0118, "sigma": 0.0832}, 31, 36),
            DistributionBin("logistic", {"mu": 0.0041, "s": 0.0586}, 36, 41),
            DistributionBin("logistic", {"mu": 0.0116, "s": 0.0657}, 41, 46),
            DistributionBin("normal", {"mu": 0.0185, "sigma": 0.1289}, 46, 51),
            DistributionBin("normal", {"mu": 0.0012, "sigma": 0.1359}, 51, 56),
            DistributionBin("normal", {"mu": 0.0133, "sigma": 0.1493}, 56, 61),
            DistributionBin("normal", {"mu": 0.0004, "sigma": 0.1776}, 61, 66),
            DistributionBin("normal", {"mu": -0.0139, "sigma": 0.1567}, 66, 71),
        ]
        
        # Height Error bins - BEST FIT distributions
        self.height_bins = [
            DistributionBin("student_t", {"nu": 2.17, "mu": 0.0073, "sigma": 0.0510}, 1, 6),
            DistributionBin("student_t", {"nu": 2.19, "mu": 0.0007, "sigma": 0.0462}, 6, 11),
            DistributionBin("student_t", {"nu": 2.10, "mu": 0.0007, "sigma": 0.0468}, 11, 16),
            DistributionBin("student_t", {"nu": 2.22, "mu": -0.0018, "sigma": 0.0492}, 16, 21),
            DistributionBin("student_t", {"nu": 2.28, "mu": 0.0014, "sigma": 0.0529}, 21, 26),
            DistributionBin("student_t", {"nu": 2.62, "mu": 0.0045, "sigma": 0.0606}, 26, 31),
            DistributionBin("student_t", {"nu": 3.27, "mu": 0.0110, "sigma": 0.0734}, 31, 36),
            DistributionBin("student_t", {"nu": 3.44, "mu": 0.0123, "sigma": 0.0844}, 36, 41),
            DistributionBin("student_t", {"nu": 3.70, "mu": 0.0245, "sigma": 0.0967}, 41, 46),
            DistributionBin("student_t", {"nu": 4.09, "mu": 0.0382, "sigma": 0.1093}, 46, 51),
            DistributionBin("student_t", {"nu": 3.40, "mu": 0.0295, "sigma": 0.1147}, 51, 56),
            DistributionBin("logistic", {"mu": 0.0513, "s": 0.0986}, 56, 61),
            DistributionBin("logistic", {"mu": 0.0494, "s": 0.0822}, 61, 66),
            DistributionBin("normal", {"mu": 0.0272, "sigma": 0.1874}, 66, 71),
        ]
    
    def _load_from_csv(self, model_name: str):
        """Load regression parameters and distribution bins from CSV files."""
        model_data = load_sensor_model(model_name)
        
        self.MAE_TO_STD = 1.2533141373  # √(π/2) for half-normal to std conversion
        
        # Store the full RegressionParams objects for access to quadratic
        self._model_data = model_data
        
        # Map CSV fields to regression parameters (linear: [intercept, slope])
        self.distal_abs_regression = [model_data.radial.intercept, model_data.radial.slope]
        self.perp_abs_regression = [model_data.lateral.intercept, model_data.lateral.slope]
        self.height_abs_regression = [model_data.vertical.intercept, model_data.vertical.slope]
        self.yaw_abs_regression = [model_data.yaw.intercept, model_data.yaw.slope]
        self.width_abs_regression = [model_data.width.intercept, model_data.width.slope]
        self.length_abs_regression = [model_data.length.intercept, model_data.length.slope]
        self.box_height_abs_regression = [model_data.box_height.intercept, model_data.box_height.slope]
        self.miss_rate_regression = [model_data.miss_rate.intercept, model_data.miss_rate.slope]
        
        # Store quadratic coefficients if available (quad: [a, b, c] for a*d^2 + b*d + c)
        self.distal_quad_regression = self._get_quad_coeffs(model_data.radial)
        self.perp_quad_regression = self._get_quad_coeffs(model_data.lateral)
        self.height_quad_regression = self._get_quad_coeffs(model_data.vertical)
        self.yaw_quad_regression = self._get_quad_coeffs(model_data.yaw)
        self.width_quad_regression = self._get_quad_coeffs(model_data.width)
        self.length_quad_regression = self._get_quad_coeffs(model_data.length)
        self.box_height_quad_regression = self._get_quad_coeffs(model_data.box_height)
        self.miss_rate_quad_regression = self._get_quad_coeffs(model_data.miss_rate)
        
        # Load distribution bins if available
        if model_data.distribution_bins:
            self._load_distribution_bins_from_csv(model_data.distribution_bins)
            self.distributions_from_csv = True
        else:
            self.distributions_from_csv = False
        
        has_quad = model_data.radial.has_quadratic if model_data.radial else False
        print(f"Loaded sensor model from CSV: {model_name} (distributions: {self.distributions_from_csv}, quadratic: {has_quad})")
    
    def _get_quad_coeffs(self, params):
        """Extract quadratic coefficients from RegressionParams, or None if not available."""
        if params and params.has_quadratic:
            return [params.quad_a, params.quad_b, params.quad_c]
        return None
    
    def _load_distribution_bins_from_csv(self, dist_bins: dict):
        """Convert loaded distribution bin data to DistributionBin objects."""
        # Initialize all bin lists
        self.distal_bins = []
        self.perp_bins = []
        self.height_bins = []
        self.width_bins = []
        self.length_bins = []
        self.yaw_bins = []
        self.box_height_bins = []
        self.miss_detection_bins = []
        
        # Map error types from CSV to bin lists
        bin_mapping = {
            'distal': self.distal_bins,
            'perpendicular': self.perp_bins,
            'height': self.height_bins,
            'width': self.width_bins,
            'length': self.length_bins,
            'yaw': self.yaw_bins,
            'box_height': self.box_height_bins,
            'miss_detection': self.miss_detection_bins,
        }
        
        for error_type, bins_list in dist_bins.items():
            if error_type in bin_mapping:
                target_list = bin_mapping[error_type]
                for bin_data in bins_list:
                    target_list.append(DistributionBin(
                        bin_data.distribution,
                        bin_data.params,
                        bin_data.dist_min,
                        bin_data.dist_max
                    ))
    
    def _init_hardcoded_regression(self):
        """Initialize with hardcoded regression parameters (fallback)."""
        # =====================================================================
        # LINEAR REGRESSION for PREDICTED errors (covariance estimation)
        # =====================================================================
        # From absolute error regression models.
        # Format: [intercept, slope] -> abs_error = intercept + slope * distance
        # 
        # To convert absolute error to std for Normal distribution:
        #   MAE = σ × √(2/π), so σ = MAE × √(π/2) ≈ 1.253 × MAE
        
        self.MAE_TO_STD = 1.2533141373  # √(π/2) for half-normal to std conversion
        
        # Radial/Distal Error (range, towards/away from sensor)
        self.distal_abs_regression = [0.014547, 0.000788]  # [intercept, slope]
        
        # Lateral/Perpendicular Error (cross-range, left/right in ground plane)  
        self.perp_abs_regression = [0.023388, 0.000279]
        
        # Vertical/Height Error (elevation, up/down)
        self.height_abs_regression = [0.013141, 0.000563]
        
        # Yaw/Angle Error (heading angle, radians)
        self.yaw_abs_regression = [0.045834, -0.000485]
        
        # Width Error (box dimension)
        self.width_abs_regression = [0.016275, 0.000228]
        
        # Length Error (box dimension)
        self.length_abs_regression = [0.034834, 0.000969]
        
        # Box Height Error (box dimension)
        self.box_height_abs_regression = [0.022837, 0.000211]
        
        # Missed Detection Rate
        self.miss_rate_regression = [0.487947, 0.002430]  # [intercept, slope]
        
    def _find_bin(self, bins: list, distance: float) -> Optional[DistributionBin]:
        """Find the appropriate bin for a given distance.
        
        Prefers more specific bins (smaller range) over wide catchall bins.
        """
        matching_bins = []
        for bin_obj in bins:
            if bin_obj.min_dist <= distance < bin_obj.max_dist:
                matching_bins.append(bin_obj)
        
        if matching_bins:
            # Prefer the bin with the smallest range (most specific)
            return min(matching_bins, key=lambda b: b.max_dist - b.min_dist)
        
        # If distance is beyond our bins, use the last one (by max_dist)
        if bins and distance >= max(b.max_dist for b in bins):
            return max(bins, key=lambda b: b.max_dist)
        # If distance is below our bins, use the first one (by min_dist)
        if bins and distance < min(b.min_dist for b in bins):
            return min(bins, key=lambda b: b.min_dist)
        return None
    
    def sample_distal_error(self, distance: float) -> float:
        """Sample a distal (radial/depth) error for the given distance."""
        bin_obj = self._find_bin(self.distal_bins, distance)
        if bin_obj:
            return bin_obj.sample()
        return 0.0
    
    def sample_perpendicular_error(self, distance: float) -> float:
        """Sample a perpendicular (lateral) error for the given distance."""
        bin_obj = self._find_bin(self.perp_bins, distance)
        if bin_obj:
            return bin_obj.sample()
        return 0.0
    
    def sample_height_error(self, distance: float) -> float:
        """Sample a height error for the given distance. Currently unused in 2D sim."""
        bin_obj = self._find_bin(self.height_bins, distance)
        if bin_obj:
            return bin_obj.sample()
        return 0.0
    
    def _abs_to_std(self, abs_error: float) -> float:
        """Convert absolute error (MAE) to standard deviation assuming half-normal."""
        return abs_error * self.MAE_TO_STD
    
    def _predict_with_regression(self, linear_coeffs, quad_coeffs, distance: float) -> float:
        """Predict absolute error using linear or quadratic regression based on settings."""
        if self.use_quadratic and quad_coeffs is not None:
            a, b, c = quad_coeffs
            return max(0.001, a * distance**2 + b * distance + c)
        else:
            intercept, slope = linear_coeffs
            return max(0.001, intercept + slope * distance)
    
    def get_distal_std(self, distance: float) -> float:
        """
        Get the standard deviation for distal/radial error at this distance.
        
        Modes based on use_gpem_model and use_quadratic flags:
        - Static (use_gpem_model=False): Uses a constant average std (no distance prediction)
        - GPEM Linear (use_gpem_model=True, use_quadratic=False): Uses linear regression
        - GPEM Quadratic (use_gpem_model=True, use_quadratic=True): Uses quadratic regression
        
        Args:
            distance: Detection distance in meters
        """
        if self.use_gpem_model:
            abs_error = self._predict_with_regression(
                self.distal_abs_regression, self.distal_quad_regression, distance
            )
            return self._abs_to_std(abs_error)
        else:
            return self.get_distal_std_average()
    
    def get_perpendicular_std(self, distance: float) -> float:
        """
        Get the standard deviation for perpendicular/lateral error at this distance.
        
        Modes based on use_gpem_model and use_quadratic flags:
        - Static (use_gpem_model=False): Uses a constant average std (no distance prediction)
        - GPEM Linear (use_gpem_model=True, use_quadratic=False): Uses linear regression
        - GPEM Quadratic (use_gpem_model=True, use_quadratic=True): Uses quadratic regression
        
        Args:
            distance: Detection distance in meters
        """
        if self.use_gpem_model:
            abs_error = self._predict_with_regression(
                self.perp_abs_regression, self.perp_quad_regression, distance
            )
            return self._abs_to_std(abs_error)
        else:
            return self.get_perpendicular_std_average()
    
    def get_distal_std_average(self) -> float:
        """
        Get the average standard deviation for distal error across all distance bins.
        Used by static covariance model (no distance prediction).
        """
        if not self.distal_bins:
            return 0.1
        stds = [bin_obj.get_std() for bin_obj in self.distal_bins]
        return np.mean(stds)
    
    def get_perpendicular_std_average(self) -> float:
        """
        Get the average standard deviation for perpendicular error across all distance bins.
        Used by static covariance model (no distance prediction).
        """
        if not self.perp_bins:
            return 0.1
        stds = [bin_obj.get_std() for bin_obj in self.perp_bins]
        return np.mean(stds)
    
    def get_height_std(self, distance: float) -> float:
        """
        Get the PREDICTED standard deviation for height/vertical error at this distance.
        Uses linear regression of absolute error, converted to std.
        """
        intercept, slope = self.height_abs_regression
        abs_error = max(0.001, intercept + slope * distance)
        return self._abs_to_std(abs_error)
    
    def get_yaw_std(self, distance: float) -> float:
        """
        Get the PREDICTED standard deviation for yaw/heading error at this distance (radians).
        Uses linear regression of absolute error, converted to std.
        """
        intercept, slope = self.yaw_abs_regression
        abs_error = max(0.001, intercept + slope * distance)
        return self._abs_to_std(abs_error)
    
    def get_width_std(self, distance: float) -> float:
        """
        Get the PREDICTED standard deviation for width error at this distance.
        Uses linear regression of absolute error, converted to std.
        """
        intercept, slope = self.width_abs_regression
        abs_error = max(0.001, intercept + slope * distance)
        return self._abs_to_std(abs_error)
    
    def get_length_std(self, distance: float) -> float:
        """
        Get the PREDICTED standard deviation for length error at this distance.
        Uses linear regression of absolute error, converted to std.
        """
        intercept, slope = self.length_abs_regression
        abs_error = max(0.001, intercept + slope * distance)
        return self._abs_to_std(abs_error)
    
    def get_actual_bin_std(self, error_type: str, distance: float) -> float:
        """
        Get the ACTUAL standard deviation from the best-fit distribution bin.
        This is for debugging/analysis, not for covariance prediction.
        
        Args:
            error_type: One of "distal", "perpendicular", "height"
            distance: Distance in meters
        
        Returns:
            The std from the best-fit distribution for that bin
        """
        if error_type == "distal":
            bins = self.distal_bins
        elif error_type == "perpendicular":
            bins = self.perp_bins
        elif error_type == "height":
            bins = self.height_bins
        else:
            return 0.1
        
        bin_obj = self._find_bin(bins, distance)
        if bin_obj:
            return bin_obj.get_std()
        return 0.1
    
    def detection_probability(self, distance: float) -> float:
        """
        Calculate detection probability at the given distance.
        
        Uses miss rate regression: miss_rate = intercept + slope * d
        Detection probability = 1 - miss_rate
        """
        if distance > self.max_range:
            return 0.0
        intercept, slope = self.miss_rate_regression
        miss_rate = intercept + slope * distance
        # Clamp miss_rate to [0, 1] then convert to detection probability
        miss_rate = max(0.0, min(1.0, miss_rate))
        return 1.0 - miss_rate
    
    def sample_width_error(self, distance: float) -> float:
        """
        Sample a width error for the given distance.
        Uses distribution bins from CSV if available, else Normal with regression std.
        """
        if hasattr(self, 'width_bins') and self.width_bins:
            bin_obj = self._find_bin(self.width_bins, distance)
            if bin_obj:
                return bin_obj.sample()
        # Fallback to regression-derived std
        std = self.get_width_std(distance)
        return np.random.normal(0, std)
    
    def sample_length_error(self, distance: float) -> float:
        """
        Sample a length error for the given distance.
        Uses distribution bins from CSV if available, else Normal with regression std.
        """
        if hasattr(self, 'length_bins') and self.length_bins:
            bin_obj = self._find_bin(self.length_bins, distance)
            if bin_obj:
                return bin_obj.sample()
        # Fallback to regression-derived std
        std = self.get_length_std(distance)
        return np.random.normal(0, std)
    
    def sample_yaw_error(self, distance: float) -> float:
        """
        Sample a yaw/heading error for the given distance (radians).
        Uses distribution bins from CSV if available, else Normal with regression std.
        """
        if hasattr(self, 'yaw_bins') and self.yaw_bins:
            bin_obj = self._find_bin(self.yaw_bins, distance)
            if bin_obj:
                return bin_obj.sample()
        # Fallback to regression-derived std
        std = self.get_yaw_std(distance)
        return np.random.normal(0, std)
    
    def sample_box_height_error(self, distance: float) -> float:
        """
        Sample a box height error for the given distance (for future 3D).
        Uses distribution bins from CSV if available.
        """
        if hasattr(self, 'box_height_bins') and self.box_height_bins:
            bin_obj = self._find_bin(self.box_height_bins, distance)
            if bin_obj:
                return bin_obj.sample()
        # Fallback to regression-derived std
        intercept, slope = self.box_height_abs_regression
        abs_error = max(0.001, intercept + slope * distance)
        std = abs_error * self.MAE_TO_STD
        return np.random.normal(0, std)
    
    def sample_errors(self, distance: float, target_angle: float) -> Tuple[float, float, float, float]:
        """
        Sample position errors for an object at the given distance.
        
        Args:
            distance: Distance to the object in meters
            target_angle: Angle to the object in radians (from sensor frame)
        
        Returns:
            Tuple of (x_error, y_error, distal_std, perpendicular_std)
            Errors are in the global frame
        """
        # Sample errors in sensor frame (distal = along line to object, perp = orthogonal)
        distal_error = self.sample_distal_error(distance)
        perp_error = self.sample_perpendicular_error(distance)
        
        # Convert to global frame
        # Distal error is along the ray to the object
        # Perpendicular error is orthogonal to the ray
        x_error = distal_error * np.cos(target_angle) - perp_error * np.sin(target_angle)
        y_error = distal_error * np.sin(target_angle) + perp_error * np.cos(target_angle)
        
        # Get expected standard deviations for covariance estimation
        distal_std = self.get_distal_std(distance)
        perp_std = self.get_perpendicular_std(distance)
        
        return x_error, y_error, distal_std, perp_std
    
    def sample_all_errors(self, distance: float, target_angle: float) -> dict:
        """
        Sample all errors (position, dimensions, yaw) for an object at the given distance.
        
        Args:
            distance: Distance to the object in meters
            target_angle: Angle to the object in radians (from sensor frame)
        
        Returns:
            Dictionary with all sampled errors and predicted stds
        """
        # Position errors
        x_error, y_error, distal_std, perp_std = self.sample_errors(distance, target_angle)
        
        # Dimension errors
        width_error = self.sample_width_error(distance)
        length_error = self.sample_length_error(distance)
        
        # Yaw error
        yaw_error = self.sample_yaw_error(distance)
        
        return {
            'x_error': x_error,
            'y_error': y_error,
            'width_error': width_error,
            'length_error': length_error,
            'yaw_error': yaw_error,
            # Predicted stds for covariance
            'distal_std': distal_std,
            'perp_std': perp_std,
            'width_std': self.get_width_std(distance),
            'length_std': self.get_length_std(distance),
            'yaw_std': self.get_yaw_std(distance),
        }


class CarSpecificErrorModel(PointPillarsOS1_128ErrorModel):
    """
    Error model specifically tuned for Car detections.
    Uses Car-specific regression testing results.
    """
    
    def __init__(self):
        super().__init__()
        
        # Override with Car-specific bins
        # Distal Error for Car
        self.distal_bins = [
            DistributionBin("logistic", {"mu": -0.0347, "s": 0.0372}, 1, 6),
            DistributionBin("student_t", {"nu": 4.97, "mu": -0.0043, "sigma": 0.0470}, 6, 11),
            DistributionBin("logistic", {"mu": 0.0135, "s": 0.0321}, 11, 16),
            DistributionBin("logistic", {"mu": 0.0195, "s": 0.0344}, 16, 21),
            DistributionBin("logistic", {"mu": 0.0112, "s": 0.0392}, 21, 26),
            DistributionBin("normal", {"mu": -0.0125, "sigma": 0.0756}, 26, 31),
            DistributionBin("normal", {"mu": 0.0178, "sigma": 0.0851}, 31, 36),
            DistributionBin("normal", {"mu": -0.0071, "sigma": 0.1052}, 36, 41),
            DistributionBin("normal", {"mu": -0.0235, "sigma": 0.0689}, 46, 51),
        ]
        
        # Perpendicular Error for Car
        self.perp_bins = [
            DistributionBin("normal", {"mu": 0.1250, "sigma": 0.0719}, 1, 6),
            DistributionBin("normal", {"mu": 0.0717, "sigma": 0.0881}, 6, 11),
            DistributionBin("normal", {"mu": 0.0363, "sigma": 0.0956}, 11, 16),
            DistributionBin("logistic", {"mu": 0.0093, "s": 0.0605}, 16, 21),
            DistributionBin("normal", {"mu": 0.0002, "sigma": 0.0937}, 21, 26),
            DistributionBin("normal", {"mu": -0.0335, "sigma": 0.0938}, 26, 31),
            DistributionBin("laplace", {"mu": -0.0031, "b": 0.0647}, 31, 36),
            DistributionBin("normal", {"mu": 0.0145, "sigma": 0.0743}, 36, 41),
            DistributionBin("laplace", {"mu": 0.0498, "b": 0.0783}, 46, 51),
        ]


class PedestrianSpecificErrorModel(PointPillarsOS1_128ErrorModel):
    """
    Error model specifically tuned for Pedestrian detections.
    Uses Pedestrian-specific regression testing results.
    """
    
    def __init__(self):
        super().__init__()
        
        # Override with Pedestrian-specific bins
        # Distal Error for Pedestrian  
        self.distal_bins = [
            DistributionBin("logistic", {"mu": 0.0195, "s": 0.0414}, 1, 6),
            DistributionBin("normal", {"mu": -0.0089, "sigma": 0.0671}, 6, 11),
            DistributionBin("normal", {"mu": 0.0092, "sigma": 0.0673}, 11, 16),
            DistributionBin("normal", {"mu": 0.0335, "sigma": 0.0900}, 16, 21),
            DistributionBin("logistic", {"mu": 0.0558, "s": 0.0530}, 21, 26),
            DistributionBin("laplace", {"mu": 0.0398, "b": 0.0828}, 26, 31),
            DistributionBin("logistic", {"mu": 0.0107, "s": 0.0713}, 31, 36),
            DistributionBin("logistic", {"mu": 0.0054, "s": 0.0565}, 36, 41),
            DistributionBin("normal", {"mu": -0.0657, "sigma": 0.1176}, 41, 46),
            DistributionBin("laplace", {"mu": 0.0133, "b": 0.0618}, 46, 51),
        ]
        
        # Perpendicular Error for Pedestrian
        self.perp_bins = [
            DistributionBin("normal", {"mu": 0.0967, "sigma": 0.0884}, 1, 6),
            DistributionBin("normal", {"mu": 0.0713, "sigma": 0.0801}, 6, 11),
            DistributionBin("normal", {"mu": 0.0149, "sigma": 0.0696}, 11, 16),
            DistributionBin("normal", {"mu": 0.0263, "sigma": 0.0698}, 16, 21),
            DistributionBin("normal", {"mu": 0.0116, "sigma": 0.0804}, 21, 26),
            DistributionBin("normal", {"mu": 0.0187, "sigma": 0.0901}, 26, 31),
            DistributionBin("normal", {"mu": 0.0001, "sigma": 0.0943}, 31, 36),
            DistributionBin("laplace", {"mu": 0.0140, "b": 0.0669}, 36, 41),
            DistributionBin("normal", {"mu": 0.0054, "sigma": 0.1141}, 41, 46),
            DistributionBin("normal", {"mu": 0.0147, "sigma": 0.0972}, 46, 51),
        ]


class CyclistSpecificErrorModel(PointPillarsOS1_128ErrorModel):
    """
    Error model specifically tuned for Cyclist detections.
    Uses Cyclist-specific regression testing results.
    """
    
    def __init__(self):
        super().__init__()
        
        # Override with Cyclist-specific bins (extended range up to 71m)
        # Distal Error for Cyclist
        self.distal_bins = [
            DistributionBin("logistic", {"mu": 0.0273, "s": 0.0850}, 1, 6),
            DistributionBin("student_t", {"nu": 3.51, "mu": -0.0038, "sigma": 0.0754}, 6, 11),
            DistributionBin("student_t", {"nu": 3.19, "mu": 0.0050, "sigma": 0.0733}, 11, 16),
            DistributionBin("student_t", {"nu": 3.96, "mu": 0.0012, "sigma": 0.0944}, 16, 21),
            DistributionBin("student_t", {"nu": 4.10, "mu": -0.0016, "sigma": 0.1124}, 21, 26),
            DistributionBin("student_t", {"nu": 4.68, "mu": 0.0043, "sigma": 0.1255}, 26, 31),
            DistributionBin("logistic", {"mu": 0.0057, "s": 0.0983}, 31, 36),
            DistributionBin("logistic", {"mu": -0.0129, "s": 0.1088}, 36, 41),
            DistributionBin("normal", {"mu": 0.0110, "sigma": 0.2194}, 41, 46),
            DistributionBin("normal", {"mu": 0.0522, "sigma": 0.2143}, 46, 51),
            DistributionBin("normal", {"mu": 0.0211, "sigma": 0.2546}, 51, 56),
            DistributionBin("normal", {"mu": 0.0135, "sigma": 0.2703}, 56, 61),
            DistributionBin("normal", {"mu": -0.0108, "sigma": 0.2846}, 61, 66),
            DistributionBin("normal", {"mu": 0.0098, "sigma": 0.3110}, 66, 71),
        ]
        
        # Perpendicular Error for Cyclist
        self.perp_bins = [
            DistributionBin("logistic", {"mu": 0.0330, "s": 0.0668}, 1, 6),
            DistributionBin("student_t", {"nu": 5.80, "mu": 0.0218, "sigma": 0.0692}, 6, 11),
            DistributionBin("student_t", {"nu": 6.42, "mu": 0.0001, "sigma": 0.0680}, 11, 16),
            DistributionBin("student_t", {"nu": 4.79, "mu": 0.0013, "sigma": 0.0687}, 16, 21),
            DistributionBin("student_t", {"nu": 4.67, "mu": -0.0074, "sigma": 0.0664}, 21, 26),
            DistributionBin("student_t", {"nu": 4.86, "mu": -0.0055, "sigma": 0.0758}, 26, 31),
            DistributionBin("student_t", {"nu": 4.79, "mu": 0.0122, "sigma": 0.0835}, 31, 36),
            DistributionBin("logistic", {"mu": 0.0034, "s": 0.0591}, 36, 41),
            DistributionBin("logistic", {"mu": 0.0118, "s": 0.0657}, 41, 46),
            DistributionBin("normal", {"mu": 0.0177, "sigma": 0.1298}, 46, 51),
            DistributionBin("normal", {"mu": 0.0017, "sigma": 0.1372}, 51, 56),
            DistributionBin("normal", {"mu": 0.0131, "sigma": 0.1497}, 56, 61),
            DistributionBin("normal", {"mu": -0.0019, "sigma": 0.1764}, 61, 66),
            DistributionBin("normal", {"mu": -0.0139, "sigma": 0.1567}, 66, 71),
        ]
        
        # Match the OS1_128 LiDAR sensor max range (100m)
        self.max_range = 100.0


# Global error model instances for easy access
_error_models = {}

def get_error_model(model_name: str, force_reload: bool = False, use_gpem_model: bool = False,
                    use_quadratic: bool = False, max_range: float = None) -> PointPillarsOS1_128ErrorModel:
    """
    Get an error model instance by name.
    
    MUST load from CSV files - raises error if model not found.
    
    Args:
        model_name: Model name matching a CSV file in src/data/sensor_models/
                   Examples: "pointpillars_os1_128", "bev_fusion"
        force_reload: If True, reload from CSV even if cached
        use_gpem_model: If True, use GPEM parameterized covariance (distance-dependent).
                       If False, use static averaged covariance.
        use_quadratic: If True and use_gpem_model=True, use quadratic regression instead of linear.
        max_range: Maximum detection range in meters. Default 60m to cut noisy far-range data.
    
    Returns:
        The error model instance loaded from CSV
        
    Raises:
        FileNotFoundError: If model CSV files don't exist
    """
    # Cache key includes all settings so different variants are cached separately
    cache_key = f"{model_name}_gpem_{use_gpem_model}_quad_{use_quadratic}_range_{max_range}"
    global _error_models
    
    if force_reload and cache_key in _error_models:
        del _error_models[cache_key]
    
    if cache_key not in _error_models:
        # Load from CSV, fail if not found
        model = PointPillarsOS1_128ErrorModel(
            model_name=model_name,
            load_from_csv=True,
            use_gpem_model=use_gpem_model,
            use_quadratic=use_quadratic,
            max_range=max_range
        )
        
        # Verify that model was actually loaded from CSV
        if not model.csv_loaded:
            raise FileNotFoundError(
                f"Error model '{model_name}' not found in src/data/sensor_models/\n"
                f"Available files must include: {model_name}.csv and {model_name}_distributions.csv"
            )
        
        _error_models[cache_key] = model
    
    return _error_models[cache_key]


def reload_all_models():
    """Clear the model cache to force reloading from CSV files."""
    global _error_models
    _error_models = {}
    
    # Also clear the sensor model loader cache if available
    if HAS_LOADER:
        try:
            from sensor_model_loader import clear_model_cache
            clear_model_cache()
        except ImportError:
            try:
                from .sensor_model_loader import clear_model_cache
                clear_model_cache()
            except ImportError:
                pass  # Cache clear not available
