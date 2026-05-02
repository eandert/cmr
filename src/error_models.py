"""
Error Models from Regression Testing

This module implements error models derived from actual sensor regression testing.
Error distributions are distance-binned and support different vehicle types.

Supports loading regression parameters from CSV files for easy configuration.
"""

import math
import random
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional
from pathlib import Path

# Try to import the sensor model loader
try:
    from sensor_model_loader import load_sensor_model, load_distribution_bins_csv, SensorModelData, DistributionBinData
    HAS_LOADER = True
except ImportError:
    try:
        from .sensor_model_loader import load_sensor_model, load_distribution_bins_csv, SensorModelData, DistributionBinData
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
    """Represents a single distance (or polar) bin with its error distribution."""

    def __init__(self, dist_type: str, params: Dict[str, float],
                 min_dist: float, max_dist: float,
                 min_angle: float = None, max_angle: float = None):
        self.dist_type = dist_type
        self.params = params
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_angle = min_angle  # degrees, None for distance-only bins
        self.max_angle = max_angle

    @property
    def is_polar(self) -> bool:
        return self.min_angle is not None

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
        elif self.dist_type == "constant":
            return self.params.get("value", 0.0)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")

    def get_std(self) -> float:
        """Get the standard deviation (or equivalent) for this distribution."""
        if self.dist_type == "normal":
            return self.params["sigma"]
        elif self.dist_type == "laplace":
            return np.sqrt(2) * self.params["b"]
        elif self.dist_type == "logistic":
            return self.params["s"] * np.pi / np.sqrt(3)
        elif self.dist_type == "student_t":
            nu = self.params["nu"]
            if nu > 2:
                return self.params["sigma"] * np.sqrt(nu / (nu - 2))
            else:
                return self.params["sigma"] * 2
        elif self.dist_type == "cauchy":
            return self.params["gamma"] * 2
        elif self.dist_type == "constant":
            return 0.0
        return 0.1  # fallback

    def get_mae(self) -> float:
        """Get the MAE stored in this bin (polar bins store MAE in params)."""
        return self.params.get("mae", self.get_std() / 1.2533)


class DetectorErrorModel:
    """
    Generic detector error model supporting distance-only and polar (angle+distance) bins.

    Supports DETR3D, BEV Fusion, CenterPoint, and other detectors.
    Errors are in distal (radial/depth), perpendicular (lateral), and height dimensions.

    Modes:
    - SAMPLING: Uses best-fit distributions per bin for error generation
    - PREDICTION: Uses regression or bin lookup for covariance estimation

    Covariance estimation modes:
    - Baseline (use_gpem_model=False): Fixed covariance
    - Static (use_gpem_model=False): Averaged error across all bins
    - GPEM Linear (use_gpem_model=True, use_quadratic=False): Linear regression of std vs distance
    - GPEM Quadratic (use_gpem_model=True, use_quadratic=True): Quadratic regression
    - GPEM Polar (use_polar=True): Direct lookup from polar (angle+distance) bins
    """

    def __init__(self, model_name: str = "pointpillars_os1_128", load_from_csv: bool = True,
                 use_gpem_model: bool = False, use_quadratic: bool = False,
                 use_polar: bool = False, max_range: float = None):
        self.model_name = model_name
        self.csv_loaded = False
        self.distributions_from_csv = False
        self.use_gpem_model = use_gpem_model  # Flag to use GPEM vs static covariance
        self.use_quadratic = use_quadratic  # Flag to use quadratic vs linear regression
        self.use_polar = use_polar  # Flag to use polar (angle+distance) bin lookup for covariance
        
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
        
        # Max detection range — default 100m to match 100m evaluation data
        self.max_range = max_range if max_range is not None else 100.0  # meters
    
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
        
        # Store direct std regression if available (bypasses MAE * 1.2533 conversion)
        self._has_std_regression = model_data.radial.has_std_regression if model_data.radial else False
        if self._has_std_regression:
            self._std_regressions = {
                'distal': model_data.radial,
                'perp': model_data.lateral,
                'height': model_data.vertical,
                'yaw': model_data.yaw,
                'width': model_data.width,
                'length': model_data.length,
                'box_height': model_data.box_height,
            }

        # Load distribution bins if available
        if model_data.distribution_bins:
            self._load_distribution_bins_from_csv(model_data.distribution_bins)
            self.distributions_from_csv = True
            # Only refit from bins if we DON'T have direct std regression
            if self.use_gpem_model and not self._has_std_regression:
                self._refit_regressions_from_bins()
            elif self.use_gpem_model and self._has_std_regression:
                # Use direct std regression — set MAE_TO_STD=1.0 and overwrite regressions
                self.MAE_TO_STD = 1.0
                self.distal_abs_regression = [model_data.radial.std_intercept, model_data.radial.std_slope]
                self.perp_abs_regression = [model_data.lateral.std_intercept, model_data.lateral.std_slope]
                # Store std quadratic if available
                if model_data.radial.has_std_quadratic:
                    self.distal_quad_regression = [model_data.radial.std_quad_a, model_data.radial.std_quad_b, model_data.radial.std_quad_c]
                if model_data.lateral.has_std_quadratic:
                    self.perp_quad_regression = [model_data.lateral.std_quad_a, model_data.lateral.std_quad_b, model_data.lateral.std_quad_c]
        else:
            self.distributions_from_csv = False

        # Load polar (angle+distance) distribution bins if available
        self.has_polar = False
        try:
            polar_path = Path(model_data.file_path).parent / f"{model_name}_polar_distributions.csv"
            if polar_path.exists():
                polar_bins = load_distribution_bins_csv(str(polar_path))
                self._load_polar_bins(polar_bins)
                self.has_polar = True
        except Exception as e:
            print(f"  Warning: failed to load polar bins for {model_name}: {e}")

        has_quad = model_data.radial.has_quadratic if model_data.radial else False
        polar_str = f", polar: {self.has_polar}" if self.has_polar else ""
        print(f"Loaded sensor model from CSV: {model_name} (distributions: {self.distributions_from_csv}, quadratic: {has_quad}{polar_str})")
    
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
                        bin_data.dist_max,
                        min_angle=bin_data.angle_min,
                        max_angle=bin_data.angle_max,
                    ))

    def _load_polar_bins(self, polar_bins: dict):
        """Load polar (angle+distance) distribution bins into separate lists."""
        self.polar_distal_bins = []
        self.polar_perp_bins = []
        self.polar_height_bins = []
        self.polar_yaw_bins = []
        self.polar_width_bins = []
        self.polar_length_bins = []
        self.polar_box_height_bins = []
        self.polar_miss_bins = []

        bin_mapping = {
            'distal': self.polar_distal_bins,
            'perpendicular': self.polar_perp_bins,
            'height': self.polar_height_bins,
            'yaw': self.polar_yaw_bins,
            'width': self.polar_width_bins,
            'length': self.polar_length_bins,
            'box_height': self.polar_box_height_bins,
            'miss_detection': self.polar_miss_bins,
        }

        for error_type, bins_list in polar_bins.items():
            if error_type in bin_mapping:
                target_list = bin_mapping[error_type]
                for bin_data in bins_list:
                    target_list.append(DistributionBin(
                        bin_data.distribution,
                        bin_data.params,
                        bin_data.dist_min,
                        bin_data.dist_max,
                        min_angle=bin_data.angle_min,
                        max_angle=bin_data.angle_max,
                    ))

    def _refit_regressions_from_bins(self):
        """
        Refit GPEM regression coefficients to predict std directly from distribution bins.

        Instead of: MAE = intercept + slope * d  →  std = MAE * 1.2533 (Normal assumption)
        We now do:  std = intercept + slope * d  (fitted from per-bin get_std() values)

        This correctly handles non-Normal distributions (Laplace, Logistic, Student-t)
        because get_std() applies the correct conversion for each distribution type.
        The regression smooths across bin boundaries naturally.
        """
        # Set MAE_TO_STD = 1.0 since regression now predicts std directly
        self.MAE_TO_STD = 1.0

        # Refit distal (radial) regressions
        if self.distal_bins:
            d, s = self._bins_to_distance_std(self.distal_bins)
            if len(d) >= 2:
                lin = np.polyfit(d, s, 1)  # [slope, intercept]
                self.distal_abs_regression = [float(lin[1]), float(lin[0])]
                if len(d) >= 3:
                    quad = np.polyfit(d, s, 2)  # [a, b, c]
                    self.distal_quad_regression = [float(quad[0]), float(quad[1]), float(quad[2])]

        # Refit perpendicular (lateral) regressions
        if self.perp_bins:
            d, s = self._bins_to_distance_std(self.perp_bins)
            if len(d) >= 2:
                lin = np.polyfit(d, s, 1)
                self.perp_abs_regression = [float(lin[1]), float(lin[0])]
                if len(d) >= 3:
                    quad = np.polyfit(d, s, 2)
                    self.perp_quad_regression = [float(quad[0]), float(quad[1]), float(quad[2])]

    @staticmethod
    def _bins_to_distance_std(bins):
        """Extract (distances, stds) arrays from distribution bins, filtering zeros."""
        distances = []
        stds = []
        for b in bins:
            center = (b.min_dist + b.max_dist) / 2.0
            s = b.get_std()
            if s > 0:
                distances.append(center)
                stds.append(s)
        return np.array(distances), np.array(stds)

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
        
    def _find_bin(self, bins: list, distance: float, angle_deg: float = None) -> Optional[DistributionBin]:
        """Find the appropriate bin for a given distance (and optionally angle).

        For polar bins (with angle data), matches both distance and angle.
        For distance-only bins, ignores angle. Prefers more specific bins.
        """
        if not bins:
            return None

        # Check if bins have angle data
        has_angle = bins[0].is_polar and angle_deg is not None

        if has_angle:
            return self._find_polar_bin(bins, distance, angle_deg)

        # Distance-only lookup
        matching_bins = []
        for bin_obj in bins:
            if bin_obj.min_dist <= distance < bin_obj.max_dist:
                matching_bins.append(bin_obj)

        if matching_bins:
            return min(matching_bins, key=lambda b: b.max_dist - b.min_dist)

        # Fallback: clamp to nearest bin
        if distance >= max(b.max_dist for b in bins):
            return max(bins, key=lambda b: b.max_dist)
        if distance < min(b.min_dist for b in bins):
            return min(bins, key=lambda b: b.min_dist)
        return None

    def _find_polar_bin(self, bins: list, distance: float, angle_deg: float) -> Optional[DistributionBin]:
        """Find bin matching both distance and angle (degrees, -180 to 180)."""
        # Normalize angle to [-180, 180)
        angle_deg = ((angle_deg + 180) % 360) - 180

        for bin_obj in bins:
            if (bin_obj.min_dist <= distance < bin_obj.max_dist and
                    bin_obj.min_angle <= angle_deg < bin_obj.max_angle):
                return bin_obj

        # Fallback: find closest by distance, then angle
        best = None
        best_dist = float('inf')
        for bin_obj in bins:
            d_center = (bin_obj.min_dist + bin_obj.max_dist) / 2
            a_center = (bin_obj.min_angle + bin_obj.max_angle) / 2
            dist_err = abs(distance - d_center)
            angle_err = abs(angle_deg - a_center)
            score = dist_err + angle_err * 0.1  # weight distance more
            if score < best_dist:
                best_dist = score
                best = bin_obj
        return best
    
    def _get_bins_for_type(self, polar_bins, fallback_bins, angle_deg):
        """Return polar bins if available and angle given, else fallback distance-only bins."""
        if self.has_polar and polar_bins and angle_deg is not None:
            return polar_bins, angle_deg
        return fallback_bins, None

    def sample_distal_error(self, distance: float, angle_deg: float = None) -> float:
        """Sample a distal (radial/depth) error for the given distance (and angle)."""
        bins, a = self._get_bins_for_type(
            getattr(self, 'polar_distal_bins', None), self.distal_bins, angle_deg)
        bin_obj = self._find_bin(bins, distance, a)
        if bin_obj:
            return bin_obj.sample()
        return 0.0

    def sample_perpendicular_error(self, distance: float, angle_deg: float = None) -> float:
        """Sample a perpendicular (lateral) error for the given distance (and angle)."""
        bins, a = self._get_bins_for_type(
            getattr(self, 'polar_perp_bins', None), self.perp_bins, angle_deg)
        bin_obj = self._find_bin(bins, distance, a)
        if bin_obj:
            return bin_obj.sample()
        return 0.0

    def sample_height_error(self, distance: float, angle_deg: float = None) -> float:
        """Sample a height error for the given distance. Currently unused in 2D sim."""
        bins, a = self._get_bins_for_type(
            getattr(self, 'polar_height_bins', None), self.height_bins, angle_deg)
        bin_obj = self._find_bin(bins, distance, a)
        if bin_obj:
            return bin_obj.sample()
        return 0.0
    
    def _abs_to_std(self, abs_error: float) -> float:
        """Convert absolute error (MAE) to standard deviation assuming half-normal."""
        return abs_error * self.MAE_TO_STD
    
    def _get_max_bin_distance(self) -> float:
        """Get the maximum distance covered by evaluation bins.

        Prevents regression extrapolation beyond measured data.
        """
        if not hasattr(self, '_max_bin_dist'):
            max_d = self.max_range
            for bins in [self.distal_bins, self.perp_bins]:
                specific = [b for b in bins if (b.max_dist - b.min_dist) <= 10]
                if specific:
                    max_d = min(max_d, max(b.max_dist for b in specific))
            self._max_bin_dist = max_d
        return self._max_bin_dist

    def _clamp_distance(self, distance: float) -> float:
        """Clamp distance to range covered by evaluation bins."""
        return max(0.0, min(distance, self._get_max_bin_distance()))

    def _predict_with_regression(self, linear_coeffs, quad_coeffs, distance: float) -> float:
        """Predict absolute error using linear or quadratic regression based on settings."""
        distance = self._clamp_distance(distance)
        if self.use_quadratic and quad_coeffs is not None:
            a, b, c = quad_coeffs
            return max(0.001, a * distance**2 + b * distance + c)
        else:
            intercept, slope = linear_coeffs
            return max(0.001, intercept + slope * distance)
    
    def _get_polar_std(self, polar_bins, distance, angle_deg):
        """Polar bin lookup for std. Returns std or None if not available."""
        if polar_bins is None or angle_deg is None:
            return None
        bin_obj = self._find_bin(polar_bins, distance, angle_deg)
        if bin_obj:
            return bin_obj.get_std()
        return None

    def get_distal_std(self, distance: float, angle_deg: float = None) -> float:
        """Get the standard deviation for distal/radial error at this distance (and angle).

        Mode selection:
        - use_polar=True: Direct lookup from polar (angle+distance) bins
        - use_gpem_model=True: Linear or quadratic regression
        - Otherwise: Static average
        """
        if self.use_polar and self.has_polar:
            polar_std = self._get_polar_std(getattr(self, 'polar_distal_bins', None), distance, angle_deg)
            if polar_std is not None:
                return polar_std

        if self.use_gpem_model:
            abs_error = self._predict_with_regression(
                self.distal_abs_regression, self.distal_quad_regression, distance
            )
            return self._abs_to_std(abs_error)
        else:
            return self.get_distal_std_average()

    def get_perpendicular_std(self, distance: float, angle_deg: float = None) -> float:
        """Get the standard deviation for perpendicular/lateral error at this distance (and angle)."""
        if self.use_polar and self.has_polar:
            polar_std = self._get_polar_std(getattr(self, 'polar_perp_bins', None), distance, angle_deg)
            if polar_std is not None:
                return polar_std

        if self.use_gpem_model:
            abs_error = self._predict_with_regression(
                self.perp_abs_regression, self.perp_quad_regression, distance
            )
            return self._abs_to_std(abs_error)
        else:
            return self.get_perpendicular_std_average()
    
    def get_distal_mse(self, distance: float, angle_deg: float = None) -> float:
        """Get MSE (bias² + variance) for distal error — used for R matrix diagonal.

        MSE = E[(z - z_true)²] = bias² + variance (bias-variance decomposition).
        Falls back to std² if variance regression is not available.
        Distance is clamped to prevent extrapolation beyond evaluation data.
        """
        if self.use_polar and self.has_polar:
            polar_std = self._get_polar_std(getattr(self, 'polar_distal_bins', None), distance, angle_deg)
            if polar_std is not None:
                return polar_std**2

        clamped = self._clamp_distance(distance)
        if self.use_gpem_model and hasattr(self, '_model_data') and self._model_data.radial:
            mse = self._model_data.radial.predict_mse(clamped, self.use_quadratic)
            if mse is not None:
                return mse

        # Fallback: std²
        std = self.get_distal_std(distance, angle_deg)
        return std**2

    def get_perpendicular_mse(self, distance: float, angle_deg: float = None) -> float:
        """Get MSE (bias² + variance) for perpendicular error — used for R matrix diagonal."""
        if self.use_polar and self.has_polar:
            polar_std = self._get_polar_std(getattr(self, 'polar_perp_bins', None), distance, angle_deg)
            if polar_std is not None:
                return polar_std**2

        clamped = self._clamp_distance(distance)
        if self.use_gpem_model and hasattr(self, '_model_data') and self._model_data.lateral:
            mse = self._model_data.lateral.predict_mse(clamped, self.use_quadratic)
            if mse is not None:
                return mse

        std = self.get_perpendicular_std(distance, angle_deg)
        return std**2

    def get_yaw_mse(self, distance: float) -> float:
        """Get MSE for yaw error — used for heading covariance in CTRV filter."""
        clamped = self._clamp_distance(distance)
        if self.use_gpem_model and hasattr(self, '_model_data') and self._model_data.yaw:
            mse = self._model_data.yaw.predict_mse(clamped, self.use_quadratic)
            if mse is not None:
                return mse
        std = self.get_yaw_std(distance)
        return std**2

    def get_distal_std_average(self) -> float:
        """
        Get the count-weighted average standard deviation for distal error.
        Used by static covariance model (no distance prediction).

        Prefers the overall (0-N m) bin which is naturally count-weighted from
        the evaluation data. Falls back to unweighted average of per-distance bins.
        """
        if not self.distal_bins:
            return 0.1
        # Use the overall bin if available (wide bin spanning full range)
        for b in self.distal_bins:
            if (b.max_dist - b.min_dist) > 10:
                std = b.get_std()
                if std > 0:
                    return std
        # Fallback: unweighted average
        variances = [b.get_std()**2 for b in self.distal_bins if b.get_std() > 0]
        return np.sqrt(np.mean(variances)) if variances else 0.1

    def get_perpendicular_std_average(self) -> float:
        """
        Get the count-weighted average standard deviation for perpendicular error.
        Used by static covariance model (no distance prediction).
        """
        if not self.perp_bins:
            return 0.1
        # Use the overall bin if available
        for b in self.perp_bins:
            if (b.max_dist - b.min_dist) > 10:
                std = b.get_std()
                if std > 0:
                    return std
        variances = [bin_obj.get_std()**2 for bin_obj in self.perp_bins if bin_obj.get_std() > 0]
        return np.sqrt(np.mean(variances))
    
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
    
    def detection_probability(self, distance: float, angle_deg: float = None) -> float:
        """Calculate detection probability at the given distance (and angle).

        When polar miss bins are available, uses per-bin miss rate.
        Otherwise falls back to linear regression.
        """
        if distance > self.max_range:
            return 0.0

        # Try polar miss bins
        if self.has_polar and hasattr(self, 'polar_miss_bins') and angle_deg is not None:
            bin_obj = self._find_bin(self.polar_miss_bins, distance, angle_deg)
            if bin_obj:
                miss_rate = bin_obj.params.get("value", 0.5)
                return max(0.0, 1.0 - miss_rate)

        # Fallback: linear regression
        intercept, slope = self.miss_rate_regression
        miss_rate = intercept + slope * distance
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
        # Convert angle to degrees for polar bin lookup
        angle_deg = np.degrees(target_angle)

        # Sample errors in sensor frame
        distal_error = self.sample_distal_error(distance, angle_deg)
        perp_error = self.sample_perpendicular_error(distance, angle_deg)

        # Convert to global frame
        x_error = distal_error * np.cos(target_angle) - perp_error * np.sin(target_angle)
        y_error = distal_error * np.sin(target_angle) + perp_error * np.cos(target_angle)

        # Get expected standard deviations for covariance estimation
        distal_std = self.get_distal_std(distance, angle_deg)
        perp_std = self.get_perpendicular_std(distance, angle_deg)

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

    _FP_CLASSES = ["car", "truck", "bus", "construction_vehicle"]

    def sample_false_positives(
        self,
        sensor_x: float,
        sensor_y: float,
        sensor_yaw: float,
        max_fp_per_frame: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> List[dict]:
        """Sample false-positive detections for one sensor frame.

        Iterates over every calibration bin, draws n ~ Poisson(fp_per_frame),
        and for each FP samples a random (range, angle) within that bin, then
        converts to global position using the sensor pose.

        max_fp_per_frame: if set, randomly subsample the result down to this
            many FPs total. Useful to keep simulation tractable when calibrated
            rates are high (e.g. detr3d sums to ~166 FP/frame/sensor).

        Returns a list of dicts, each with:
            x, y          — global position
            width, length, height, yaw — sampled box dimensions
            score         — detection score in [0, 1]
            class_label   — one of "car", "truck", "bus", "construction_vehicle"
            range, angle_rad — sensor-relative position (for covariance building)
        """
        cal = (self._model_data.calibration
               if hasattr(self, "_model_data") and self._model_data else None)
        if cal is None or not cal.bins:
            return []

        fps: List[dict] = []
        for b in cal.bins:
            if b.fp_per_frame <= 0.0:
                continue
            n = np.random.poisson(b.fp_per_frame)
            if n == 0:
                continue

            # Build class distribution for this bin (fall back to uniform).
            class_counts = [
                b.n_fp_car, b.n_fp_truck, b.n_fp_bus, b.n_fp_construction_vehicle
            ]
            total_cls = sum(class_counts)
            if total_cls > 0:
                probs = [c / total_cls for c in class_counts]
            else:
                probs = [0.25, 0.25, 0.25, 0.25]

            fp_mu  = b.fp_score_mean if b.fp_score_mean is not None else cal._global_fp_mean
            # Cap per-bin std at global average: prevents sparse-bin noise and bimodal
            # bins from producing unrealistically wide score distributions.
            _raw_sig = b.fp_score_std if b.fp_score_std is not None else cal._global_fp_std
            fp_sig = max(0.05, min(_raw_sig, cal._global_fp_std))

            # Use Beta distribution: naturally bounded [0,1], avoids Gaussian tails
            # into the TP score range. Require both α>1 and β>1 (unimodal shape) which
            # holds when conc > 1/fp_mu. Fall back to global stats otherwise.
            _var = fp_sig * fp_sig
            _conc = fp_mu * (1.0 - fp_mu) / _var - 1.0
            _unimodal = _conc > 1.0 and fp_mu * _conc > 1.0
            if _unimodal:
                _beta_a = fp_mu * _conc
                _beta_b = (1.0 - fp_mu) * _conc
                _use_beta = True
            else:
                _g_sig = min(_raw_sig, cal._global_fp_std)
                _use_beta = False

            for _ in range(n):
                r     = random.uniform(b.range_lo, b.range_hi)
                a_deg = random.uniform(b.angle_lo_deg, b.angle_hi_deg)
                a_rad = math.radians(a_deg)
                global_angle = sensor_yaw + a_rad
                gx = sensor_x + r * math.cos(global_angle)
                gy = sensor_y + r * math.sin(global_angle)

                w = max(0.3, random.gauss(b.fp_width_mean,  max(b.fp_width_std,  1e-3)))
                l = max(0.5, random.gauss(b.fp_length_mean, max(b.fp_length_std, 1e-3)))
                h = max(0.1, random.gauss(b.fp_height_mean, max(b.fp_height_std, 1e-3)))
                yaw = random.gauss(0.0, max(b.fp_yaw_std, 1e-3))
                if _use_beta:
                    score = float(np.random.beta(_beta_a, _beta_b))
                else:
                    score = max(0.0, min(1.0, random.gauss(cal._global_fp_mean, _g_sig)))
                if score < score_threshold:
                    continue

                # Bayesian P(TP | score, range, angle) from calibration distributions
                p_tp = cal.tp_probability(r, a_rad, score)

                cls_idx = random.choices(range(4), weights=probs, k=1)[0]

                fps.append({
                    "x": gx, "y": gy,
                    "width": w, "length": l, "height": h, "yaw": yaw,
                    "score": score,
                    "p_tp": p_tp,
                    "class_label": self._FP_CLASSES[cls_idx],
                    "range": r, "angle_rad": global_angle,
                })

        if max_fp_per_frame is not None and len(fps) > max_fp_per_frame:
            fps = random.sample(fps, max_fp_per_frame)
        return fps


class CarSpecificErrorModel(DetectorErrorModel):
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


class PedestrianSpecificErrorModel(DetectorErrorModel):
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


class CyclistSpecificErrorModel(DetectorErrorModel):
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
                    use_quadratic: bool = False, use_polar: bool = False,
                    max_range: float = None) -> DetectorErrorModel:
    """
    Get an error model instance by name.

    Args:
        model_name: Model name matching a CSV file in src/data/sensor_models/
        force_reload: If True, reload from CSV even if cached
        use_gpem_model: If True, use GPEM parameterized covariance (distance-dependent).
        use_quadratic: If True and use_gpem_model=True, use quadratic regression.
        use_polar: If True, use polar (angle+distance) bin lookup for covariance.
        max_range: Maximum detection range in meters.

    Returns:
        The error model instance loaded from CSV
    """
    cache_key = f"{model_name}_gpem_{use_gpem_model}_quad_{use_quadratic}_polar_{use_polar}_range_{max_range}"
    global _error_models

    if force_reload and cache_key in _error_models:
        del _error_models[cache_key]

    if cache_key not in _error_models:
        model = DetectorErrorModel(
            model_name=model_name,
            load_from_csv=True,
            use_gpem_model=use_gpem_model,
            use_quadratic=use_quadratic,
            use_polar=use_polar,
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
