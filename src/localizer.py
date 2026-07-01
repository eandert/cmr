import utils
import math
import numpy as np
import csv
from pathlib import Path
from typing import Optional, List
from error import ErrorPackage, ErrorType

# Path to sensor model data
SENSOR_MODELS_DIR = Path(__file__).parent / "data" / "sensor_models"

# Available localizer models (maps config name to CSV base name)
LOCALIZER_MODELS = {
    "kiss_icp": "kiss_icp",
    "KISS_ICP": "kiss_icp",
    "kiss_icp_noisy": "kiss_icp_noisy",
    "KISS_ICP_NOISY": "kiss_icp_noisy",
    "orb_slam3": "kiss_icp_noisy",       # Placeholder: use noisy KISS-ICP until real ORB-SLAM3 data available
    "ORB_SLAM3": "kiss_icp_noisy",       # Placeholder: use noisy KISS-ICP until real ORB-SLAM3 data available
    "stereo_vo": "stereo_vo",
    "STEREO_VO": "stereo_vo",
    "CT_ICP": "kiss_icp",                # Use kiss_icp as proxy for CT_ICP (similar lidar-based)
    "ct_icp": "kiss_icp",
    "kiss_icp_accurate": "kiss_icp",     # kiss_icp IS the accurate cross-seq data now
    "orb_slam3_accurate": "kiss_icp_noisy",  # Placeholder: use noisy KISS-ICP until real ORB-SLAM3 data available
    "kiss_icp_clean_seed": "kiss_icp",       # Clean GPS seed → same ICP model
    "kiss_icp_noisy_seed": "kiss_icp_noisy", # Noisy GPS seed → noisy ICP model
}

# MAE to standard deviation conversion factor (half-normal distribution)
MAE_TO_STD = 1.2533141373


class VelocityBin:
    """A velocity bin with distribution parameters for error sampling."""
    
    def __init__(self, distribution: str, params: dict, min_vel: float, max_vel: float):
        self.distribution = distribution
        self.params = params
        self.min_vel = min_vel
        self.max_vel = max_vel
    
    def contains(self, velocity: float) -> bool:
        return self.min_vel <= velocity < self.max_vel
    
    def sample(self) -> float:
        """Sample an error from this bin's distribution."""
        if self.distribution == "normal":
            mean = self.params.get("mean", self.params.get("mu", 0))
            std = self.params.get("std", self.params.get("sigma", 0.1))
            return np.random.normal(mean, std)
        else:
            # Fallback to normal
            return np.random.normal(0, 0.1)
    
    def get_std(self) -> float:
        """Get the standard deviation for this bin."""
        if self.distribution == "normal":
            return self.params.get("std", self.params.get("sigma", 0.1))
        return 0.1


def load_localizer_distributions(localizer_type: str) -> dict:
    """Load velocity-binned distributions from CSV.

    Uses the strict ErrorModel loader underneath but returns the legacy dict
    of {error_type: [VelocityBin]} that the Localizer class expects.

    Raises FileNotFoundError / ErrorModelSchemaError / ErrorModelCoverageError
    if the CSV is malformed or missing — no silent {} fallback.
    """
    from error_model import ErrorModel
    csv_name = LOCALIZER_MODELS.get(localizer_type, localizer_type.lower())
    # Use linear mode for distributions access — we just need the bin list,
    # which is loaded for all modes. Velocity max_range 22 m/s matches the
    # canonical localizer characterization range.
    em = ErrorModel(csv_name, mode="linear",
                    independent_var="velocity", max_range=22.0)
    # Translate ErrorModel's internal _bins (keyed by public axis name like
    # 'lateral'/'longitudinal') to the legacy VelocityBin format. The
    # Localizer code below reads bins[error_type] where error_type matches
    # the legacy CSV-side names — same as the public axis names here.
    bins: dict[str, list[VelocityBin]] = {}
    for axis, bin_list in em._bins.items():
        out: list[VelocityBin] = []
        for b in bin_list:
            params = {"mean": float(b.params.get("mu", 0.0)),
                      "std":  float(b.params.get("sigma", 0.1))}
            out.append(VelocityBin(b.dist_type, params, b.dist_lo, b.dist_hi))
        out.sort(key=lambda v: v.min_vel)
        bins[axis] = out
    return bins


def load_localizer_model(localizer_type: str) -> dict:
    """
    Load localizer error model parameters from CSV.
    
    Args:
        localizer_type: Name of localizer (e.g., "kiss_icp", "orb_slam3", "CT_ICP")
    
    Returns:
        dict with keys:
            - lateral_linear: (intercept, slope) for lateral error
            - lateral_quadratic: (c, b, a) for lateral error (quad_c, quad_b, quad_a)
            - longitudinal_linear: (intercept, slope) for longitudinal/radial error
            - longitudinal_quadratic: (c, b, a) for longitudinal error
            - yaw_linear: (intercept, slope) for yaw error
            - yaw_quadratic: (c, b, a) for yaw error
    """
    csv_name = LOCALIZER_MODELS.get(localizer_type, localizer_type.lower())
    csv_path = SENSOR_MODELS_DIR / f"{csv_name}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Localizer model not found: {csv_path}")
    
    result = {}
    
    with open(csv_path) as f:
        # Skip comment lines
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    
    if not lines:
        raise ValueError(f"Empty localizer model file: {csv_path}")
    
    reader = csv.DictReader(lines)
    for row in reader:
        error_type = row.get('error_type', '')
        intercept = float(row.get('intercept', 0))
        slope = float(row.get('slope', 0))
        
        # Quadratic coefficients: error = quad_a * v^2 + quad_b * v + quad_c
        quad_a = float(row.get('quad_a', 0)) if row.get('quad_a') else None
        quad_b = float(row.get('quad_b', 0)) if row.get('quad_b') else None
        quad_c = float(row.get('quad_c', 0)) if row.get('quad_c') else None
        
        # Map error types to our naming
        if error_type == 'lateral':
            result['lateral_linear'] = (intercept, slope)
            if quad_a is not None:
                result['lateral_quadratic'] = (quad_c, quad_b, quad_a)  # c, b, a for polynomial
        elif error_type in ('radial', 'longitudinal'):
            result['longitudinal_linear'] = (intercept, slope)
            if quad_a is not None:
                result['longitudinal_quadratic'] = (quad_c, quad_b, quad_a)
        elif error_type == 'yaw':
            result['yaw_linear'] = (intercept, slope)
            if quad_a is not None:
                result['yaw_quadratic'] = (quad_c, quad_b, quad_a)
    
    return result


def get_localizer(localizer_type: str, error_package: 'ErrorPackage',
                  use_gpem_model: bool = False, use_quadratic: bool = False) -> 'Localizer':
    """
    Factory function to create a Localizer from a ground-truth-based model.
    
    Args:
        localizer_type: Name of localizer (e.g., "kiss_icp", "orb_slam3", "CT_ICP")
        error_package: ErrorPackage for error injection
        use_gpem_model: If True, use velocity-dependent covariance
        use_quadratic: If True and use_gpem_model=True, use quadratic model
    
    Returns:
        Configured Localizer instance
    """
    # No silent fallback: if the localizer CSV is missing, raise.
    # Use the alias table to resolve to the canonical CSV stem first; only
    # then does load_localizer_model raise FileNotFoundError when the
    # canonical CSV truly does not exist.
    model = load_localizer_model(localizer_type)
    
    # Load velocity-binned distributions
    distributions = load_localizer_distributions(localizer_type)
    
    # Extract coefficients for Polynomial (format: [c, b, a] for a*x^2 + b*x + c)
    lat_linear = model.get('lateral_linear', (0.025, 0.001))
    long_linear = model.get('longitudinal_linear', (0.025, 0.001))
    lat_quad = model.get('lateral_quadratic', None)
    long_quad = model.get('longitudinal_quadratic', None)
    
    # Linear polynomial: [intercept, slope] -> evaluates as intercept + slope*v
    lat_linear_coeffs = [lat_linear[0], lat_linear[1]]
    long_linear_coeffs = [long_linear[0], long_linear[1]]
    
    # Quadratic polynomial: [c, b, a] -> evaluates as c + b*v + a*v^2
    lat_quad_coeffs = list(lat_quad) if lat_quad else None
    long_quad_coeffs = list(long_quad) if long_quad else None
    
    # Load full model data for MSE computation (has bias + variance regressions)
    try:
        from sensor_model_loader import load_sensor_model
        csv_name = LOCALIZER_MODELS.get(localizer_type, localizer_type.lower())
        model_data = load_sensor_model(csv_name)
    except Exception:
        model_data = None

    return Localizer(
        lateral_error_coefficients=lat_linear_coeffs,
        longitudinal_error_coefficients=long_linear_coeffs,
        error_package=error_package,
        use_gpem_model=use_gpem_model,
        use_quadratic=use_quadratic,
        lateral_error_coefficients_quad=lat_quad_coeffs,
        longitudinal_error_coefficients_quad=long_quad_coeffs,
        velocity_bins=distributions,
        localizer_type=localizer_type,
        model_data=model_data,
    )

class Localizer:
    """
    Class representing a localizer with its properties and error models.
    
    Attributes:
        lateral_error_polynomial (Polynomial): Polynomial for lateral error (linear coefficients).
        longitudinal_error_polynomial (Polynomial): Polynomial for longitudinal error (linear coefficients).
        lateral_error_polynomial_quad (Polynomial): Polynomial for lateral error (quadratic coefficients).
        longitudinal_error_polynomial_quad (Polynomial): Polynomial for longitudinal error (quadratic coefficients).
        error_package (ErrorPackage): The error package to apply localization errors.
        use_gpem_model (bool): If True, use GPEM parameterized velocity-based covariance.
                              If False, use static averaged covariance.
        use_quadratic (bool): If True and use_gpem_model=True, use quadratic polynomial.
                             If False and use_gpem_model=True, use linear polynomial.
        velocity_bins (dict): Velocity-binned distributions for error sampling.
    
    Covariance estimation modes (velocity-based):
    - Static Covariance (use_gpem_model=False): Uses constant average covariance based on evaluation at 15 m/s
    - GPEM Linear (use_gpem_model=True, use_quadratic=False): Uses velocity bin lookup
    - GPEM Quadratic (use_gpem_model=True, use_quadratic=True): Uses velocity bin lookup
    
    Velocity bins are clamped at the top and bottom - if velocity exceeds the max bin,
    the max bin is used; if velocity is below the min bin, the min bin is used.
    """
    
    # Default quadratic coefficients for localizer error models
    # Format: [c, b, a] for a*v^2 + b*v + c
    DEFAULT_LATERAL_QUAD_COEFFICIENTS = [0.015, 0.0005, 0.00002]
    DEFAULT_LONGITUDINAL_QUAD_COEFFICIENTS = [0.02, 0.0008, 0.00003]
    
    def __init__(self, lateral_error_coefficients, longitudinal_error_coefficients, error_package,
                 use_gpem_model=False, use_quadratic=False,
                 lateral_error_coefficients_quad=None, longitudinal_error_coefficients_quad=None,
                 velocity_bins=None, localizer_type=None, model_data=None):
        """
        Initialize the localizer with its error models.
        
        Args:
            lateral_error_coefficients (list): Coefficients for the lateral error polynomial (linear).
            longitudinal_error_coefficients (list): Coefficients for the longitudinal error polynomial (linear).
            error_package (ErrorPackage): The error package to apply localization errors.
            use_gpem_model (bool): If True, use velocity-dependent GPEM covariance.
            use_quadratic (bool): If True and use_gpem_model=True, use quadratic polynomial.
            lateral_error_coefficients_quad (list, optional): Quadratic coefficients for lateral error.
            longitudinal_error_coefficients_quad (list, optional): Quadratic coefficients for longitudinal error.
            velocity_bins (dict, optional): Velocity-binned distributions loaded from CSV.
            localizer_type (str, optional): Name of the localizer for logging.
        """
        # Linear polynomial coefficients (fallback if bins not available)
        self.lateral_error_polynomial = utils.Polynomial(lateral_error_coefficients)
        self.longitudinal_error_polynomial = utils.Polynomial(longitudinal_error_coefficients)
        
        # Quadratic polynomial coefficients (fallback if bins not available)
        lat_quad_coeffs = lateral_error_coefficients_quad or self.DEFAULT_LATERAL_QUAD_COEFFICIENTS
        long_quad_coeffs = longitudinal_error_coefficients_quad or self.DEFAULT_LONGITUDINAL_QUAD_COEFFICIENTS
        self.lateral_error_polynomial_quad = utils.Polynomial(lat_quad_coeffs)
        self.longitudinal_error_polynomial_quad = utils.Polynomial(long_quad_coeffs)
        
        self.error_package = error_package
        self.use_gpem_model = use_gpem_model
        self.use_quadratic = use_quadratic
        self.localizer_type = localizer_type

        # Instance-level MAE→STD factor (overridden to 1.0 when std-from-bins is used)
        self.mae_to_std = MAE_TO_STD

        # Velocity-binned distributions (support both 'longitudinal' and 'radial' naming)
        self.velocity_bins = velocity_bins or {}
        self.lateral_bins = self.velocity_bins.get('lateral', [])
        self.longitudinal_bins = (self.velocity_bins.get('longitudinal', [])
                                  or self.velocity_bins.get('radial', []))

        # Store full model data for MSE computation (bias² + variance)
        self._model_data = model_data

        # Ornstein-Uhlenbeck drift state for realistic localization error
        # Instead of IID noise per frame, errors persist and drift slowly
        # theta: mean-reversion rate (lower = more persistent drift)
        # At theta=0.5 with dt=0.1, autocorrelation ~0.95 between frames (~2s half-life)
        self._ou_theta = 0.5       # Mean-reversion rate (1/s)
        self._ou_dt = 0.1          # Timestep (matches SUMO step_length)
        self._ou_long_state = 0.0  # Current longitudinal drift (meters)
        self._ou_lat_state = 0.0   # Current lateral drift (meters)

        # Refit regressions from bin stds if bins are available
        if self.lateral_bins or self.longitudinal_bins:
            self._refit_regressions_from_bins()
    
    def _refit_regressions_from_bins(self):
        """
        Refit regression coefficients to predict std directly from velocity bin stds.

        Instead of: MAE = intercept + slope * v → std = MAE * 1.2533
        We now do:  std = intercept + slope * v (fitted on per-bin get_std() values)

        This eliminates the MAE→std conversion factor assumption.
        """
        self.mae_to_std = 1.0

        if self.longitudinal_bins:
            vel, std = self._bins_to_velocity_std(self.longitudinal_bins)
            if len(vel) >= 2:
                lin = np.polyfit(vel, std, 1)  # [slope, intercept]
                self.longitudinal_error_polynomial = utils.Polynomial([float(lin[1]), float(lin[0])])
                if len(vel) >= 3:
                    quad = np.polyfit(vel, std, 2)  # [a, b, c]
                    self.longitudinal_error_polynomial_quad = utils.Polynomial(
                        [float(quad[2]), float(quad[1]), float(quad[0])])

        if self.lateral_bins:
            vel, std = self._bins_to_velocity_std(self.lateral_bins)
            if len(vel) >= 2:
                lin = np.polyfit(vel, std, 1)
                self.lateral_error_polynomial = utils.Polynomial([float(lin[1]), float(lin[0])])
                if len(vel) >= 3:
                    quad = np.polyfit(vel, std, 2)
                    self.lateral_error_polynomial_quad = utils.Polynomial(
                        [float(quad[2]), float(quad[1]), float(quad[0])])

    @staticmethod
    def _bins_to_velocity_std(bins):
        """Extract (velocity_centers, stds) arrays from velocity bins, filtering overall bins."""
        velocities = []
        stds = []
        for b in bins:
            if (b.max_vel - b.min_vel) > 10:  # Skip "overall" bins
                continue
            center = (b.min_vel + b.max_vel) / 2.0
            s = b.get_std()
            if s > 0:
                velocities.append(center)
                stds.append(s)
        return np.array(velocities), np.array(stds)

    def _find_bin(self, bins: List[VelocityBin], velocity: float) -> Optional[VelocityBin]:
        """
        Find the appropriate bin for a given velocity with clamping.
        
        Prefers narrower bins over wide "overall" bins.
        If velocity exceeds max specific bin, returns the max specific bin.
        If velocity is below min specific bin, returns the min specific bin.
        """
        if not bins:
            return None
        
        # Filter out "overall" bins (those spanning > 10 m/s range)
        specific_bins = [b for b in bins if (b.max_vel - b.min_vel) <= 10]
        
        if not specific_bins:
            # Fallback to any matching bin
            for bin_obj in bins:
                if bin_obj.contains(velocity):
                    return bin_obj
            return None
        
        # Find exact match in specific bins
        matching_bins = [b for b in specific_bins if b.contains(velocity)]
        if matching_bins:
            # Prefer the narrowest bin
            return min(matching_bins, key=lambda b: b.max_vel - b.min_vel)
        
        # Clamp to min/max specific bin
        min_bin = min(specific_bins, key=lambda b: b.min_vel)
        max_bin = max(specific_bins, key=lambda b: b.max_vel)
        
        if velocity < min_bin.min_vel:
            return min_bin
        if velocity >= max_bin.max_vel:
            return max_bin
        
        return None
    
    def _get_bin_std(self, bins: List[VelocityBin], velocity: float, fallback_polynomial) -> float:
        """Get std from velocity bin, with fallback to polynomial."""
        bin_obj = self._find_bin(bins, velocity)
        if bin_obj:
            return bin_obj.get_std()
        # Fallback to polynomial
        mae = abs(fallback_polynomial.evaluate(velocity))
        return max(0.001, mae * self.mae_to_std)
    
    def _sample_from_bin(self, bins: List[VelocityBin], velocity: float, fallback_polynomial) -> float:
        """Sample error from velocity bin, with fallback to polynomial."""
        bin_obj = self._find_bin(bins, velocity)
        if bin_obj:
            # Sample from a normal centered at 0 with bin's std
            # (bin stores mean error magnitude, we want zero-mean for sampling)
            return np.random.normal(0, bin_obj.get_std())
        # Fallback to polynomial-derived std
        mae = abs(fallback_polynomial.evaluate(velocity))
        std = max(0.001, mae * self.mae_to_std)
        return np.random.normal(0, std)
        
    def lateral_error_at_velocity(self, velocity, use_quadratic=None):
        """
        Get the lateral error at a given velocity (using polynomial - for backward compat).
        """
        quad = use_quadratic if use_quadratic is not None else self.use_quadratic
        if quad:
            return self.lateral_error_polynomial_quad.evaluate(velocity)
        return self.lateral_error_polynomial.evaluate(velocity)
        
    def longitudinal_error_at_velocity(self, velocity, use_quadratic=None):
        """
        Get the longitudinal error at a given velocity (using polynomial - for backward compat).
        """
        quad = use_quadratic if use_quadratic is not None else self.use_quadratic
        if quad:
            return self.longitudinal_error_polynomial_quad.evaluate(velocity)
        return self.longitudinal_error_polynomial.evaluate(velocity)
    
    def get_lateral_localization_std(self, velocity: float) -> float:
        """
        Get the standard deviation for lateral localization error at a given velocity.
        
        Uses regression for covariance ESTIMATION (static/linear/quadratic modes).
        Bins are used only for error SAMPLING (actual error injection).
        
        Modes:
        - Static (use_gpem_model=False): Constant value at 15 m/s
        - GPEM Linear (use_gpem_model=True, use_quadratic=False): Linear regression
        - GPEM Quadratic (use_gpem_model=True, use_quadratic=True): Quadratic regression
        """
        if self.use_gpem_model:
            # Use regression for covariance estimation (same as detector)
            abs_error = abs(self.lateral_error_at_velocity(velocity))
            return max(0.001, abs_error * self.mae_to_std)
        else:
            return self._get_lateral_std_average()
    
    def get_longitudinal_localization_std(self, velocity: float) -> float:
        """
        Get the standard deviation for longitudinal localization error at a given velocity.
        
        Uses regression for covariance ESTIMATION (static/linear/quadratic modes).
        Bins are used only for error SAMPLING (actual error injection).
        
        Modes:
        - Static (use_gpem_model=False): Constant value at 15 m/s
        - GPEM Linear (use_gpem_model=True, use_quadratic=False): Linear regression
        - GPEM Quadratic (use_gpem_model=True, use_quadratic=True): Quadratic regression
        """
        if self.use_gpem_model:
            # Use regression for covariance estimation (same as detector)
            abs_error = abs(self.longitudinal_error_at_velocity(velocity))
            return max(0.001, abs_error * self.mae_to_std)
        else:
            return self._get_longitudinal_std_average()
    
    def _clamp_velocity(self, velocity: float) -> float:
        """Clamp velocity to the range covered by evaluation bins.

        Prevents regression extrapolation beyond measured data (e.g., highway
        speeds beyond the 22 m/s max of the evaluation dataset).
        """
        bins = self.longitudinal_bins or self.lateral_bins
        if not bins:
            return velocity
        specific = [b for b in bins if (b.max_vel - b.min_vel) <= 10]
        if not specific:
            return velocity
        max_v = max(b.max_vel for b in specific)
        min_v = min(b.min_vel for b in specific)
        return max(min_v, min(velocity, max_v))

    def get_longitudinal_mse(self, velocity: float) -> float:
        """Get MSE (bias² + variance) for longitudinal error. Falls back to std²."""
        velocity = self._clamp_velocity(velocity)
        if self.use_gpem_model and self._model_data:
            params = self._model_data.params.get('longitudinal') or self._model_data.params.get('radial')
            if params and params.has_var_regression:
                mse = params.predict_mse(velocity, self.use_quadratic)
                if mse is not None:
                    return mse
        std = self.get_longitudinal_localization_std(velocity)
        return std**2

    def get_lateral_mse(self, velocity: float) -> float:
        """Get MSE (bias² + variance) for lateral error. Falls back to std²."""
        velocity = self._clamp_velocity(velocity)
        if self.use_gpem_model and self._model_data:
            params = self._model_data.params.get('lateral')
            if params and params.has_var_regression:
                mse = params.predict_mse(velocity, self.use_quadratic)
                if mse is not None:
                    return mse
        std = self.get_lateral_localization_std(velocity)
        return std**2

    def get_localization_covariance(self, velocity: float, yaw: float) -> np.ndarray:
        """
        Get the 2x2 localization covariance matrix rotated by the vehicle's yaw.

        Uses MSE (bias² + variance) when available, else std².
        """
        import gaussians

        long_mse = self.get_longitudinal_mse(velocity)
        lat_mse = self.get_lateral_mse(velocity)

        g = gaussians.BivariateGaussian(long_mse, lat_mse, yaw)
        return g.covariance

    def _calculate_velocity_based_std(self, velocity: float) -> float:
        """Calculate average of lateral and longitudinal std devs."""
        lat_std = self.get_lateral_localization_std(velocity)
        long_std = self.get_longitudinal_localization_std(velocity)
        return (lat_std + long_std) / 2.0
    
    def _get_lateral_std_average(self) -> float:
        """
        Get the average standard deviation for lateral localization error.
        Used by static covariance model (no velocity prediction).
        Averages variance across all specific bins, then sqrt (Jensen's inequality fix).
        """
        if self.lateral_bins:
            variances = [b.get_std()**2 for b in self.lateral_bins
                         if (b.max_vel - b.min_vel) <= 10]
            if variances:
                return np.sqrt(np.mean(variances))
        mae = abs(self.lateral_error_polynomial.evaluate(15.0))
        return max(0.001, mae * self.mae_to_std)

    def _get_longitudinal_std_average(self) -> float:
        """
        Get the average standard deviation for longitudinal localization error.
        Used by static covariance model (no velocity prediction).
        Averages variance across all specific bins, then sqrt (Jensen's inequality fix).
        """
        if self.longitudinal_bins:
            variances = [b.get_std()**2 for b in self.longitudinal_bins
                         if (b.max_vel - b.min_vel) <= 10]
            if variances:
                return np.sqrt(np.mean(variances))
        mae = abs(self.longitudinal_error_polynomial.evaluate(15.0))
        return max(0.001, mae * self.mae_to_std)

    def sample_lateral_error(self, velocity: float) -> float:
        """
        Sample a lateral localization error for the given velocity.
        Uses velocity bin distribution when available.
        """
        if self.lateral_bins:
            return self._sample_from_bin(self.lateral_bins, velocity, self.lateral_error_polynomial)
        std = self.get_lateral_localization_std(velocity)
        return np.random.normal(0, std)
    
    def sample_longitudinal_error(self, velocity: float) -> float:
        """
        Sample a longitudinal localization error for the given velocity.
        Uses velocity bin distribution when available.
        """
        if self.longitudinal_bins:
            return self._sample_from_bin(self.longitudinal_bins, velocity, self.longitudinal_error_polynomial)
        std = self.get_longitudinal_localization_std(velocity)
        return np.random.normal(0, std)

    def get_localization_pose(self, ground_truth_x, ground_truth_y, ground_truth_yaw, velocity, has_error=False):
        """
        Get the localized pose with persistent, slowly-drifting errors.

        Uses an Ornstein-Uhlenbeck process so that localization error persists
        across frames and drifts slowly, rather than being IID per frame.
        This is more realistic: SLAM/ICP localizers have pose estimates that
        drift gradually, not jump randomly each frame.

        The OU process ensures:
        - Long-term variance matches the characterized error distribution
        - Frame-to-frame correlation is high (~0.95 at dt=0.1s, theta=0.5)
        - Drift half-life is ~1.4s (mean-reversion keeps it bounded)

        Args:
            ground_truth_x (float): The ground truth x-coordinate.
            ground_truth_y (float): The ground truth y-coordinate.
            ground_truth_yaw (float): The ground truth yaw angle (standard math convention).
            velocity (float): The current velocity.
            has_error (bool): Whether this CAV has permanent error flag.

        Returns:
            tuple: The localized pose as (x, y, yaw) with injected errors.
        """
        # Get the target std for this velocity (from characterized error model)
        long_std = abs(self.get_longitudinal_localization_std(velocity))
        lat_std = abs(self.get_lateral_localization_std(velocity))

        # Ornstein-Uhlenbeck update: dx = -theta * x * dt + sigma * sqrt(dt) * N(0,1)
        # Stationary variance of OU = sigma² / (2*theta), so sigma = std * sqrt(2*theta)
        #
        # Theta scales with velocity: slower vehicles have less odometry data,
        # so drift persists longer (lower theta = slower mean-reversion).
        # At 0 m/s: theta = 0.125 (half-life ~5.5s, 4x more persistent)
        # At 25 m/s: theta = 0.5 (half-life ~1.4s, normal)
        dt = self._ou_dt
        max_speed = 25.0  # m/s, roughly highway speed
        speed_ratio = min(velocity / max_speed, 1.0)
        theta = self._ou_theta * (0.25 + 0.75 * speed_ratio)  # range: [0.125, 0.5]
        ou_sigma_long = long_std * math.sqrt(2.0 * theta)
        ou_sigma_lat = lat_std * math.sqrt(2.0 * theta)

        self._ou_long_state += -theta * self._ou_long_state * dt + ou_sigma_long * math.sqrt(dt) * np.random.normal()
        self._ou_lat_state += -theta * self._ou_lat_state * dt + ou_sigma_lat * math.sqrt(dt) * np.random.normal()

        longitudinal_error = self._ou_long_state
        lateral_error = self._ou_lat_state

        adjusted_x = ground_truth_x + longitudinal_error * math.cos(ground_truth_yaw) - lateral_error * math.sin(ground_truth_yaw)
        adjusted_y = ground_truth_y + longitudinal_error * math.sin(ground_truth_yaw) + lateral_error * math.cos(ground_truth_yaw)
        adjusted_yaw = ground_truth_yaw

        localization_data = {'x': adjusted_x, 'y': adjusted_y, 'yaw': adjusted_yaw}

        if self.error_package.error_type == ErrorType.LOCALIZATION:
            localization_data = self.error_package.inject_error(localization_data, has_error)

        return localization_data['x'], localization_data['y'], localization_data['yaw']
