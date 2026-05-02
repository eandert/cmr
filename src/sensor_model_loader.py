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
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


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
    # Bias (signed error) regression: bias = bias_intercept + bias_slope * d
    bias_intercept: Optional[float] = None
    bias_slope: Optional[float] = None
    # Bias quadratic: bias = bias_quad_a * d^2 + bias_quad_b * d + bias_quad_c
    bias_quad_a: Optional[float] = None
    bias_quad_b: Optional[float] = None
    bias_quad_c: Optional[float] = None
    # Std regression (direct, no MAE conversion needed): std = std_intercept + std_slope * d
    std_intercept: Optional[float] = None
    std_slope: Optional[float] = None
    # Std quadratic: std = std_quad_a * d^2 + std_quad_b * d + std_quad_c
    std_quad_a: Optional[float] = None
    std_quad_b: Optional[float] = None
    std_quad_c: Optional[float] = None
    # Variance regression (direct): var = var_intercept + var_slope * d
    var_intercept: Optional[float] = None
    var_slope: Optional[float] = None
    # Variance quadratic: var = var_quad_a * d^2 + var_quad_b * d + var_quad_c
    var_quad_a: Optional[float] = None
    var_quad_b: Optional[float] = None
    var_quad_c: Optional[float] = None

    @property
    def has_quadratic(self) -> bool:
        return self.quad_a is not None and self.quad_b is not None and self.quad_c is not None

    @property
    def has_bias(self) -> bool:
        return self.bias_intercept is not None and self.bias_slope is not None

    @property
    def has_bias_quadratic(self) -> bool:
        return self.bias_quad_a is not None and self.bias_quad_b is not None and self.bias_quad_c is not None

    @property
    def has_std_regression(self) -> bool:
        return self.std_intercept is not None and self.std_slope is not None

    @property
    def has_std_quadratic(self) -> bool:
        return self.std_quad_a is not None and self.std_quad_b is not None and self.std_quad_c is not None

    @property
    def has_var_regression(self) -> bool:
        return self.var_intercept is not None and self.var_slope is not None

    @property
    def has_var_quadratic(self) -> bool:
        return self.var_quad_a is not None and self.var_quad_b is not None and self.var_quad_c is not None

    def predict_linear(self, distance: float) -> float:
        """Predict absolute error (MAE) using linear model."""
        return max(0.001, self.intercept + self.slope * distance)

    def predict_quadratic(self, distance: float) -> float:
        """Predict absolute error (MAE) using quadratic model."""
        if not self.has_quadratic:
            return self.predict_linear(distance)
        return max(0.001, self.quad_a * distance**2 + self.quad_b * distance + self.quad_c)

    def predict(self, distance: float, use_quadratic: bool = False) -> float:
        """Predict absolute error (MAE) at the given distance."""
        if use_quadratic and self.has_quadratic:
            return self.predict_quadratic(distance)
        return self.predict_linear(distance)

    def predict_std_direct(self, distance: float, use_quadratic: bool = False) -> Optional[float]:
        """Predict std directly from std regression (no MAE conversion). Returns None if unavailable."""
        if use_quadratic and self.has_std_quadratic:
            return max(0.001, self.std_quad_a * distance**2 + self.std_quad_b * distance + self.std_quad_c)
        if self.has_std_regression:
            return max(0.001, self.std_intercept + self.std_slope * distance)
        return None

    def predict_var_direct(self, distance: float, use_quadratic: bool = False) -> Optional[float]:
        """Predict variance directly from variance regression. Returns None if unavailable."""
        if use_quadratic and self.has_var_quadratic:
            return max(1e-6, self.var_quad_a * distance**2 + self.var_quad_b * distance + self.var_quad_c)
        if self.has_var_regression:
            return max(1e-6, self.var_intercept + self.var_slope * distance)
        return None

    def predict_mse(self, distance: float, use_quadratic: bool = False) -> Optional[float]:
        """Predict MSE = bias² + variance (for R matrix). Returns None if variance unavailable."""
        var = self.predict_var_direct(distance, use_quadratic=use_quadratic)
        if var is None:
            return None
        bias = self.predict_bias(distance, use_quadratic=use_quadratic)
        return bias**2 + var

    def predict_std(self, distance: float, mae_to_std: float = 1.2533141373, use_quadratic: bool = False) -> float:
        """Predict std — uses direct std regression if available, else MAE * conversion factor."""
        direct = self.predict_std_direct(distance, use_quadratic=use_quadratic)
        if direct is not None:
            return direct
        return self.predict(distance, use_quadratic=use_quadratic) * mae_to_std

    def predict_bias(self, distance: float, use_quadratic: bool = False) -> float:
        """Predict signed bias at the given distance. Returns 0 if no bias regression."""
        if not self.has_bias:
            return 0.0
        if use_quadratic and self.has_bias_quadratic:
            return self.bias_quad_a * distance**2 + self.bias_quad_b * distance + self.bias_quad_c
        return self.bias_intercept + self.bias_slope * distance


@dataclass
class DistributionBinData:
    """A single distribution bin for error sampling."""
    error_type: str
    dist_min: float
    dist_max: float
    distribution: str  # normal, laplace, logistic, student_t, cauchy, constant
    params: Dict[str, float]  # distribution-specific parameters
    angle_min: Optional[float] = None  # polar bins: angle range in degrees
    angle_max: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            'error_type': self.error_type,
            'dist_min': self.dist_min,
            'dist_max': self.dist_max,
            'distribution': self.distribution,
            'params': self.params
        }
        if self.angle_min is not None:
            d['angle_min'] = self.angle_min
            d['angle_max'] = self.angle_max
        return d


@dataclass
class CalibrationBin:
    """One (range, angle) bin of detector TP/FP calibration data."""
    range_lo: float          # m
    range_hi: float
    angle_lo_deg: float      # degrees, [-180, 180)
    angle_hi_deg: float
    gt_count: int            # GT objects in this bin during characterization
    n_missed: int            # of those, missed by detector
    miss_rate: float         # n_missed / gt_count (or fallback global)
    n_fp: int                # FP detections in this bin
    fp_per_frame: float      # Poisson rate λ_FP per ego-frame
    score_mean: Optional[float]      # TP score N(μ, σ) — None if no data
    score_std: Optional[float]
    fp_score_mean: Optional[float]   # FP score N(μ, σ) — None if no data
    fp_score_std: Optional[float]
    source: str              # "measured" or "fallback"
    # FP shape distributions (added by enrich_fp_calibration / import_sensor_models)
    fp_width_mean: float  = 0.0
    fp_width_std:  float  = 0.5
    fp_length_mean: float = 0.0
    fp_length_std:  float = 0.5
    fp_height_mean: float = 0.0
    fp_height_std:  float = 0.3
    fp_yaw_std:    float  = 1.5708  # π/2 — flat prior if not calibrated
    # FP class counts (multinomial prior for class sampling)
    n_fp_car:                  int = 0
    n_fp_truck:                int = 0
    n_fp_bus:                  int = 0
    n_fp_construction_vehicle: int = 0


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


class CalibrationTable:
    """Per-detector P(TP | range, angle, score) lookup.

    Built from upstream `polar_binned_errors.csv`. Per (range, angle) bin we
    have empirical TP and FP score distributions plus per-bin counts. At
    runtime we apply Bayes:

        P(TP | bin, score) = (n_TP · L_TP(score)) / (n_TP · L_TP(score) + n_FP · L_FP(score))

    where n_TP = gt_count − n_missed (Laplace-smoothed) and n_FP = n_fp.
    The n_frames factor cancels.

    Bins flagged `source == "fallback"` have no measured score data; those
    fall back to the globally-pooled TP/FP score distributions across
    measured bins, so the calibration degrades gracefully in sparse regions.
    """

    MIN_SIGMA = 0.05    # Floor on Gaussian σ to avoid div-by-zero on narrow bins
    LAPLACE   = 1.0     # Pseudo-count smoothing on n_TP / n_FP

    def __init__(self, bins: List[CalibrationBin]):
        self.bins = bins
        self._index: Dict[Tuple[int, int], CalibrationBin] = {}
        if not bins:
            self._range_min = 0.0
            self._range_step = 5.0
            self._angle_min = -180.0
            self._angle_step = 10.0
            self._global_tp_mean = 0.7
            self._global_fp_mean = 0.3
            self._global_tp_std = 0.15
            self._global_fp_std = 0.15
            return

        range_steps = sorted({b.range_lo for b in bins})
        angle_steps = sorted({b.angle_lo_deg for b in bins})
        self._range_min = range_steps[0]
        # Single-bin tables fall back to the bin's own (hi - lo) span; multi-
        # bin tables use the inter-bin gap (uniform-grid assumption).
        if len(range_steps) > 1:
            self._range_step = range_steps[1] - range_steps[0]
        else:
            self._range_step = bins[0].range_hi - bins[0].range_lo
        self._angle_min = angle_steps[0]
        if len(angle_steps) > 1:
            self._angle_step = angle_steps[1] - angle_steps[0]
        else:
            self._angle_step = bins[0].angle_hi_deg - bins[0].angle_lo_deg
        self._range_max = max(b.range_hi for b in bins)

        for b in bins:
            # Floor-divide for both indexing and lookup keeps them aligned.
            r_idx = int((b.range_lo - self._range_min) / self._range_step)
            a_idx = int((b.angle_lo_deg - self._angle_min) / self._angle_step)
            self._index[(r_idx, a_idx)] = b

        # Global pooled fallbacks — averaged across all `measured` bins.
        tp_means = [b.score_mean for b in bins
                    if b.source == "measured" and b.score_mean is not None]
        fp_means = [b.fp_score_mean for b in bins
                    if b.source == "measured" and b.fp_score_mean is not None]
        tp_stds  = [b.score_std for b in bins
                    if b.source == "measured" and b.score_std is not None and b.score_std > 0]
        fp_stds  = [b.fp_score_std for b in bins
                    if b.source == "measured" and b.fp_score_std is not None and b.fp_score_std > 0]
        self._global_tp_mean = sum(tp_means) / len(tp_means) if tp_means else 0.7
        self._global_fp_mean = sum(fp_means) / len(fp_means) if fp_means else 0.3
        self._global_tp_std  = max(sum(tp_stds) / len(tp_stds) if tp_stds else 0.15, self.MIN_SIGMA)
        self._global_fp_std  = max(sum(fp_stds) / len(fp_stds) if fp_stds else 0.15, self.MIN_SIGMA)

    def lookup_bin(self, distance_m: float, angle_rad: float) -> Optional[CalibrationBin]:
        """Find the bin a (range, angle) lookup falls into. Angle is wrapped
        to [-180, 180) to match the upstream CSV convention."""
        angle_deg = math.degrees(angle_rad)
        while angle_deg < -180.0:
            angle_deg += 360.0
        while angle_deg >= 180.0:
            angle_deg -= 360.0
        r_idx = int((distance_m - self._range_min) / self._range_step)
        a_idx = int((angle_deg - self._angle_min) / self._angle_step)
        return self._index.get((r_idx, a_idx))

    def tp_probability(self, distance_m: float, angle_rad: float, score: float) -> float:
        """Calibrated P(TP | distance, angle, score) ∈ [0, 1].

        Returns 1.0 if no bin matches (out of range) — equivalent to "no
        calibration available, don't gate". This makes the function safe
        to call unconditionally; the caller can decide whether to treat
        the absence of calibration as a no-op or a hard pass-through.
        """
        b = self.lookup_bin(distance_m, angle_rad)
        if b is None:
            return 1.0

        n_tp = max(0.0, float(b.gt_count - b.n_missed)) + self.LAPLACE
        n_fp = float(b.n_fp) + self.LAPLACE

        tp_mu    = b.score_mean if b.score_mean is not None else self._global_tp_mean
        tp_sigma = max(b.score_std if b.score_std is not None else self._global_tp_std, self.MIN_SIGMA)
        fp_mu    = b.fp_score_mean if b.fp_score_mean is not None else self._global_fp_mean
        fp_sigma = max(b.fp_score_std if b.fp_score_std is not None else self._global_fp_std, self.MIN_SIGMA)

        # Compute likelihoods. For extreme-tail scores (e.g., score=0.20 in a
        # bin whose TP scores cluster around 0.75 σ=0.1 — five standard
        # deviations away), one or both PDFs can underflow to 0.0, which
        # would naively give 0/0 → ambiguous. Fall back to the count-based
        # prior in that case (i.e., assume the score gave no information,
        # use the empirical TP/FP ratio).
        L_tp = _gaussian_pdf(score, tp_mu, tp_sigma)
        L_fp = _gaussian_pdf(score, fp_mu, fp_sigma)

        num   = n_tp * L_tp
        denom = num + n_fp * L_fp
        if denom > 0.0:
            return num / denom

        # Both likelihoods underflowed. The Bayes posterior in log space:
        #   P_TP = sigmoid( log(n_TP/n_FP) + log(L_TP/L_FP) )
        # Compute log(L_TP/L_FP) directly since we can do the difference of
        # exponents without underflow.
        z_tp = (score - tp_mu) / tp_sigma
        z_fp = (score - fp_mu) / fp_sigma
        log_lr = (-0.5 * z_tp * z_tp + math.log(1.0 / tp_sigma)) - \
                 (-0.5 * z_fp * z_fp + math.log(1.0 / fp_sigma))
        log_post_odds = math.log(n_tp / n_fp) + log_lr
        if log_post_odds >= 0:
            return 1.0 / (1.0 + math.exp(-log_post_odds))
        e = math.exp(log_post_odds)
        return e / (1.0 + e)


def load_calibration_csv(file_path: str) -> CalibrationTable:
    """Parse upstream `polar_binned_errors.csv` into a CalibrationTable.

    Schema (relevant columns; extra columns are ignored):
        range_lo, range_hi, angle_lo, angle_hi,
        gt_count, missed, miss_rate, source, n_fp, fp_per_frame,
        score_mean, score_std, fp_score_mean, fp_score_std
        [optional] fp_width_mean/std, fp_length_mean/std, fp_height_mean/std,
                   fp_yaw_std, n_fp_car, n_fp_truck, n_fp_bus,
                   n_fp_construction_vehicle

    Empty / "nan" / non-numeric score fields become None and trigger the
    pooled-global fallback at lookup time.
    """
    def _opt_float(s: str) -> Optional[float]:
        s = (s or "").strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _flt(row: dict, key: str, default: float) -> float:
        v = _opt_float(row.get(key, ""))
        return v if v is not None else default

    def _int0(row: dict, key: str) -> int:
        return int(float(row.get(key, "0") or "0"))

    bins: List[CalibrationBin] = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bins.append(CalibrationBin(
                range_lo=float(row["range_lo"]),
                range_hi=float(row["range_hi"]),
                angle_lo_deg=float(row["angle_lo"]),
                angle_hi_deg=float(row["angle_hi"]),
                gt_count=_int0(row, "gt_count"),
                n_missed=_int0(row, "missed"),
                miss_rate=float(row.get("miss_rate", "0") or "0"),
                n_fp=_int0(row, "n_fp"),
                fp_per_frame=float(row.get("fp_per_frame", "0") or "0"),
                score_mean=_opt_float(row.get("score_mean", "")),
                score_std=_opt_float(row.get("score_std", "")),
                fp_score_mean=_opt_float(row.get("fp_score_mean", "")),
                fp_score_std=_opt_float(row.get("fp_score_std", "")),
                source=(row.get("source", "measured") or "measured").strip(),
                fp_width_mean=_flt(row, "fp_width_mean", 0.0),
                fp_width_std =_flt(row, "fp_width_std",  0.5),
                fp_length_mean=_flt(row, "fp_length_mean", 0.0),
                fp_length_std =_flt(row, "fp_length_std",  0.5),
                fp_height_mean=_flt(row, "fp_height_mean", 0.0),
                fp_height_std =_flt(row, "fp_height_std",  0.3),
                fp_yaw_std    =_flt(row, "fp_yaw_std",     1.5708),
                n_fp_car                 =_int0(row, "n_fp_car"),
                n_fp_truck               =_int0(row, "n_fp_truck"),
                n_fp_bus                 =_int0(row, "n_fp_bus"),
                n_fp_construction_vehicle=_int0(row, "n_fp_construction_vehicle"),
            ))
    return CalibrationTable(bins)


@dataclass
class SensorModelData:
    """Complete sensor model data loaded from CSV."""
    name: str
    file_path: str
    params: Dict[str, RegressionParams]
    # Distribution bins for sampling (loaded from *_distributions.csv)
    distribution_bins: Dict[str, List[DistributionBinData]] = None
    # P(TP | range, angle, score) calibration (loaded from
    # *_polar_calibration.csv if present; None if no calibration available).
    calibration: Optional[CalibrationTable] = None

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

            # Parse bias regression coefficients (optional)
            bias_intercept = None
            bias_slope = None
            if 'bias_intercept' in row and row['bias_intercept'].strip():
                try:
                    bias_intercept = float(row['bias_intercept'])
                    bias_slope = float(row['bias_slope'])
                except (ValueError, KeyError):
                    pass

            bias_quad_a = None
            bias_quad_b = None
            bias_quad_c = None
            if 'bias_quad_a' in row and row.get('bias_quad_a', '').strip():
                try:
                    bias_quad_a = float(row['bias_quad_a'])
                    bias_quad_b = float(row['bias_quad_b'])
                    bias_quad_c = float(row['bias_quad_c'])
                except (ValueError, KeyError):
                    pass

            # Parse std regression (direct, no MAE conversion)
            std_intercept = None
            std_slope = None
            if 'std_intercept' in row and row.get('std_intercept', '').strip():
                try:
                    std_intercept = float(row['std_intercept'])
                    std_slope = float(row['std_slope'])
                except (ValueError, KeyError):
                    pass

            std_quad_a = None
            std_quad_b = None
            std_quad_c = None
            if 'std_quad_a' in row and row.get('std_quad_a', '').strip():
                try:
                    std_quad_a = float(row['std_quad_a'])
                    std_quad_b = float(row['std_quad_b'])
                    std_quad_c = float(row['std_quad_c'])
                except (ValueError, KeyError):
                    pass

            # Parse variance regression (direct, for MSE-based R matrix)
            var_intercept = None
            var_slope = None
            if 'var_intercept' in row and row.get('var_intercept', '').strip():
                try:
                    var_intercept = float(row['var_intercept'])
                    var_slope = float(row['var_slope'])
                except (ValueError, KeyError):
                    pass

            var_quad_a = None
            var_quad_b = None
            var_quad_c = None
            if 'var_quad_a' in row and row.get('var_quad_a', '').strip():
                try:
                    var_quad_a = float(row['var_quad_a'])
                    var_quad_b = float(row['var_quad_b'])
                    var_quad_c = float(row['var_quad_c'])
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
                bias_intercept=bias_intercept,
                bias_slope=bias_slope,
                bias_quad_a=bias_quad_a,
                bias_quad_b=bias_quad_b,
                bias_quad_c=bias_quad_c,
                std_intercept=std_intercept,
                std_slope=std_slope,
                std_quad_a=std_quad_a,
                std_quad_b=std_quad_b,
                std_quad_c=std_quad_c,
                var_intercept=var_intercept,
                var_slope=var_slope,
                var_quad_a=var_quad_a,
                var_quad_b=var_quad_b,
                var_quad_c=var_quad_c,
            )
    
    name = Path(file_path).stem
    model_data = SensorModelData(name=name, file_path=file_path, params=params)
    
    # Try to load distribution bins from companion file
    dist_file = Path(file_path).parent / f"{name}_distributions.csv"
    if dist_file.exists():
        model_data.distribution_bins = load_distribution_bins_csv(str(dist_file))

    # Try to load P(TP) calibration from companion file (graceful fallback:
    # detectors without this file are treated as "no calibration" — Phase A
    # birth gate becomes a no-op for them).
    calib_file = Path(file_path).parent / f"{name}_polar_calibration.csv"
    if calib_file.exists():
        model_data.calibration = load_calibration_csv(str(calib_file))

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
            param1 = float(row['param1']) if row.get('param1', '').strip() else None
            param2 = float(row['param2']) if row.get('param2', '').strip() else None
            param3 = float(row['param3']) if row.get('param3', '').strip() else None

            # Build params dict based on distribution type
            if distribution == 'normal':
                params = {'mu': param1, 'sigma': param2}
                if param3 is not None:
                    params['mae'] = param3  # MAE for GPEM covariance (polar bins)
            elif distribution == 'laplace':
                params = {'mu': param1, 'b': param2}
            elif distribution == 'logistic':
                params = {'mu': param1, 's': param2}
            elif distribution == 'student_t':
                params = {'nu': param1, 'mu': param2, 'sigma': param3}
            elif distribution == 'cauchy':
                params = {'x0': param1, 'gamma': param2}
            elif distribution == 'constant':
                params = {'value': param1}
            else:
                raise ValueError(f"Unknown distribution type: {distribution}")

            # Parse optional angle columns (polar bins)
            angle_min = None
            angle_max = None
            if 'angle_min' in row and row['angle_min'].strip():
                angle_min = float(row['angle_min'])
                angle_max = float(row['angle_max'])

            bin_data = DistributionBinData(
                error_type=error_type,
                dist_min=float(row['dist_min']),
                dist_max=float(row['dist_max']),
                distribution=distribution,
                params=params,
                angle_min=angle_min,
                angle_max=angle_max,
            )

            if error_type not in bins:
                bins[error_type] = []
            bins[error_type].append(bin_data)

    # Sort bins by distance (and angle for polar bins)
    for error_type in bins:
        bins[error_type].sort(key=lambda b: (b.dist_min, b.angle_min or -999))

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
