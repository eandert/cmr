import math
import traci
import ground_truth
import utils
import sensor
import localizer
from config import sensor_type, detector_type
from metrics import metrics, minimum_safety_envelope, time_to_collision, amota
from metrics.hota import HotaAccumulator
import time
import sensor_fusion
import csv
from error import ErrorPackage, ErrorType
from config_templates import get_perfect_config, get_normal_sensors_config
from scoring import PerceptionScorer
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Set start method for multiprocessing (fork is faster on Linux, but spawn is safer)
# Using 'fork' to share memory and avoid pickling overhead
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set

# Global process pool (created once, reused across frames)
_process_pool = None

def get_process_pool(max_workers=None):
    """Get or create the global process pool."""
    global _process_pool
    if _process_pool is None:
        if max_workers is None:
            max_workers = os.cpu_count()
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _process_pool

def shutdown_process_pool():
    """Shutdown the global process pool."""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None


def _detection_with_covariance_from_model(det, participant_x, participant_y, participant_yaw,
                                          detector_error_model, localizer, velocity,
                                          variance_floor=None):
    """
    Return a copy of DetectedObject with error_covariance and dimension stds computed from
    the participant's detector and localizer models at the detection's position.

    - Uses detection position (with error) for distance/angle (what predictor sees)
    - Combines perception covariance + localization covariance (Eq 18: Σ_total = Σ_loc + Σ_perc)
    - detector_error_model and localizer should have matching GPEM mode (static/linear/quadratic)
    - variance_floor: if set, clamps diagonal elements of total_cov to be >= this value.
      Prevents over-confidence at close range where GPEM predictions can be very small.
    """
    from gaussians import BivariateGaussian
    import numpy as np

    # Distance/angle from participant to detection (detection position includes error)
    dx = det.centroid[0] - participant_x
    dy = det.centroid[1] - participant_y
    distance = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)

    # Perception covariance from detector model
    d_std = detector_error_model.get_distal_std(distance)
    p_std = detector_error_model.get_perpendicular_std(distance)
    perc_cov = BivariateGaussian(d_std**2, p_std**2, angle).covariance

    # Localization covariance from localizer (velocity-dependent, rotated by participant yaw)
    loc_cov = localizer.get_localization_covariance(velocity, participant_yaw)

    # Eq 18: total covariance = perception + localization
    total_cov = perc_cov + loc_cov

    # Apply variance floor to prevent over-confidence at close range
    if variance_floor is not None:
        total_cov[0, 0] = max(total_cov[0, 0], variance_floor)
        total_cov[1, 1] = max(total_cov[1, 1], variance_floor)

    w_std = detector_error_model.get_width_std(distance)
    l_std = detector_error_model.get_length_std(distance)

    return sensor.DetectedObject(
        vehicle_id=det.vehicle_id,
        vehicle_type=det.type,
        detected_bbox=det.detected_bbox,
        centroid=det.centroid,
        width=det.dimensions[0],
        length=det.dimensions[1],
        angle=det.angle,
        expected_error_gaussian=None,
        velocity_vector=det.velocity_vector,
        error_covariance=total_cov,
        width_std=w_std,
        length_std=l_std,
    )


# Baseline covariance: computed as the true average of all static detector + localizer
# covariances used in the experiment (weighted by allocation). This replaces the old
# arbitrary 0.1*I with a principled value derived from the actual error models.
_BASELINE_COV = None  # Lazy init


def _compute_true_average_covariance(detector_allocation, localizer_allocation, max_range, reference_velocity=15.0):
    """
    Compute the true average total covariance (perception + localization) as a scalar variance,
    averaged across all detector and localizer types weighted by their allocation fractions.

    Uses static (distance-averaged) detector models and static localizer covariance at a
    reference velocity of 15 m/s (typical highway speed).

    Returns np.eye(2) * avg_variance.
    """
    import numpy as np
    from error_models import get_error_model

    total_weighted_var = 0.0
    total_weight = 0.0

    for det_frac, det_name in detector_allocation:
        if det_frac <= 0:
            continue
        # Map allocation names to error model names (lowercase)
        em_name = det_name.lower()
        em = get_error_model(em_name, use_gpem_model=False, max_range=max_range)
        d_std = em.get_distal_std(30.0)  # Static mode returns distance-averaged value
        p_std = em.get_perpendicular_std(30.0)
        perc_var = (d_std**2 + p_std**2) / 2.0  # Average of radial and lateral variance

        for loc_frac, loc_name in localizer_allocation:
            if loc_frac <= 0:
                continue
            from error import ErrorPackage, ErrorType
            dummy_error = ErrorPackage(ErrorType.LOCALIZATION, 0)
            loc = localizer.get_localizer(loc_name, dummy_error, use_gpem_model=False)
            loc_cov = loc.get_localization_covariance(reference_velocity, 0.0)
            loc_var = (loc_cov[0, 0] + loc_cov[1, 1]) / 2.0

            weight = det_frac * loc_frac
            total_weighted_var += weight * (perc_var + loc_var)
            total_weight += weight

    avg_var = total_weighted_var / max(total_weight, 1e-9)
    return np.eye(2) * avg_var

def _detection_with_flat_covariance(det, flat_cov):
    """Return a copy of DetectedObject with a fixed flat covariance (baseline mode)."""
    return sensor.DetectedObject(
        vehicle_id=det.vehicle_id,
        vehicle_type=det.type,
        detected_bbox=det.detected_bbox,
        centroid=det.centroid,
        width=det.dimensions[0],
        length=det.dimensions[1],
        angle=det.angle,
        expected_error_gaussian=None,
        velocity_vector=det.velocity_vector,
        error_covariance=flat_cov,
        width_std=0.3,  # Flat default
        length_std=0.3,
    )


# Cache for per-detector error models in three GPEM modes
_detector_error_model_cache = {}


def _get_detector_error_models(detector_model_name, max_range):
    """
    Get (static, linear, quadratic) error models for a detector, using cache.
    """
    from error_models import get_error_model
    cache_key = (detector_model_name, max_range)
    if cache_key not in _detector_error_model_cache:
        em_static = get_error_model(detector_model_name, use_gpem_model=False, max_range=max_range)
        em_linear = get_error_model(detector_model_name, use_gpem_model=True, use_quadratic=False, max_range=max_range)
        em_quadratic = get_error_model(detector_model_name, use_gpem_model=True, use_quadratic=True, max_range=max_range)
        _detector_error_model_cache[cache_key] = (em_static, em_linear, em_quadratic)
    return _detector_error_model_cache[cache_key]


# Cache for per-localizer instances in three GPEM modes
_localizer_cache = {}


def _get_localizer_variants(base_localizer):
    """
    Get (static, static, static) localizer variants from a base localizer.
    
    For triple-fusion testing, we use the SAME localizer covariance (static) for all three
    detector GPEM modes. This isolates the detector GPEM effect from the localizer GPEM effect.
    
    The detector models (static, linear, quadratic) will still vary, but localization
    uncertainty is held constant so we can measure detector GPEM impact alone.
    """
    # Localizers are lightweight; create variants with different GPEM modes
    # We need the localizer_type to create new instances
    loc_type = getattr(base_localizer, 'localizer_type', None)
    if loc_type is None:
        # Fallback: try to infer from class or just create copies with different settings
        loc_type = "kiss_icp"

    cache_key = (loc_type,)  # Cache by localizer type, not instance
    if cache_key not in _localizer_cache:
        from error import ErrorPackage, ErrorType
        # Create a dummy error package (not used for covariance, only for error injection)
        dummy_error = ErrorPackage(ErrorType.LOCALIZATION, 0)
        # Use STATIC localizer for ALL three variants to isolate detector GPEM effect
        loc_static = localizer.get_localizer(loc_type, dummy_error, use_gpem_model=False, use_quadratic=False)
        # All three return the same static localizer
        _localizer_cache[cache_key] = (loc_static, loc_static, loc_static)
    return _localizer_cache[cache_key]


# Function to run the simulation
def run_simulation(config):
    """
    Run a SUMO simulation with the given configuration.
    
    Args:
        config: Dictionary containing simulation configuration including:
            - sumo_config_path: Path to SUMO config file
            - step_length: Simulation step size
            - sensor_types, detector_types: Sensor configuration
            - warmup_steps: Steps before recording metrics (default: 0)
            - record_steps: Number of steps to record (default: None = all)
            - max_steps: Maximum steps to run (default: None = run to completion)
            - error_injection_start: Step to start error injection (default: 0)
            - error_injection_end: Step to stop error injection (default: None = never)
            - And other standard config options...
    
    Returns:
        Dictionary with simulation results and metrics
    """
    # Configuration for parallelization
    use_parallel = config.get("use_parallel", True)  # Enable by default
    min_packages_for_parallel = 2  # Minimum packages to justify parallel overhead
    
    # Data collection timing
    fast_forward_steps = config.get("fast_forward_steps", 0)
    warmup_steps = config.get("warmup_steps", 0)
    record_steps = config.get("record_steps", None)  # None = record all after FF + warmup
    max_steps = config.get("max_steps", None)  # None = run to completion
    
    # Error injection timing
    error_injection_start = config.get("error_injection_start", 0)
    error_injection_end = config.get("error_injection_end", None)
    
    # Output control
    quiet_mode = config.get("quiet_mode", False)
    progress_callback = config.get("_progress_callback", None)
    
    # Calculate total expected steps for progress
    if record_steps is not None and warmup_steps is not None:
        total_expected_steps = fast_forward_steps + warmup_steps + record_steps
    elif max_steps is not None:
        total_expected_steps = max_steps
    else:
        total_expected_steps = 0  # Unknown
    
    # Instantiate ErrorPackage
    if config["error_type"] == "None":
        # Create a dummy ErrorPackage that injects no errors
        error_package_instance = ErrorPackage(ErrorType.SINGLE_SENSOR_EXTRINSICS, 0)
        error_injection_active = False  # No errors to inject
    else:
        # Use bracket notation to look up enum by name (e.g., "SINGLE_SENSOR_EXTRINSICS")
        error_package_instance = ErrorPackage(ErrorType[config["error_type"]], config["error_level"])
        error_injection_active = True
        # Start with errors inactive if error_injection_start is set
        if error_injection_start > 0:
            error_package_instance.is_active = False

    # Start SUMO in server mode with unique label for parallel execution
    sumoBinary = "sumo"  # or "sumo-gui" if you need the GUI
    sumoCmd = [
        sumoBinary,
        "-c", os.path.join(os.path.dirname(__file__), "..", config["sumo_config_path"]), # Updated path
        "--step-length", str(config["step_length"])  # Set the step size from config
    ]
    
    # Add quiet mode flags if requested (for parallel execution)
    if config.get("quiet_mode", False):
        sumoCmd.extend([
            "--no-warnings",
            "--no-step-log",
            "--log", "/dev/null",
            "--message-log", "/dev/null",
            "-v", "false"
        ])
    
    # Generate unique simulation label for parallel execution
    # Uses process ID and timestamp to ensure uniqueness
    simulation_label = config.get("_simulation_label", f"sim_{os.getpid()}_{int(time.time() * 1000) % 100000}")
    traci.start(sumoCmd, label=simulation_label)
    
    # Get the connection object for this specific simulation
    traci_conn = traci.getConnection(simulation_label)

    # Get GPEM model flags from config
    use_gpem_model = config.get("use_gpem_model", False)
    use_quadratic = config.get("use_quadratic", False)
    detector_max_range = config.get("detector_max_range", 70.0)

    # Optional: distribution of detector and localizer across CAVs (for GPEM distribution sweep)
    detector_allocation = config.get("detector_allocation")  # list of (weight, detector_name)
    localizer_allocation = config.get("localizer_allocation")  # list of (weight, localizer_name)

    if detector_allocation is not None and localizer_allocation is not None:
        # Build multiple sensor packages: one per (detector, localizer) with probability = det_weight * loc_weight
        total_det = sum(w for w, _ in detector_allocation)
        total_loc = sum(w for w, _ in localizer_allocation)
        det_weights = [(w / total_det if total_det else 1.0 / len(detector_allocation), n) for w, n in detector_allocation]
        loc_weights = [(w / total_loc if total_loc else 1.0 / len(localizer_allocation), n) for w, n in localizer_allocation]
        # Default sensor type for each package (single sensor per package)
        base_sensor_type = config.get("distribution_base_sensor_type", "OS1_128")
        base_extrinsics = config.get("sensor_extrinsics", [(0.0, 0.0, 0.0)])
        sensor_packages_for_managers = []
        for det_prob, det_name in det_weights:
            for loc_prob, loc_name in loc_weights:
                prob = det_prob * loc_prob
                _detector_type = getattr(detector_type.DetectorType, det_name)
                one_sensor = sensor.Sensor(
                    sensor_type=getattr(sensor_type.SensorType, base_sensor_type),
                    detector_type=_detector_type,
                    use_gpem_model=use_gpem_model,
                    use_quadratic=use_quadratic,
                    detector_max_range=detector_max_range
                )
                # Load localizer from ground-truth CSV data
                loc_instance = localizer.get_localizer(
                    localizer_type=loc_name,
                    error_package=error_package_instance,
                    use_gpem_model=use_gpem_model,
                    use_quadratic=use_quadratic
                )
                sensor_packages_for_managers.append((prob, [one_sensor], base_extrinsics, loc_instance))
    else:
        # Define sensor detector combos based on config (original single-package behavior)
        sensors = []
        for i in range(len(config["sensor_types"])):
            _sensor_type = getattr(sensor_type.SensorType, config["sensor_types"][i])
            _detector_type = getattr(detector_type.DetectorType, config["detector_types"][i])
            sensors.append(sensor.Sensor(
                sensor_type=_sensor_type,
                detector_type=_detector_type,
                use_gpem_model=use_gpem_model,
                use_quadratic=use_quadratic,
                detector_max_range=detector_max_range
            ))
        # Load localizer from ground-truth CSV data (default to kiss_icp for lidar-based)
        default_localizer_type = config.get("localizer_type", "kiss_icp")
        lidar_slam = localizer.get_localizer(
            localizer_type=default_localizer_type,
            error_package=error_package_instance,
            use_gpem_model=use_gpem_model,
            use_quadratic=use_quadratic
        )
        sensor_packages_for_managers = [(1, sensors, config["sensor_extrinsics"], lidar_slam)]

    # Initialize VehicleProbabilityManager for CAVs
    cav_manager = utils.VehicleProbabilityManager(probability=config["cav_probability"], type="CAV", sumo_type="CAV_passenger", sensor_packages=sensor_packages_for_managers, error_package=error_package_instance)

    # Initialize TrafficLightProbabilityManager for traffic lights
    tfl_manager = utils.TrafficLightProbabilityManager(probability=config["tfl_probability"], traci_instance=traci_conn, sensor_packages=sensor_packages_for_managers, error_package=error_package_instance)
    cis_id_list = tfl_manager.get_tracked_traffic_light_ids()
    
    # Pre-cache all traffic light positions (they never change) - avoid TraCI calls in loop
    for cis_id in cis_id_list:
        ground_truth.create_ground_truth_for_traffic_light_by_id(traci_conn, cis_id)

    # This is for doing the global sensor fusion
    use_trust_scoring = config.get("use_trust_scoring", True)
    gpem_triple_fusion = config.get("gpem_triple_fusion", False)
    _det_max_range = config.get("detector_max_range", 70.0)
    use_static_matching = config.get("use_static_matching", False)
    _variance_floor = config.get("variance_floor", None)  # Min diagonal variance for R matrix
    if gpem_triple_fusion:
        # One SUMO run, three fusion instances (static, GPEM linear, GPEM quadratic) for synced A/B/C comparison
        # Error models are loaded per-participant (by detector type) in the loop
        # Enable passthrough_covariance to bypass Kalman filtering of covariance
        # This ensures GPEM predictions flow through directly to global fusion
        #
        # If use_static_matching=True, all three streams use the same static covariance
        # for matching (Mahalanobis gating) but their own covariance for Kalman fusion.
        # This isolates whether matching or fusion drives the GPEM difference.
        import numpy as np
        static_match_cov = None
        if use_static_matching:
            # Compute average static matching covariance from the static error model at ~30m
            from error_models import get_error_model
            _em_static_ref = get_error_model("bev_fusion", use_gpem_model=False, max_range=_det_max_range)
            d_std = _em_static_ref.get_distal_std(30.0)
            p_std = _em_static_ref.get_perpendicular_std(30.0)
            static_match_cov = np.diag([max(d_std, p_std)**2, max(d_std, p_std)**2])
        global_fusion_static = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov)
        global_fusion_linear = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov)
        global_fusion_quadratic = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov)
        global_fusion_baseline = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov)
        # CI filter streams: same covariance inputs as Kalman streams but use CovarianceIntersectionFilter
        from filters.covariance_intersection import CovarianceIntersectionFilter
        global_fusion_ci_static = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=CovarianceIntersectionFilter)
        global_fusion_ci_linear = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=CovarianceIntersectionFilter)
        global_fusion_ci_quadratic = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=CovarianceIntersectionFilter)
        global_fusion_ci_baseline = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=CovarianceIntersectionFilter)
        # Adaptive Kalman filter streams: same covariance inputs but with online Q adaptation
        from filters.adaptive_kalman import AdaptiveKalman
        global_fusion_akf_static = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=AdaptiveKalman)
        global_fusion_akf_linear = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=AdaptiveKalman)
        global_fusion_akf_quadratic = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=AdaptiveKalman)
        global_fusion_akf_baseline = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=AdaptiveKalman)
        # Particle filter streams: SIR particle filter
        from filters.particle_filter import ParticleFilter
        global_fusion_pf_static = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=ParticleFilter)
        global_fusion_pf_linear = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=ParticleFilter)
        global_fusion_pf_quadratic = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=ParticleFilter)
        global_fusion_pf_baseline = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False, passthrough_covariance=True, static_matching_cov=static_match_cov, filter_class=ParticleFilter)
        # Compute baseline covariance as the true weighted average of all static detector +
        # localizer covariances used in this experiment (principled, not arbitrary)
        det_alloc = config.get("detector_allocation", [(1.0, "BEV_FUSION")])
        loc_alloc = config.get("localizer_allocation", [(1.0, "CT_ICP")])
        baseline_cov = _compute_true_average_covariance(det_alloc, loc_alloc, _det_max_range)
        _baseline_var = baseline_cov[0, 0]
        print(f"    Baseline covariance: {_baseline_var:.4f} (std={np.sqrt(_baseline_var):.4f}m) — computed from detector+localizer models")
        if _variance_floor is not None:
            print(f"    Variance floor: {_variance_floor:.4f} (std={np.sqrt(_variance_floor):.4f}m)")
        global_fusion = None  # not used when triple_fusion
    else:
        global_fusion = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=use_trust_scoring)
        global_fusion_static = global_fusion_linear = global_fusion_quadratic = global_fusion_baseline = None
        global_fusion_ci_static = global_fusion_ci_linear = global_fusion_ci_quadratic = global_fusion_ci_baseline = None
        global_fusion_akf_static = global_fusion_akf_linear = global_fusion_akf_quadratic = global_fusion_akf_baseline = None
        global_fusion_pf_static = global_fusion_pf_linear = global_fusion_pf_quadratic = global_fusion_pf_baseline = None
        baseline_cov = None

    # Track the IDs of all polygons
    polygon_ids = []

    # Track total 
    total_mse_violations_gt = 0
    total_ttc_violations_gt = 0
    total_mse_violations_sensor = 0
    total_ttc_violations_sensor = 0

    # Timing variables
    total_simulation_step_time = 0
    total_cis_detection_time = 0 
    total_cav_detection_time = 0
    total_global_fusion_time = 0
    total_random_overhead_time = 0
    iterations = 0

    # Metrics accumulation - base metrics plus AV/non-AV split
    def create_metrics_dict():
        return {
            "total_mse_violations_gt": 0,
            "total_mse_violations_sensor": 0,
            "total_ttc_violations_gt": 0,
            "total_ttc_violations_sensor": 0,
            "true_positives_mse": 0,
            "false_positives_mse": 0,
            "false_negatives_mse": 0,
            "true_positives_ttc": 0,
            "false_positives_ttc": 0,
            "false_negatives_ttc": 0,
            "amota": 0.0,
            "amotp": 0.0,
            # AV/non-AV split metrics
            "total_mse_violations_gt_av": 0,
            "total_mse_violations_gt_non_av": 0,
            "total_mse_violations_sensor_av": 0,
            "total_mse_violations_sensor_non_av": 0,
            "total_ttc_violations_gt_av": 0,
            "total_ttc_violations_gt_non_av": 0,
            "total_ttc_violations_sensor_av": 0,
            "total_ttc_violations_sensor_non_av": 0,
        }
    
    total_cav_metrics = create_metrics_dict()
    total_cis_metrics = create_metrics_dict()  # CIS metrics
    total_global_metrics = create_metrics_dict()
    if gpem_triple_fusion:
        total_global_metrics_static = create_metrics_dict()
        total_global_metrics_linear = create_metrics_dict()
        total_global_metrics_quadratic = create_metrics_dict()
        total_global_metrics_baseline = create_metrics_dict()
        total_global_metrics_ci_static = create_metrics_dict()
        total_global_metrics_ci_linear = create_metrics_dict()
        total_global_metrics_ci_quadratic = create_metrics_dict()
        total_global_metrics_ci_baseline = create_metrics_dict()
        total_global_metrics_akf_static = create_metrics_dict()
        total_global_metrics_akf_linear = create_metrics_dict()
        total_global_metrics_akf_quadratic = create_metrics_dict()
        total_global_metrics_akf_baseline = create_metrics_dict()
        total_global_metrics_pf_static = create_metrics_dict()
        total_global_metrics_pf_linear = create_metrics_dict()
        total_global_metrics_pf_quadratic = create_metrics_dict()
        total_global_metrics_pf_baseline = create_metrics_dict()
        triple_global_amota_frames_count_static = 0
        triple_global_amota_frames_count_linear = 0
        triple_global_amota_frames_count_quadratic = 0
        triple_global_amota_frames_count_baseline = 0
        triple_global_amota_frames_count_ci_static = 0
        triple_global_amota_frames_count_ci_linear = 0
        triple_global_amota_frames_count_ci_quadratic = 0
        triple_global_amota_frames_count_ci_baseline = 0
        triple_global_amota_frames_count_akf_static = 0
        triple_global_amota_frames_count_akf_linear = 0
        triple_global_amota_frames_count_akf_quadratic = 0
        triple_global_amota_frames_count_akf_baseline = 0
        triple_global_amota_frames_count_pf_static = 0
        triple_global_amota_frames_count_pf_linear = 0
        triple_global_amota_frames_count_pf_quadratic = 0
        triple_global_amota_frames_count_pf_baseline = 0
        triple_global_amotp_frames_count_static = 0
        triple_global_amotp_frames_count_linear = 0
        triple_global_amotp_frames_count_quadratic = 0
        triple_global_amotp_frames_count_baseline = 0
        triple_global_amotp_frames_count_ci_static = 0
        triple_global_amotp_frames_count_ci_linear = 0
        triple_global_amotp_frames_count_ci_quadratic = 0
        triple_global_amotp_frames_count_ci_baseline = 0
        triple_global_amotp_frames_count_akf_static = 0
        triple_global_amotp_frames_count_akf_linear = 0
        triple_global_amotp_frames_count_akf_quadratic = 0
        triple_global_amotp_frames_count_akf_baseline = 0
        triple_global_amotp_frames_count_pf_static = 0
        triple_global_amotp_frames_count_pf_linear = 0
        triple_global_amotp_frames_count_pf_quadratic = 0
        triple_global_amotp_frames_count_pf_baseline = 0
    total_active_cavs = 0
    cav_amota_frames_count = 0 # New counter for frames where CAV AMOTA is calculated
    cis_amota_frames_count = 0 # New counter for frames where CIS AMOTA is calculated
    global_amota_frames_count = 0 # New counter for frames where Global AMOTA is calculated
    cav_amotp_frames_count = 0 # Counter for frames where CAV AMOTP is calculated
    cis_amotp_frames_count = 0 # Counter for frames where CIS AMOTP is calculated
    global_amotp_frames_count = 0 # Counter for frames where Global AMOTP is calculated
    # HOTA accumulators (require cross-frame tracking data)
    global_hota_acc = HotaAccumulator()
    if gpem_triple_fusion:
        hota_acc_baseline = HotaAccumulator()
        hota_acc_static = HotaAccumulator()
        hota_acc_linear = HotaAccumulator()
        hota_acc_quadratic = HotaAccumulator()
        hota_acc_ci_baseline = HotaAccumulator()
        hota_acc_ci_static = HotaAccumulator()
        hota_acc_ci_linear = HotaAccumulator()
        hota_acc_ci_quadratic = HotaAccumulator()
        hota_acc_akf_baseline = HotaAccumulator()
        hota_acc_akf_static = HotaAccumulator()
        hota_acc_akf_linear = HotaAccumulator()
        hota_acc_akf_quadratic = HotaAccumulator()
        hota_acc_pf_baseline = HotaAccumulator()
        hota_acc_pf_static = HotaAccumulator()
        hota_acc_pf_linear = HotaAccumulator()
        hota_acc_pf_quadratic = HotaAccumulator()
    recorded_steps = 0  # Counter for steps where metrics are recorded (after warmup)
    
    # Calculate recording end step
    if record_steps is not None:
        record_end_step = fast_forward_steps + warmup_steps + record_steps
    else:
        record_end_step = None  # Record until end
    
    # Initialize perception scorer for Conclave-style monitoring
    # Buffer size of 200 frames, skip first 25 frames for warmup
    perception_scorer = PerceptionScorer(
        buffer_size=200,
        min_detections=3,
        missed_detection_penalty=3.0,
        anomaly_threshold=1.2,  # 20% deviation triggers anomaly
        warmup_frames=25
    )
    perception_baseline_captured = False

    # Debug variables
    visualize_ground_truth = config["visualize_ground_truth"]
    visualize_cis_detections = config["visualize_cis_detections"]
    visualize_cav_detections = config["visualize_cav_detections"]
    visualize_global_fusion = config["visualize_global_fusion"]
    do_global_fusion = config["do_global_fusion"]
    
    # Quick check if ANY visualization is enabled (optimization: skip all drawing code if not)
    any_visualization = visualize_ground_truth or visualize_cis_detections or visualize_cav_detections or visualize_global_fusion

    # Get process pool if using parallel processing
    pool = get_process_pool() if use_parallel else None

    # Simulation loop
    step = 0
    while traci_conn.simulation.getMinExpectedNumber() > 0:
        # Check if we should stop based on max_steps
        if max_steps is not None and step >= max_steps:
            break
        
        # Check if we should stop recording based on record_end_step
        if record_end_step is not None and step >= record_end_step:
            break
        
        # Stages logic
        is_fast_forwarding = step < fast_forward_steps
        is_warming_up = fast_forward_steps <= step < fast_forward_steps + warmup_steps
        is_recording = step >= fast_forward_steps + warmup_steps
        
        # Determine if error injection should be active this step and update the error package
        if error_injection_active:
            error_active_this_step = (
                step >= error_injection_start and 
                (error_injection_end is None or step < error_injection_end)
            )
            # Dynamically enable/disable error injection based on timing
            error_package_instance.is_active = error_active_this_step
        
        start_time = time.time()
        traci_conn.simulationStep()
        simulation_step_time = time.time() - start_time

        if not quiet_mode and step % 100 == 0:
            stage_str = "FF" if is_fast_forwarding else ("warmup" if is_warming_up else "recording")
            print(f"Step: {step} ({stage_str})")

        random_overhead_start_time = time.time()
        # Remove all polygons from the last frame (only if visualization is enabled)
        if any_visualization and polygon_ids:
            for polygon_id in polygon_ids:
                traci_conn.polygon.remove(polygon_id)
            polygon_ids = []

        # Get the IDs of all vehicles in the simulation
        vehicle_ids = traci_conn.vehicle.getIDList()

        # Update CAVs if step >= 1 - WE ALWAYS DO THIS to track who are AVs
        if step >= 1:
            cav_manager.update_vehicles(traci_conn, vehicle_ids)
        
        # IF FAST FORWARDING, SKIP EVERYTHING ELSE
        if is_fast_forwarding:
            iterations += 1
            step += 1
            # Call progress callback if provided
            if progress_callback is not None and step % 10 == 0:
                progress_callback(step, total_expected_steps, {
                    'global_amota': 0, 'cis_amota': 0, 'cav_amota': 0,
                    'cav_count': len(cav_manager.get_active_vehicle_ids()),
                    'recorded_steps': 0, 'step_time': simulation_step_time,
                })
            continue

        # Create ground truth data for all vehicles
        ground_truth_global = ground_truth.create_ground_truth_for_vehicle_from_list(traci_conn, vehicle_ids)

        # Create a cache dictionary for ground truth objects to avoid recreation
        # Key: vehicle_id, Value: GroundTruthObject
        ground_truth_cache = {gt.vehicle_id: gt for gt in ground_truth_global}

        # Build spatial index for fast range queries (optimization #2)
        ground_truth_spatial_index, _ = ground_truth.build_ground_truth_spatial_index(ground_truth_global)

        # Draw the bounding boxes in SUMO for debugging
        if visualize_ground_truth:
            for gt_obj in ground_truth_global:
                polygon_bbox_id = gt_obj.draw_bounding_box_in_sumo(traci_conn, layer=10, polygon_id_prefix="gt")
                if polygon_bbox_id:
                    polygon_ids.append(polygon_bbox_id)
                
                # Position vector also needs a unique ID
                polygon_vector_id = gt_obj.draw_position_vector_in_sumo(traci_conn, layer=11, polygon_id_prefix="gt")
                if polygon_vector_id:
                    polygon_ids.append(polygon_vector_id)

        # Get active CAVs and their ground truth objects
        cav_id_list = cav_manager.get_active_vehicle_ids()
        total_active_cavs += len(cav_id_list)
        
        # Create set of AV IDs for metrics split
        av_ids = set(cav_id_list)
        
        # Simulation time now
        simulation_time_now = traci_conn.simulation.getTime()

        # Initialize a dictionary to store unique detectable ground truth objects
        unique_detectable_ground_truth = {}
        random_overhead_time = time.time() - random_overhead_start_time

        # ==================== CIS DETECTION ====================
        cis_gt_cache = {}
        cis_detection_start_time = time.time()
        
        # Prepare CIS data
        cis_instances = list(tfl_manager.get_tracked_traffic_light_instances())
        cis_results = []
        
        # Check if we should use parallel processing for CIS
        use_parallel_cis = use_parallel and pool is not None and len(cis_id_list) >= min_packages_for_parallel
        
        if use_parallel_cis:
            # Submit all CIS tasks to the process pool
            futures = []
            for cis_id, cis_instance in zip(cis_id_list, cis_instances):
                cis_gt_obj = ground_truth.create_ground_truth_for_traffic_light_by_id(traci_conn, cis_id)
                cis_gt_cache[cis_id] = cis_gt_obj
                future = pool.submit(
                    process_sensor_package,
                    cis_id, cis_instance, cis_gt_obj,
                    ground_truth_global, ground_truth_spatial_index,
                    simulation_time_now, config["cleanup_time"]
                )
                futures.append((cis_id, cis_instance, future))
            
            # Collect results and update fusion state
            for cis_id, cis_instance, future in futures:
                fusion_result, detected_objects_for_metrics, detectable_ground_truth, pkg_id, updated_fusion_state = future.result()
                # Update the sensor package's fusion state from the worker
                update_fusion_state(cis_instance.sensor_fusion, updated_fusion_state)
                cis_results.append((cis_id, fusion_result, detected_objects_for_metrics, detectable_ground_truth))
        else:
            # Sequential processing for CIS
            for cis_id, cis_instance in zip(cis_id_list, cis_instances):
                cis_gt_obj = ground_truth.create_ground_truth_for_traffic_light_by_id(traci_conn, cis_id)
                cis_gt_cache[cis_id] = cis_gt_obj
                
                fusion_result, detected_objects_for_metrics, detectable_ground_truth = \
                    cis_instance.create_detection_sets(cis_gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now)
                cis_results.append((cis_id, fusion_result, detected_objects_for_metrics, detectable_ground_truth))
        
        # Process CIS results
        # CIS participants get IDs starting from 0
        for cis_idx, (cis_id, fusion_result, detected_objects_for_metrics, detectable_ground_truth) in enumerate(cis_results):
            unique_detectable_ground_truth.update(detectable_ground_truth)

            if visualize_cis_detections:
                for each in detected_objects_for_metrics:
                    polygon_bbox_id = each.draw_bounding_box_in_sumo(traci_conn, color=(255, 255, 0, 100), layer=9, sensor_package_id=cis_id)
                    if polygon_bbox_id:
                        polygon_ids.append(polygon_bbox_id)

            # Only record metrics after warmup
            if is_recording:
                cis_gt_obj_for_metrics = cis_gt_cache.get(cis_id)
                if cis_gt_obj_for_metrics is None:
                    cis_gt_obj_for_metrics = ground_truth.create_ground_truth_for_traffic_light_by_id(traci_conn, cis_id)
                cis_metrics = metrics.calculate_violations(cis_gt_obj_for_metrics, detectable_ground_truth, detected_objects_for_metrics, category="Aggressive", av_ids=av_ids)

                for key, value in cis_metrics.items():
                    if key != "amota":
                        total_cis_metrics[key] += value
                
                if cis_metrics["amota"] != 0.0 and len(detectable_ground_truth) > 0:
                    total_cis_metrics["amota"] += cis_metrics["amota"]
                    cis_amota_frames_count += 1

            if do_global_fusion:
                if gpem_triple_fusion:
                    # Get CIS instance from results (need to find it in cis_instances)
                    cis_instance = cis_instances[cis_idx]
                    cis_gt_obj = cis_gt_cache.get(cis_id)
                    px, py = cis_gt_obj.centroid[0], cis_gt_obj.centroid[1]
                    p_yaw = cis_gt_obj.angle if hasattr(cis_gt_obj, 'angle') else 0.0
                    # CIS is stationary, velocity = 0
                    velocity = 0.0
                    # Get detector model name from first sensor
                    detector_model_name = cis_instance.sensors[0].detector_type.error_model_name or "bev_fusion"
                    em_static, em_linear, em_quadratic = _get_detector_error_models(detector_model_name, _det_max_range)
                    # Get localizer variants (CIS uses localizer for covariance even if stationary)
                    loc_static, loc_linear, loc_quadratic = _get_localizer_variants(cis_instance.localizer)
                    list_static = [_detection_with_covariance_from_model(d, px, py, p_yaw, em_static, loc_static, velocity, variance_floor=_variance_floor) for d in detected_objects_for_metrics]
                    list_linear = [_detection_with_covariance_from_model(d, px, py, p_yaw, em_linear, loc_linear, velocity, variance_floor=_variance_floor) for d in detected_objects_for_metrics]
                    list_quad = [_detection_with_covariance_from_model(d, px, py, p_yaw, em_quadratic, loc_quadratic, velocity, variance_floor=_variance_floor) for d in detected_objects_for_metrics]
                    list_baseline = [_detection_with_flat_covariance(d, baseline_cov) for d in detected_objects_for_metrics]
                    global_fusion_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=cis_idx)
                    # CI filter streams: same covariance data, different filter algorithm
                    global_fusion_ci_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_ci_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_ci_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_ci_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=cis_idx)
                    # Adaptive Kalman filter streams: same covariance data, adaptive Q
                    global_fusion_akf_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_akf_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_akf_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_akf_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=cis_idx)
                    # Particle filter streams: SIR particle filter
                    global_fusion_pf_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_pf_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_pf_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=cis_idx)
                    global_fusion_pf_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=cis_idx)
                else:
                    # Pass CIS participant ID (0, 1, 2, ...) for trust scoring
                    global_fusion.processDetectionFrame(simulation_time_now, detected_objects_for_metrics, config["cleanup_time"], source_participant_id=cis_idx)

        cis_detection_time = time.time() - cis_detection_start_time

        # ==================== CAV DETECTION ====================
        cav_detection_start_time = time.time()
        
        cav_instances = list(cav_manager.get_active_vehicle_instances())
        cav_results = []
        
        # Check if we should use parallel processing for CAVs
        use_parallel_cav = use_parallel and pool is not None and len(cav_id_list) >= min_packages_for_parallel
        
        if use_parallel_cav:
            # Submit all CAV tasks to the process pool
            futures = []
            for cav_id, cav_instance in zip(cav_id_list, cav_instances):
                cav_gt_obj = ground_truth_cache.get(cav_id)
                if cav_gt_obj is None:
                    cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci_conn, cav_id)
                
                future = pool.submit(
                    process_sensor_package,
                    cav_id, cav_instance, cav_gt_obj,
                    ground_truth_global, ground_truth_spatial_index,
                    simulation_time_now, config["cleanup_time"]
                )
                futures.append((cav_id, cav_instance, future))
            
            # Collect results and update fusion state
            for cav_id, cav_instance, future in futures:
                fusion_result, detected_objects_for_metrics, detectable_ground_truth, pkg_id, updated_fusion_state = future.result()
                # Update the sensor package's fusion state from the worker
                update_fusion_state(cav_instance.sensor_fusion, updated_fusion_state)
                cav_results.append((cav_id, fusion_result, detected_objects_for_metrics, detectable_ground_truth))
        else:
            # Sequential processing for CAVs
            for cav_id, cav_instance in zip(cav_id_list, cav_instances):
                cav_gt_obj = ground_truth_cache.get(cav_id)
                if cav_gt_obj is None:
                    cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci_conn, cav_id)
                
                fusion_result, detected_objects_for_metrics, detectable_ground_truth = \
                    cav_instance.create_detection_sets(cav_gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now)
                cav_results.append((cav_id, fusion_result, detected_objects_for_metrics, detectable_ground_truth))
        
        # Process CAV results
        # CAV participants get IDs starting after CIS (len(cis_results) + cav_idx)
        num_cis = len(cis_results)
        for cav_idx, (cav_id, fusion_result, detected_objects_for_metrics, detectable_ground_truth) in enumerate(cav_results):
            unique_detectable_ground_truth.update(detectable_ground_truth)

            if visualize_cav_detections:
                for each in detected_objects_for_metrics:
                    polygon_bbox_id = each.draw_bounding_box_in_sumo(traci_conn, color=(255, 255, 0, 255), layer=9, sensor_package_id=cav_id)
                    if polygon_bbox_id:
                        polygon_ids.append(polygon_bbox_id)

            # Only record metrics after warmup
            if is_recording:
                cav_gt_obj_for_metrics = ground_truth_cache.get(cav_id)
                if cav_gt_obj_for_metrics is None:
                    cav_gt_obj_for_metrics = ground_truth.create_ground_truth_for_vehicle_by_id(traci_conn, cav_id)
                cav_metrics = metrics.calculate_violations(cav_gt_obj_for_metrics, detectable_ground_truth, detected_objects_for_metrics, category="Aggressive", av_ids=av_ids)

                for key, value in cav_metrics.items():
                    if key != "amota":
                        total_cav_metrics[key] += value
                
                if cav_metrics["amota"] != 0.0 and len(detectable_ground_truth) > 0:
                    total_cav_metrics["amota"] += cav_metrics["amota"]
                    cav_amota_frames_count += 1

            if do_global_fusion:
                if gpem_triple_fusion:
                    cav_instance = cav_instances[cav_idx]
                    cav_gt_obj = ground_truth_cache.get(cav_id)
                    if cav_gt_obj is not None:
                        px, py = cav_gt_obj.centroid[0], cav_gt_obj.centroid[1]
                        p_yaw = cav_gt_obj.angle
                        # Get velocity from ground truth
                        velocity = math.hypot(cav_gt_obj.velocity_vector[0], cav_gt_obj.velocity_vector[1])
                        # Get detector model name from first sensor
                        detector_model_name = cav_instance.sensors[0].detector_type.error_model_name or "bev_fusion"
                        em_static, em_linear, em_quadratic = _get_detector_error_models(detector_model_name, _det_max_range)
                        # Get localizer variants
                        loc_static, loc_linear, loc_quadratic = _get_localizer_variants(cav_instance.localizer)
                        list_static = [_detection_with_covariance_from_model(d, px, py, p_yaw, em_static, loc_static, velocity, variance_floor=_variance_floor) for d in detected_objects_for_metrics]
                        list_linear = [_detection_with_covariance_from_model(d, px, py, p_yaw, em_linear, loc_linear, velocity, variance_floor=_variance_floor) for d in detected_objects_for_metrics]
                        list_quad = [_detection_with_covariance_from_model(d, px, py, p_yaw, em_quadratic, loc_quadratic, velocity, variance_floor=_variance_floor) for d in detected_objects_for_metrics]
                        list_baseline = [_detection_with_flat_covariance(d, baseline_cov) for d in detected_objects_for_metrics]
                        global_fusion_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        # CI filter streams: same covariance data, different filter algorithm
                        global_fusion_ci_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_ci_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_ci_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_ci_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        # Adaptive Kalman filter streams: same covariance data, adaptive Q
                        global_fusion_akf_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_akf_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_akf_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_akf_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        # Particle filter streams: SIR particle filter
                        global_fusion_pf_static.processDetectionFrame(simulation_time_now, list_static, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_pf_linear.processDetectionFrame(simulation_time_now, list_linear, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_pf_quadratic.processDetectionFrame(simulation_time_now, list_quad, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                        global_fusion_pf_baseline.processDetectionFrame(simulation_time_now, list_baseline, config["cleanup_time"], source_participant_id=num_cis + cav_idx)
                else:
                    # Pass CAV participant ID (num_cis + 0, num_cis + 1, ...) for trust scoring
                    global_fusion.processDetectionFrame(simulation_time_now, detected_objects_for_metrics, config["cleanup_time"], source_participant_id=num_cis + cav_idx)

        cav_detection_time = time.time() - cav_detection_start_time

        # ==================== GLOBAL FUSION ====================
        global_fusion_start_time = time.time()
        if do_global_fusion:
            if gpem_triple_fusion:
                # Fuse all four (no perception monitor); record metrics for each
                _, global_detection_result_s, _, _ = global_fusion_static.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_l, _, _ = global_fusion_linear.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_q, _, _ = global_fusion_quadratic.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_b, _, _ = global_fusion_baseline.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_ci_s, _, _ = global_fusion_ci_static.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_ci_l, _, _ = global_fusion_ci_linear.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_ci_q, _, _ = global_fusion_ci_quadratic.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_ci_b, _, _ = global_fusion_ci_baseline.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_akf_s, _, _ = global_fusion_akf_static.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_akf_l, _, _ = global_fusion_akf_linear.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_akf_q, _, _ = global_fusion_akf_quadratic.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_akf_b, _, _ = global_fusion_akf_baseline.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_pf_s, _, _ = global_fusion_pf_static.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_pf_l, _, _ = global_fusion_pf_linear.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_pf_q, _, _ = global_fusion_pf_quadratic.fuseDetectionFrame(simulation_time_now, monitor=False)
                _, global_detection_result_pf_b, _, _ = global_fusion_pf_baseline.fuseDetectionFrame(simulation_time_now, monitor=False)
                if is_recording:
                    all_detectable_ground_truth = list(unique_detectable_ground_truth.values())
                    triple_amota_counts = [
                        triple_global_amota_frames_count_baseline, triple_global_amota_frames_count_static,
                        triple_global_amota_frames_count_linear, triple_global_amota_frames_count_quadratic,
                        triple_global_amota_frames_count_ci_baseline, triple_global_amota_frames_count_ci_static,
                        triple_global_amota_frames_count_ci_linear, triple_global_amota_frames_count_ci_quadratic,
                        triple_global_amota_frames_count_akf_baseline, triple_global_amota_frames_count_akf_static,
                        triple_global_amota_frames_count_akf_linear, triple_global_amota_frames_count_akf_quadratic,
                        triple_global_amota_frames_count_pf_baseline, triple_global_amota_frames_count_pf_static,
                        triple_global_amota_frames_count_pf_linear, triple_global_amota_frames_count_pf_quadratic,
                    ]
                    triple_amotp_counts = [
                        triple_global_amotp_frames_count_baseline, triple_global_amotp_frames_count_static,
                        triple_global_amotp_frames_count_linear, triple_global_amotp_frames_count_quadratic,
                        triple_global_amotp_frames_count_ci_baseline, triple_global_amotp_frames_count_ci_static,
                        triple_global_amotp_frames_count_ci_linear, triple_global_amotp_frames_count_ci_quadratic,
                        triple_global_amotp_frames_count_akf_baseline, triple_global_amotp_frames_count_akf_static,
                        triple_global_amotp_frames_count_akf_linear, triple_global_amotp_frames_count_akf_quadratic,
                        triple_global_amotp_frames_count_pf_baseline, triple_global_amotp_frames_count_pf_static,
                        triple_global_amotp_frames_count_pf_linear, triple_global_amotp_frames_count_pf_quadratic,
                    ]
                    triple_hota_accs = [
                        hota_acc_baseline, hota_acc_static, hota_acc_linear, hota_acc_quadratic,
                        hota_acc_ci_baseline, hota_acc_ci_static, hota_acc_ci_linear, hota_acc_ci_quadratic,
                        hota_acc_akf_baseline, hota_acc_akf_static, hota_acc_akf_linear, hota_acc_akf_quadratic,
                        hota_acc_pf_baseline, hota_acc_pf_static, hota_acc_pf_linear, hota_acc_pf_quadratic,
                    ]
                    for idx, (global_detection_result, total_X) in enumerate([
                        (global_detection_result_b, total_global_metrics_baseline),
                        (global_detection_result_s, total_global_metrics_static),
                        (global_detection_result_l, total_global_metrics_linear),
                        (global_detection_result_q, total_global_metrics_quadratic),
                        (global_detection_result_ci_b, total_global_metrics_ci_baseline),
                        (global_detection_result_ci_s, total_global_metrics_ci_static),
                        (global_detection_result_ci_l, total_global_metrics_ci_linear),
                        (global_detection_result_ci_q, total_global_metrics_ci_quadratic),
                        (global_detection_result_akf_b, total_global_metrics_akf_baseline),
                        (global_detection_result_akf_s, total_global_metrics_akf_static),
                        (global_detection_result_akf_l, total_global_metrics_akf_linear),
                        (global_detection_result_akf_q, total_global_metrics_akf_quadratic),
                        (global_detection_result_pf_b, total_global_metrics_pf_baseline),
                        (global_detection_result_pf_s, total_global_metrics_pf_static),
                        (global_detection_result_pf_l, total_global_metrics_pf_linear),
                        (global_detection_result_pf_q, total_global_metrics_pf_quadratic),
                    ]):
                        # Filter to only tracks updated this frame (exclude stale predicted-only tracks)
                        fresh_detections = [d for d in global_detection_result if hasattr(d, 'last_measurement') and abs(d.last_measurement - simulation_time_now) < 0.05]
                        amota_score = amota.calculate_amota(fresh_detections, all_detectable_ground_truth)
                        if amota_score is not None:
                            total_X["amota"] += amota_score
                            triple_amota_counts[idx] += 1
                        if len(fresh_detections) > 0:
                            amotp_score = amota.calculate_amotp(fresh_detections, all_detectable_ground_truth)
                            total_X["amotp"] += amotp_score
                            triple_amotp_counts[idx] += 1
                        # Feed HOTA accumulator with fresh detections vs GT
                        triple_hota_accs[idx].update(fresh_detections, all_detectable_ground_truth)
                        for cav_id in cav_id_list:
                            cav_gt_obj = ground_truth_cache.get(cav_id)
                            if cav_gt_obj is None:
                                cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci_conn, cav_id)
                            global_metrics = metrics.calculate_violations(cav_gt_obj, unique_detectable_ground_truth, global_detection_result, category="Aggressive", calc_amota=False, av_ids=av_ids)
                            for key in total_X:
                                total_X[key] += global_metrics.get(key, 0)
                    (triple_global_amota_frames_count_baseline, triple_global_amota_frames_count_static,
                     triple_global_amota_frames_count_linear, triple_global_amota_frames_count_quadratic,
                     triple_global_amota_frames_count_ci_baseline, triple_global_amota_frames_count_ci_static,
                     triple_global_amota_frames_count_ci_linear, triple_global_amota_frames_count_ci_quadratic,
                     triple_global_amota_frames_count_akf_baseline, triple_global_amota_frames_count_akf_static,
                     triple_global_amota_frames_count_akf_linear, triple_global_amota_frames_count_akf_quadratic,
                     triple_global_amota_frames_count_pf_baseline, triple_global_amota_frames_count_pf_static,
                     triple_global_amota_frames_count_pf_linear, triple_global_amota_frames_count_pf_quadratic) = triple_amota_counts
                    (triple_global_amotp_frames_count_baseline, triple_global_amotp_frames_count_static,
                     triple_global_amotp_frames_count_linear, triple_global_amotp_frames_count_quadratic,
                     triple_global_amotp_frames_count_ci_baseline, triple_global_amotp_frames_count_ci_static,
                     triple_global_amotp_frames_count_ci_linear, triple_global_amotp_frames_count_ci_quadratic,
                     triple_global_amotp_frames_count_akf_baseline, triple_global_amotp_frames_count_akf_static,
                     triple_global_amotp_frames_count_akf_linear, triple_global_amotp_frames_count_akf_quadratic,
                     triple_global_amotp_frames_count_pf_baseline, triple_global_amotp_frames_count_pf_static,
                     triple_global_amotp_frames_count_pf_linear, triple_global_amotp_frames_count_pf_quadratic) = triple_amotp_counts
            else:
                # Single global fusion (existing path)
                # Enable monitor=True for cooperative perception scoring
                global_fusion_result, global_detection_result, cooperative_monitoring, trupercept_monitoring = \
                    global_fusion.fuseDetectionFrame(simulation_time_now, monitor=True)
                
                # Process cooperative monitoring data for perception scoring
                perception_scorer.process_cooperative_monitoring(cooperative_monitoring)
                perception_scorer.process_trupercept_monitoring(trupercept_monitoring)
                perception_scorer.step()
                
                # Capture baseline at end of warmup period
                if step == (fast_forward_steps + warmup_steps) and not perception_baseline_captured:
                    perception_scorer.capture_baseline()
                    perception_baseline_captured = True
                
                # FEEDBACK LOOP: Update global fusion trust scores from perception scorer
                if use_trust_scoring:
                    # Calculate current trust scores and feed back to global fusion
                    perception_scorer.calculate_scores()
                    # Convert string participant IDs to int for global fusion
                    trust_scores = {
                        int(pid): score 
                        for pid, score in perception_scorer.current_scores.items()
                    }
                    global_fusion.update_trust_scores(trust_scores)
                
                # Only draw if visualization is enabled
                if visualize_global_fusion:
                    for each in global_detection_result:
                        polygon_bbox_id = each.draw_bounding_box_in_sumo(traci_conn, color=(255, 100, 0, 255), layer=9, sensor_package_id="global_fusion")
                        if polygon_bbox_id:
                            polygon_ids.append(polygon_bbox_id)

                # Only record metrics after warmup
                if is_recording:
                    # Convert the dictionary of all detectable ground truth objects to a list
                    all_detectable_ground_truth = list(unique_detectable_ground_truth.values())

                    # Filter to only tracks updated this frame (exclude stale predicted-only tracks)
                    fresh_detections = [d for d in global_detection_result if hasattr(d, 'last_measurement') and abs(d.last_measurement - simulation_time_now) < 0.05]

                    # Calculate AMOTA using unique detectable ground truth
                    amota_score = amota.calculate_amota(fresh_detections, all_detectable_ground_truth)

                    # Add the AMOTA score to the global metrics here because we only want it one time
                    if amota_score is not None:
                        total_global_metrics["amota"] += amota_score
                        global_amota_frames_count += 1

                    # Calculate AMOTP (localization error) using unique detectable ground truth
                    amotp_score = amota.calculate_amotp(fresh_detections, all_detectable_ground_truth)

                    # Add the AMOTP score whenever we have detections
                    if len(fresh_detections) > 0:
                        total_global_metrics["amotp"] += amotp_score
                        global_amotp_frames_count += 1

                    # Feed HOTA accumulator
                    global_hota_acc.update(fresh_detections, all_detectable_ground_truth)

                    # Calculate violations for the global fusion for each CAV
                    for cav_id in cav_id_list:
                        # Get the ground truth object for the CAV - reuse from cache
                        cav_gt_obj = ground_truth_cache.get(cav_id)
                        if cav_gt_obj is None:
                            # Fallback to creation if not in cache (shouldn't happen for active CAVs, but safe)
                            cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci_conn, cav_id)
                        global_metrics = metrics.calculate_violations(cav_gt_obj, unique_detectable_ground_truth, global_detection_result, category="Aggressive", calc_amota=False, av_ids=av_ids)
                        for key in total_global_metrics:
                            total_global_metrics[key] += global_metrics.get(key, 0)

                # TODO: Use global_detection_result for input to the motion planner

        global_fusion_time = time.time() - global_fusion_start_time

        # Update total times (always track timing, even during warmup)
        total_simulation_step_time += simulation_step_time
        total_cis_detection_time += cis_detection_time
        total_cav_detection_time += cav_detection_time
        total_global_fusion_time += global_fusion_time
        total_random_overhead_time += random_overhead_time
        iterations += 1
        
        # Track recorded steps
        if is_recording:
            recorded_steps += 1
        
        # Call progress callback if provided (every 10 steps to reduce overhead)
        if progress_callback is not None and step % 10 == 0:
            # Calculate current AMOTA values (when triple_fusion use static for display)
            if gpem_triple_fusion:
                _gfc = triple_global_amota_frames_count_static
                _gamo = total_global_metrics_static['amota'] if _gfc > 0 else 0
                current_global_amota = (_gamo / _gfc) if _gfc > 0 else 0
            else:
                current_global_amota = (total_global_metrics['amota'] / global_amota_frames_count) if global_amota_frames_count > 0 else 0
            current_cis_amota = (total_cis_metrics['amota'] / cis_amota_frames_count) if cis_amota_frames_count > 0 else 0
            current_cav_amota = (total_cav_metrics['amota'] / cav_amota_frames_count) if cav_amota_frames_count > 0 else 0
            
            progress_callback(step, total_expected_steps, {
                'global_amota': current_global_amota,
                'cis_amota': current_cis_amota,
                'cav_amota': current_cav_amota,
                'cav_count': len(cav_id_list),
                'recorded_steps': recorded_steps,
                'step_time': simulation_step_time,
            })

        step += 1
    
    # Cleanup progress callback
    if progress_callback is not None and hasattr(progress_callback, 'cleanup'):
        progress_callback.cleanup()

    traci_conn.close()
    
    # Calculate final perception scores
    final_perception_scores = perception_scorer.calculate_scores()
    perception_anomalies = perception_scorer.get_anomalies()
    perception_summary = perception_scorer.get_summary()

    out = {
        # Step counts
        "iterations": iterations,
        "recorded_steps": recorded_steps,
        "warmup_steps": warmup_steps,
        
        # Active entities
        "total_active_cavs": total_active_cavs,
        
        # Metrics
        "total_cav_metrics": total_cav_metrics,
        "total_cis_metrics": total_cis_metrics,
        "total_global_metrics": total_global_metrics,
        
        # AMOTA frame counts (for calculating averages)
        "cav_amota_frames_count": cav_amota_frames_count,
        "cis_amota_frames_count": cis_amota_frames_count,
        "global_amota_frames_count": global_amota_frames_count,
        # AMOTP frame counts (for calculating averages)
        "cav_amotp_frames_count": cav_amotp_frames_count,
        "cis_amotp_frames_count": cis_amotp_frames_count,
        "global_amotp_frames_count": global_amotp_frames_count,

        # Timing averages
        "avg_step_time": total_simulation_step_time / max(iterations, 1),
        "avg_cis_detection_time": total_cis_detection_time / max(iterations, 1),
        "avg_cav_detection_time": total_cav_detection_time / max(iterations, 1),
        "avg_global_fusion_time": total_global_fusion_time / max(iterations, 1),
        
        # Perception scoring (Conclave)
        "perception_scores": final_perception_scores,
        "perception_anomalies": perception_anomalies,
        "perception_summary": perception_summary,
        # HOTA metrics (computed from accumulated cross-frame data)
        "global_hota": global_hota_acc.compute(),
    }
    # When gpem_triple_fusion, add triple outputs (same run, three synced result sets)
    if gpem_triple_fusion:
        out["triple_global_metrics"] = {
            "static": total_global_metrics_static,
            "gpem_linear": total_global_metrics_linear,
            "gpem_quadratic": total_global_metrics_quadratic,
            "baseline": total_global_metrics_baseline,
            "ci_static": total_global_metrics_ci_static,
            "ci_gpem_linear": total_global_metrics_ci_linear,
            "ci_gpem_quadratic": total_global_metrics_ci_quadratic,
            "ci_baseline": total_global_metrics_ci_baseline,
            "akf_static": total_global_metrics_akf_static,
            "akf_gpem_linear": total_global_metrics_akf_linear,
            "akf_gpem_quadratic": total_global_metrics_akf_quadratic,
            "akf_baseline": total_global_metrics_akf_baseline,
            "pf_static": total_global_metrics_pf_static,
            "pf_gpem_linear": total_global_metrics_pf_linear,
            "pf_gpem_quadratic": total_global_metrics_pf_quadratic,
            "pf_baseline": total_global_metrics_pf_baseline,
        }
        out["triple_global_amota_frames_count"] = {
            "static": triple_global_amota_frames_count_static,
            "gpem_linear": triple_global_amota_frames_count_linear,
            "gpem_quadratic": triple_global_amota_frames_count_quadratic,
            "baseline": triple_global_amota_frames_count_baseline,
            "ci_static": triple_global_amota_frames_count_ci_static,
            "ci_gpem_linear": triple_global_amota_frames_count_ci_linear,
            "ci_gpem_quadratic": triple_global_amota_frames_count_ci_quadratic,
            "ci_baseline": triple_global_amota_frames_count_ci_baseline,
            "akf_static": triple_global_amota_frames_count_akf_static,
            "akf_gpem_linear": triple_global_amota_frames_count_akf_linear,
            "akf_gpem_quadratic": triple_global_amota_frames_count_akf_quadratic,
            "akf_baseline": triple_global_amota_frames_count_akf_baseline,
            "pf_static": triple_global_amota_frames_count_pf_static,
            "pf_gpem_linear": triple_global_amota_frames_count_pf_linear,
            "pf_gpem_quadratic": triple_global_amota_frames_count_pf_quadratic,
            "pf_baseline": triple_global_amota_frames_count_pf_baseline,
        }
        out["triple_global_amotp_frames_count"] = {
            "static": triple_global_amotp_frames_count_static,
            "gpem_linear": triple_global_amotp_frames_count_linear,
            "gpem_quadratic": triple_global_amotp_frames_count_quadratic,
            "baseline": triple_global_amotp_frames_count_baseline,
            "ci_static": triple_global_amotp_frames_count_ci_static,
            "ci_gpem_linear": triple_global_amotp_frames_count_ci_linear,
            "ci_gpem_quadratic": triple_global_amotp_frames_count_ci_quadratic,
            "ci_baseline": triple_global_amotp_frames_count_ci_baseline,
            "akf_static": triple_global_amotp_frames_count_akf_static,
            "akf_gpem_linear": triple_global_amotp_frames_count_akf_linear,
            "akf_gpem_quadratic": triple_global_amotp_frames_count_akf_quadratic,
            "akf_baseline": triple_global_amotp_frames_count_akf_baseline,
            "pf_static": triple_global_amotp_frames_count_pf_static,
            "pf_gpem_linear": triple_global_amotp_frames_count_pf_linear,
            "pf_gpem_quadratic": triple_global_amotp_frames_count_pf_quadratic,
            "pf_baseline": triple_global_amotp_frames_count_pf_baseline,
        }
        out["triple_global_hota"] = {
            "baseline": hota_acc_baseline.compute(),
            "static": hota_acc_static.compute(),
            "gpem_linear": hota_acc_linear.compute(),
            "gpem_quadratic": hota_acc_quadratic.compute(),
            "ci_baseline": hota_acc_ci_baseline.compute(),
            "ci_static": hota_acc_ci_static.compute(),
            "ci_gpem_linear": hota_acc_ci_linear.compute(),
            "ci_gpem_quadratic": hota_acc_ci_quadratic.compute(),
            "akf_baseline": hota_acc_akf_baseline.compute(),
            "akf_static": hota_acc_akf_static.compute(),
            "akf_gpem_linear": hota_acc_akf_linear.compute(),
            "akf_gpem_quadratic": hota_acc_akf_quadratic.compute(),
            "pf_baseline": hota_acc_pf_baseline.compute(),
            "pf_static": hota_acc_pf_static.compute(),
            "pf_gpem_linear": hota_acc_pf_linear.compute(),
            "pf_gpem_quadratic": hota_acc_pf_quadratic.compute(),
        }
    return out

def process_sensor_package(package_id, sensor_package_instance, gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now, cleanup_time):
    """
    Helper function to process a single sensor package (CAV or CIS) in a separate process.
    Returns the fusion state so it can be synchronized back to the main process.
    """
    fusion_result, detected_objects_for_metrics, detectable_ground_truth = \
        sensor_package_instance.create_detection_sets(gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now)
    
    # Extract fusion state to sync back to main process
    fusion_state = extract_fusion_state(sensor_package_instance.sensor_fusion)
    
    return fusion_result, detected_objects_for_metrics, detectable_ground_truth, package_id, fusion_state

def extract_fusion_state(fusion_obj):
    """
    Extract the essential state from a Fusion object for serialization.
    This allows us to sync state back from worker processes.
    """
    tracked_list_state = []
    for track in fusion_obj.tracked_list:
        track_state = {
            'x': track.x,
            'y': track.y,
            'dx': track.dx,
            'dy': track.dy,
            'error_covariance': track.error_covariance.tolist() if hasattr(track.error_covariance, 'tolist') else track.error_covariance,
            'd_covariance': track.d_covariance.tolist() if hasattr(track.d_covariance, 'tolist') else track.d_covariance,
            'last_measurement': track.last_measurement,
            'last_update': track.last_update,
            'id': track.id,
            'fusion_steps': track.fusion_steps,
            'width': track.width,
            'length': track.length,
            'angle': track.angle,
            # Kalman state
            'kalman_x': track.kalman.x,
            'kalman_y': track.kalman.y,
            'kalman_X_hat_t': track.kalman.X_hat_t.tolist() if hasattr(track.kalman.X_hat_t, 'tolist') else track.kalman.X_hat_t,
            'kalman_P_hat_t': track.kalman.P_hat_t.tolist() if hasattr(track.kalman.P_hat_t, 'tolist') else track.kalman.P_hat_t,
            'kalman_idx': track.kalman.idx,
            'kalman_last_update': track.kalman.last_update,
            'kalman_last_measurement': track.kalman.last_measurement,
        }
        tracked_list_state.append(track_state)
    
    return {
        'tracked_list': tracked_list_state,
        'current_track_id': fusion_obj.current_track_id,
    }

def update_fusion_state(fusion_obj, state):
    """
    Update a Fusion object with state from a worker process.
    """
    import numpy as np
    
    fusion_obj.current_track_id = state['current_track_id']
    
    # Update existing tracks and add new ones
    state_track_ids = {t['id'] for t in state['tracked_list']}
    existing_track_ids = {t.id for t in fusion_obj.tracked_list}
    
    # Update existing tracks
    for track in fusion_obj.tracked_list:
        for track_state in state['tracked_list']:
            if track.id == track_state['id']:
                track.x = track_state['x']
                track.y = track_state['y']
                track.dx = track_state['dx']
                track.dy = track_state['dy']
                track.error_covariance = np.array(track_state['error_covariance'])
                track.d_covariance = np.array(track_state['d_covariance'])
                track.last_measurement = track_state['last_measurement']
                track.last_update = track_state['last_update']
                track.fusion_steps = track_state['fusion_steps']
                track.width = track_state['width']
                track.length = track_state['length']
                track.angle = track_state['angle']
                # Update Kalman state
                track.kalman.x = track_state['kalman_x']
                track.kalman.y = track_state['kalman_y']
                track.kalman.X_hat_t = np.array(track_state['kalman_X_hat_t'])
                track.kalman.P_hat_t = np.array(track_state['kalman_P_hat_t'])
                track.kalman.idx = track_state['kalman_idx']
                track.kalman.last_update = track_state['kalman_last_update']
                track.kalman.last_measurement = track_state['kalman_last_measurement']
                break
    
    # Remove tracks that are no longer in state
    fusion_obj.tracked_list = [t for t in fusion_obj.tracked_list if t.id in state_track_ids]
    
    # Note: Adding new tracks would require recreating GlobalTracked objects
    # For now, we rely on the main process to handle new track creation
    # This is a limitation of the current approach

# Main execution block - for quick testing
# For experiments with multiple runs and configs, use experiment_runner.py
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CMR SUMO simulation")
    parser.add_argument("--config", type=str, default="perfect", 
                        choices=["perfect", "normal"],
                        help="Config preset to use")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup steps before recording metrics")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps to run (None = run to completion)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config == "perfect":
        config = get_perfect_config()
    else:
        config = get_normal_sensors_config()
    
    # Apply command line overrides
    config["warmup_steps"] = args.warmup
    if args.max_steps:
        config["max_steps"] = args.max_steps
    
    print(f"Running simulation with {args.config} config...")
    print(f"  Warmup steps: {config.get('warmup_steps', 0)}")
    print(f"  Max steps: {config.get('max_steps', 'unlimited')}")
    
    result = run_simulation(config)
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total iterations: {result['iterations']}")
    print(f"Recorded steps: {result['recorded_steps']}")
    print(f"Total active CAVs: {result['total_active_cavs']}")
    
    # Calculate averages
    if result['cav_amota_frames_count'] > 0:
        avg_cav_amota = result['total_cav_metrics']['amota'] / result['cav_amota_frames_count']
        print(f"Average CAV AMOTA: {avg_cav_amota:.4f}")
    
    if result['cis_amota_frames_count'] > 0:
        avg_cis_amota = result['total_cis_metrics']['amota'] / result['cis_amota_frames_count']
        print(f"Average CIS AMOTA: {avg_cis_amota:.4f}")
    
    if result['global_amota_frames_count'] > 0:
        avg_global_amota = result['total_global_metrics']['amota'] / result['global_amota_frames_count']
        print(f"Average Global AMOTA: {avg_global_amota:.4f}")
    
    print(f"\nTiming:")
    print(f"  Avg step time: {result['avg_step_time']*1000:.2f}ms")
    print(f"  Avg CIS detection: {result['avg_cis_detection_time']*1000:.2f}ms")
    print(f"  Avg CAV detection: {result['avg_cav_detection_time']*1000:.2f}ms")
    print(f"  Avg global fusion: {result['avg_global_fusion_time']*1000:.2f}ms")
    
    # Cleanup process pool
    shutdown_process_pool()
    
    print("\nFor multi-run experiments, use: python experiment_runner.py --help")
