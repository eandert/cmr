"""
Configuration templates for CMR simulation.

Each template returns a configuration dictionary that can be passed to run_simulation().
Templates can be modified or extended for specific experiments.

Maps: "fast_city" (default), "fast_highway", "fast_rural", "city", "highway", "rural". Use with_sumo_map(config, name) to switch map.
"""

import os
import copy

# Available SUMO map directories under maps/
SUMO_MAP_FAST_CITY = "fast_city"
SUMO_MAP_FAST_HIGHWAY = "fast_highway"
SUMO_MAP_FAST_RURAL = "fast_rural"
SUMO_MAP_CITY = "city"
SUMO_MAP_HIGHWAY = "highway"
SUMO_MAP_RURAL = "rural"
SUMO_MAP_TEMPE_2X3 = "tempe_2x3"  # legacy alias


def with_sumo_map(config, map_subdir):
    """
    Return a copy of config that uses the given SUMO map.

    Args:
        config: Configuration dict (e.g. from get_perfect_config(), get_pointpillars_os1_128_config()).
        map_subdir: One of "fast_city", "fast_highway", "fast_rural", "city", "highway", "rural" (directory name under maps/).

    Returns:
        New config dict with sumo_config_path set to maps/<map_subdir>/osm.sumocfg.

    Example:
        config = with_sumo_map(get_perfect_config(), "tempe_2x3")
    """
    c = copy.deepcopy(config)
    c["sumo_config_path"] = os.path.join("maps", map_subdir, "osm.sumocfg")
    return c


def get_base_config():
    """
    Returns a base configuration dictionary with sensible defaults.
    Use this as a starting point and override specific values.
    """
    return {
        # SUMO simulation settings
        "sumo_config_path": os.path.join("maps", "fast_city", "osm.sumocfg"),
        "step_length": 0.1,

        # Sensor and detector configurations
        "sensor_types": ["OS1_128"],
        "detector_types": ["PERFECT"],
        "sensor_extrinsics": [(0.0, 0.0, 0.0)],

        # Vehicle and traffic light probabilities
        "cav_probability": 0.1,
        "tfl_probability": 1,

        # Localization error settings
        "lateral_error_coefficients": [0.01, 0.001],
        "longitudinal_error_coefficients": [0.01, 0.001],

        # Error injection settings
        "error_type": "None",
        "error_level": 0,

        # Track management settings
        "cleanup_time": 0.5,

        # Trust scoring settings
        "use_trust_scoring": True,  # Enable/disable trust-based filtering in sensor fusion
        
        # Error modeling settings
        "use_gpem_model": False,  # Use GPEM parameterized error model vs static covariance
        "use_quadratic": False,  # Use quadratic regression instead of linear (when use_gpem_model=True)
        "detector_max_range": 100.0,  # Max detection range in meters (matches 100m evaluation data)

        # Visualization settings (off by default for automated runs)
        "visualize_ground_truth": False,
        "visualize_cis_detections": False,
        "visualize_cav_detections": False,
        "visualize_global_fusion": False,
        "do_global_fusion": True,
        
        # Parallelization
        "use_parallel": True,
        
        # Data collection timing (used by experiment_runner)
        "warmup_steps": 0,
        "record_steps": None,  # None = record all
        "max_steps": None,     # None = run to completion
        
        # Error injection timing
        "error_injection_start": 0,
        "error_injection_end": None,  # None = until end
    }


def get_perfect_config():
    """
    Returns a configuration dictionary for a 'perfect' simulation.
    Uses perfect sensors with no noise or errors.
    """
    config = get_base_config()
    config.update({
        "sensor_types": ["OS1_128"],
        "detector_types": ["PERFECT"],
        "sensor_extrinsics": [(0.0, 0.0, 0.0)],
        "error_type": "None",
        "error_level": 0,
    })
    return config

def get_normal_sensors_config():
    """
    Returns a configuration dictionary for a simulation with normal camera and lidar sensors.
    Uses realistic sensor noise and detection models (LiDAR: PointPillars, camera: DETR3D).
    """
    config = get_base_config()
    config.update({
        "sensor_types": ["OS1_128", "CAMERA"],
        "detector_types": ["POINTPILLARS_KITTI", "DETR3D"],
        "sensor_extrinsics": [
            (0.0, 0.0, 0.0),   # LiDAR sensor at the origin
            (-1.0, 0.0, 0.0)  # Camera sensor 1 meter forward
        ],
    })
    return config


def get_lidar_only_config():
    """
    Returns a configuration with only LiDAR sensor.
    """
    config = get_base_config()
    config.update({
        "sensor_types": ["OS1_128"],
        "detector_types": ["POINTPILLARS_KITTI"],
        "sensor_extrinsics": [(0.0, 0.0, 0.0)],
    })
    return config


def get_pointpillars_os1_128_config():
    """
    Returns a configuration for PointPillars KITTI with OS1_128 LiDAR.
    
    This uses the regression-tested error model from actual sensor testing,
    with distance-binned error distributions (Normal, Laplace, Logistic, Student-t).
    
    The actual sensor suite has 4 cameras for low-level fusion, but since we're
    simulating late fusion, we model it as a single sensor with the combined 
    PointPillars + camera fusion error characteristics.
    """
    config = get_base_config()
    config.update({
        "sensor_types": ["OS1_128"],
        "detector_types": ["POINTPILLARS_KITTI"],
        "sensor_extrinsics": [(0.0, 0.0, 0.0)],
    })
    return config


def get_camera_only_config():
    """
    Returns a configuration with only camera sensor (DETR3D).
    """
    config = get_base_config()
    config.update({
        "sensor_types": ["CAMERA"],
        "detector_types": ["DETR3D"],
        "sensor_extrinsics": [(0.0, 0.0, 0.0)],
    })
    return config


def get_error_injection_config(base_config=None, error_type="SINGLE_SENSOR_EXTRINSICS", error_level=1):
    """
    Returns a configuration with error injection enabled.
    
    Args:
        base_config: Base configuration to modify (default: perfect config)
        error_type: Type of error to inject (string matching ErrorType enum)
        error_level: Severity level of error (0-10)
    
    Returns:
        Configuration dictionary with error injection enabled
    """
    if base_config is None:
        config = get_perfect_config()
    else:
        config = copy.deepcopy(base_config)
    
    config.update({
        "error_type": error_type,
        "error_level": error_level,
    })
    return config


def get_visualization_config():
    """
    Returns a configuration with all visualizations enabled.
    Useful for debugging and demos.
    """
    config = get_perfect_config()
    config.update({
        "visualize_ground_truth": True,
        "visualize_cis_detections": True,
        "visualize_cav_detections": True,
        "visualize_global_fusion": True,
    })
    return config


def create_experiment_config(
    name: str,
    sensor_types: list = None,
    detector_types: list = None,
    error_type: str = "None",
    error_level: int = 0,
    warmup_steps: int = 600,
    record_steps: int = 6000,
    **kwargs
):
    """
    Create a custom experiment configuration.
    
    Args:
        name: Name for this configuration (for logging)
        sensor_types: List of sensor type names
        detector_types: List of detector type names
        error_type: Error type to inject
        error_level: Error severity level
        warmup_steps: Steps before recording
        record_steps: Number of steps to record
        **kwargs: Additional config overrides
    
    Returns:
        Configuration dictionary
    """
    config = get_base_config()
    
    if sensor_types:
        config["sensor_types"] = sensor_types
    if detector_types:
        config["detector_types"] = detector_types
    
    config.update({
        "error_type": error_type,
        "error_level": error_level,
        "warmup_steps": warmup_steps,
        "record_steps": record_steps,
    })
    
    # Apply any additional overrides
    config.update(kwargs)
    
    return config
