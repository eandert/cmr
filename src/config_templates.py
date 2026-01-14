import os

def get_perfect_config():
    """
    Returns a configuration dictionary for a 'perfect' simulation.
    """
    config = {
        # SUMO simulation settings
        "sumo_config_path": os.path.join("maps", "single", "osm.sumocfg"),
        "step_length": 0.1,

        # Sensor and detector configurations
        "sensor_types": ["OS1_128"],  # Corresponds to SensorType.OS1_128
        "detector_types": ["PERFECT"], # Corresponds to DetectorType.PERFECT

        # Sensor extrinsics (x, y, yaw) relative to the SensorPackage
        "sensor_extrinsics": [
            (0.0, 0.0, 0.0) # Single perfect sensor at the origin of the SensorPackage
        ],

        # Vehicle and traffic light probabilities
        "cav_probability": 0.1,
        "tfl_probability": 1,

        # Localization error settings
        "lateral_error_coefficients": [0.01, 0.001],
        "longitudinal_error_coefficients": [0.01, 0.001],

        # Error injection settings (for a 'perfect' config, these are disabled)
        "error_type": "None",
        "error_level": 0,

        # Track management settings
        "cleanup_time": 0.5, # Configurable cleanup time for GlobalTracked objects

        # Visualization settings
        "visualize_ground_truth": False,
        "visualize_cis_detections": False,
        "visualize_cav_detections": False,
        "visualize_global_fusion": True,
        "do_global_fusion": True
    }
    return config

def get_normal_sensors_config():
    """
    Returns a configuration dictionary for a simulation with normal camera and lidar sensors.
    """
    config = {
        # SUMO simulation settings
        "sumo_config_path": os.path.join("maps", "single", "osm.sumocfg"),
        "step_length": 0.1,

        # Sensor and detector configurations
        "sensor_types": ["OS1_128", "CAMERA"],  # Corresponds to SensorType enums
        "detector_types": ["POINT_PILLARS", "YOLO"], # Corresponds to DetectorType enums
        
        # Sensor extrinsics (x, y, yaw) relative to the SensorPackage for each sensor in sensor_types
        "sensor_extrinsics": [
            (0.0, 0.0, 0.0),  # LiDAR sensor at the origin
            (-1.0, 0.0, 0.0)  # Camera sensor 1 meter forward
        ],

        # Vehicle and traffic light probabilities
        "cav_probability": 0.1,
        "tfl_probability": 1,

        # Localization error settings
        "lateral_error_coefficients": [0.01, 0.001],
        "longitudinal_error_coefficients": [0.01, 0.001],

        # Error injection settings (for a 'normal' config, these are disabled by default)
        "error_type": "None",
        "error_level": 0,

        # Track management settings
        "cleanup_time": 0.5, # Configurable cleanup time for GlobalTracked objects

        # Visualization settings
        "visualize_ground_truth": True,
        "visualize_cis_detections": True,
        "visualize_cav_detections": True,
        "visualize_global_fusion": True,
        "do_global_fusion": True
    }
    return config
