import random
from enum import Enum
from sensor import DetectedObject
import numpy as np
import math

class ErrorType(Enum):
    SINGLE_SENSOR_EXTRINSICS = "Single sensor extrinsics"
    MULTI_SENSOR_EXTRINSICS = "Multi sensor extrinsics"
    LOCALIZATION = "Localization"
    MALICIOUS_REMOVAL = "Malicious Removal"
    MALICIOUS_ADDITION = "Malicious Addition"
    MALICIOUS_CONVOY = "Malicious Convoy"
    HIGH_MISS_RATE = "High Miss Rate"  # Sensor misses more than it reports
    DETECTION_LAG = "Detection Lag"    # Detections are delayed by N steps

class ErrorPackage:
    """
    Class representing an error package that injects errors based on the specified error type.
    
    Error injection has two components:
    1. CAV_ERROR_PROBABILITY (10% default): Which percentage of CAVs are affected
       - This is set ONCE when CAVs are created and is PERMANENT for that CAV
    2. severity: How severe the error is (controlled by error_level parameter)
       - Extrinsics: 1-10 degrees of angular error
       - Localization: 10-100% of max error (1 meter position, 10 degrees angle)
       - Malicious: probability of injection per frame (10-100%)
    
    Attributes:
        error_type (ErrorType): The type of error to inject.
        severity (float): The severity of the error (1-10 or 10-100 depending on type).
        is_active (bool): Whether error injection is currently active.
    """
    
    # Fixed percentage of CAVs that will have errors (set once at creation)
    CAV_ERROR_PROBABILITY = 0.10  # 10% of CAVs
    
    # Maximum error magnitudes
    MAX_POSITION_ERROR_M = 1.0  # 1 meter max position error
    MAX_ANGLE_ERROR_DEG = 10.0  # 10 degrees max angle error
    
    def __init__(self, error_type, severity):
        """
        Initialize the ErrorPackage with the specified error type and severity.
        
        Args:
            error_type (ErrorType): The type of error to inject.
            severity (float): The severity level:
                - Extrinsics: 1-10 (degrees of error)
                - Localization: 10-100 (percentage of max error)
                - Malicious: 10-100 (percentage probability per frame)
        """
        self.error_type = error_type
        self.severity = severity
        self.is_active = True  # Can be toggled to enable/disable error injection
        
        # Pre-calculate severity factors for different error types
        if error_type in [ErrorType.SINGLE_SENSOR_EXTRINSICS, ErrorType.MULTI_SENSOR_EXTRINSICS]:
            # Extrinsics: severity = degrees (1-10)
            # Also add proportional position error (0.1-1.0 meters)
            self.angle_error_rad = math.radians(severity)  # Convert degrees to radians
            self.position_error_m = severity / 10.0  # 1 deg = 0.1m, 10 deg = 1.0m
        elif error_type == ErrorType.LOCALIZATION:
            # Localization: severity = 10-100 (percentage of max)
            self.error_scale = severity / 100.0  # 10 = 0.1, 100 = 1.0
        elif error_type == ErrorType.HIGH_MISS_RATE:
            # High miss rate: severity = additional miss percentage (10-100)
            # e.g., severity=50 means 50% additional miss rate on top of normal
            self.additional_miss_rate = severity / 100.0
        elif error_type == ErrorType.DETECTION_LAG:
            # Detection lag: severity = number of steps to delay (1-10)
            self.lag_steps = int(severity)
            # Buffer to store delayed detections: {vehicle_id: deque of (step, detections)}
            self.detection_buffer = {}
        else:
            # Malicious: severity = probability percentage per frame
            self.probability = severity / 100.0  # 10 = 0.1, 100 = 1.0

    def inject_error(self, data, has_error):
        """
        Inject the specified error into the data if the vehicle has an error flag.
        
        Args:
            data: The data to inject the error into.
            has_error (bool): Whether this vehicle was marked as having errors (set at creation).
        
        Returns:
            The data with the injected error (if applicable).
        """
        if not self.is_active:
            return data
        if not has_error:
            return data
            
        # Vehicle has permanent error flag - always inject error (severity controls magnitude)
        if self.error_type == ErrorType.SINGLE_SENSOR_EXTRINSICS:
            return self.inject_single_sensor_extrinsics_error(data)
        elif self.error_type == ErrorType.MULTI_SENSOR_EXTRINSICS:
            return self.inject_multi_sensor_extrinsics_error(data)
        elif self.error_type == ErrorType.LOCALIZATION:
            return self.inject_localization_error(data)
        elif self.error_type == ErrorType.MALICIOUS_REMOVAL:
            return self.inject_malicious_removal_error(data)
        elif self.error_type == ErrorType.MALICIOUS_ADDITION:
            return self.inject_malicious_addition_error(data)
        elif self.error_type == ErrorType.MALICIOUS_CONVOY:
            return self.inject_malicious_convoy_error(data)
        elif self.error_type == ErrorType.HIGH_MISS_RATE:
            return self.inject_high_miss_rate_error(data)
        elif self.error_type == ErrorType.DETECTION_LAG:
            # Detection lag requires vehicle_id and current_step - handled separately
            # This will be called with (data, vehicle_id, current_step) from sensor_package
            return data  # Fallback - actual lag handled in inject_detection_lag_error
        return data

    def inject_single_sensor_extrinsics_error(self, data):
        """
        Inject a single sensor extrinsics error by shifting its position and rotating its angle.
        
        Severity controls the magnitude:
        - severity=1: 1 degree rotation, 0.1m position shift
        - severity=10: 10 degree rotation, 1.0m position shift
        
        Args:
            data (Sensor): The sensor object to inject the error into.
        
        Returns:
            Sensor: The sensor object with the injected error.
        """
        # Apply position error scaled by severity
        data.x_offset_error += random.uniform(-self.position_error_m, self.position_error_m)
        data.y_offset_error += random.uniform(-self.position_error_m, self.position_error_m)
        # Apply angle error directly from severity (in radians)
        data.angle_offset_error += random.uniform(-self.angle_error_rad, self.angle_error_rad)
        return data

    def inject_multi_sensor_extrinsics_error(self, data):
        """
        Inject a multi-sensor extrinsics error by shifting the positions and rotating the angles of multiple sensor objects.
        
        Args:
            data (list): A list of Sensor objects to inject the error into.
        
        Returns:
            list: The list of Sensor objects with the injected error.
        """
        # Apply the same error to all sensors in the list to simulate a common extrinsics error
        x_shift = random.uniform(-self.position_error_m, self.position_error_m)
        y_shift = random.uniform(-self.position_error_m, self.position_error_m)
        angle_shift = random.uniform(-self.angle_error_rad, self.angle_error_rad)

        for sensor_obj in data:
            sensor_obj.x_offset_error += x_shift
            sensor_obj.y_offset_error += y_shift
            sensor_obj.angle_offset_error += angle_shift
        return data

    def inject_localization_error(self, data):
        """
        Inject a localization error by shifting the position and rotating the angle.
        
        Severity controls the magnitude:
        - severity=10: 10% of max (0.1m position, 1 degree angle)
        - severity=100: 100% of max (1.0m position, 10 degrees angle)
        
        Args:
            data (dict): A dictionary containing 'x', 'y', and 'yaw' of the localization.
        
        Returns:
            dict: The localization data with the injected error.
        """
        # Scale errors by severity (10-100%)
        max_pos = self.MAX_POSITION_ERROR_M * self.error_scale
        max_angle = math.radians(self.MAX_ANGLE_ERROR_DEG) * self.error_scale
        
        data['x'] += random.uniform(-max_pos, max_pos)
        data['y'] += random.uniform(-max_pos, max_pos)
        data['yaw'] += random.uniform(-max_angle, max_angle)
        return data

    def inject_malicious_removal_error(self, data):
        """
        Inject a malicious removal error by removing detections.
        
        Severity controls what percentage of detections are removed:
        - severity=10: Remove 10% of detections
        - severity=50: Remove 50% of detections  
        - severity=100: Remove all detections
        
        Args:
            data (list): A list of DetectedObject instances.
        
        Returns:
            list: The data with the injected error (some or all removed).
        """
        if not data:
            return data
        
        # Remove a percentage of detections based on severity
        num_to_remove = int(len(data) * self.probability)
        
        # Always remove at least 1 if probability > 0 and we have data
        if num_to_remove == 0 and self.probability > 0 and len(data) > 0:
            num_to_remove = 1
        
        # Remove random detections
        for _ in range(min(num_to_remove, len(data))):
            if data:  # Check in case list becomes empty
                data.pop(random.randint(0, len(data) - 1))
        
        return data

    def inject_malicious_addition_error(self, data):
        """
        Inject a malicious addition error by adding fake detections.
        
        Severity controls how many fake detections to add:
        - severity=10: Add 1 fake detection per frame
        - severity=50: Add 2-3 fake detections per frame
        - severity=100: Add 5 fake detections per frame
        
        Args:
            data (list): The list of DetectedObject instances to inject the error into.
        
        Returns:
            list: The data with the injected error.
        """
        # Scale number of fake detections by severity
        # severity=10 → 1, severity=50 → 2-3, severity=100 → 5
        num_to_add = max(1, int(self.probability * 5))
        
        for _ in range(num_to_add):
            fake_detection = self.create_fake_detection()
            data.append(fake_detection)
        
        return data

    def inject_malicious_convoy_error(self, data):
        """
        Inject a malicious convoy error by consistently spawning fake detections in a designated area.
        
        Severity controls the convoy size:
        - severity=10: Inject 1 fake vehicle in convoy area
        - severity=50: Inject 2-3 fake vehicles in convoy area
        - severity=100: Inject 5 fake vehicles in convoy area
        
        Args:
            data (list): The list of DetectedObject instances to inject the error into.
        
        Returns:
            list: The data with the injected error.
        """
        # Define a designated area for the convoy
        convoy_area_x_min, convoy_area_x_max = 20, 40
        convoy_area_y_min, convoy_area_y_max = 20, 40

        # Scale number of convoy vehicles by severity
        num_convoy = max(1, int(self.probability * 5))
        
        for _ in range(num_convoy):
            fake_detection = self.create_fake_detection()
            
            # Set the fake detection's centroid within the convoy area
            fake_detection.centroid = (random.uniform(convoy_area_x_min, convoy_area_x_max),
                                       random.uniform(convoy_area_y_min, convoy_area_y_max))
            data.append(fake_detection)
        
        return data

    def inject_high_miss_rate_error(self, data):
        """
        Inject a high miss rate error by randomly dropping detections.
        
        This simulates a sensor that has a higher miss rate than it reports,
        meaning the covariance doesn't account for this additional uncertainty.
        
        Severity controls additional miss percentage:
        - severity=10: 10% additional miss rate
        - severity=50: 50% additional miss rate
        - severity=100: 100% additional miss rate (all detections dropped)
        
        Args:
            data (list): A list of DetectedObject instances.
        
        Returns:
            list: The data with some detections removed.
        """
        if not data:
            return data
        
        # Filter out detections based on additional miss rate
        surviving_detections = []
        for detection in data:
            if random.random() > self.additional_miss_rate:
                surviving_detections.append(detection)
        
        return surviving_detections

    def inject_detection_lag_error(self, data, vehicle_id, current_step):
        """
        Inject a detection lag error by buffering detections and returning old ones.
        
        This simulates processing delays where detections are delayed by N steps.
        The system sees stale data, but reports current covariance.
        
        Severity controls lag in steps:
        - severity=1: 1 step delay (~0.1 seconds)
        - severity=5: 5 step delay (~0.5 seconds)
        - severity=10: 10 step delay (~1.0 second)
        
        Args:
            data (list): Current list of DetectedObject instances.
            vehicle_id (str): ID of the vehicle (for per-vehicle buffering).
            current_step (int): Current simulation step.
        
        Returns:
            list: Delayed detections from lag_steps ago (or empty if buffer not full).
        """
        from collections import deque
        
        # Initialize buffer for this vehicle if needed
        if vehicle_id not in self.detection_buffer:
            self.detection_buffer[vehicle_id] = deque(maxlen=self.lag_steps + 1)
        
        buffer = self.detection_buffer[vehicle_id]
        
        # Store current detections with timestamp
        buffer.append((current_step, data))
        
        # Return delayed detections if buffer is full
        if len(buffer) > self.lag_steps:
            _, delayed_data = buffer[0]
            return delayed_data
        else:
            # Buffer not yet full - return empty (simulating startup delay)
            return []

    def create_fake_detection(self):
        """
        Create a fake detection.
        
        Returns:
            DetectedObject: A fake DetectedObject instance.
        """
        fake_id = "fake_" + str(random.randint(100000, 999999))
        fake_centroid = (random.uniform(-100, 100), random.uniform(-100, 100))
        fake_width = random.uniform(1.5, 2.5)
        fake_length = random.uniform(3.0, 5.0)
        fake_angle = random.uniform(0, 2 * math.pi)
        fake_velocity_vector = (random.uniform(-5, 5), random.uniform(-5, 5))
        fake_error_covariance = np.array([[random.uniform(0.1, 0.5), 0],
                                          [0, random.uniform(0.1, 0.5)]], dtype='float')

        # Create a fake DetectedObject instance
        fake_detection_object = DetectedObject(
            vehicle_id=fake_id,
            vehicle_type="car",
            detected_bbox=None,
            centroid=fake_centroid,
            width=fake_width,
            length=fake_length,
            angle=fake_angle,
            expected_error_gaussian=None,
            velocity_vector=fake_velocity_vector,
            error_covariance=fake_error_covariance
        )
        return fake_detection_object
    
    @classmethod
    def should_cav_have_error(cls) -> bool:
        """
        Determine if a newly created CAV should have errors.
        
        Called once when a CAV is created. If True, this CAV will permanently
        have errors for the duration of the simulation.
        
        Returns:
            bool: True if this CAV should have errors (10% probability)
        """
        return random.random() < cls.CAV_ERROR_PROBABILITY
