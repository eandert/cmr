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

class ErrorPackage:
    """
    Class representing an error package that injects errors based on the specified error type and probability.
    
    Attributes:
        error_type (ErrorType): The type of error to inject.
        probability (float): The probability of injecting the error (0 to 100%).
    """
    
    def __init__(self, error_type, probability):
        """
        Initialize the ErrorPackage with the specified error type and probability.
        
        Args:
            error_type (ErrorType): The type of error to inject.
            probability (float): The probability of injecting the error (0 to 100%).
        """
        self.error_type = error_type
        self.probability = probability / 100.0  # Convert percentage to a probability value between 0 and 1

    def inject_error(self, data, has_error):
        """
        Inject the specified error into the data based on the probability and whether the vehicle has an error.
        
        Args:
            data: The data to inject the error into.
            has_error (bool): Whether the vehicle has an error.
        
        Returns:
            The data with the injected error.
        """
        if has_error and random.random() < self.probability:
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
        return data

    def inject_single_sensor_extrinsics_error(self, data):
        """
        Inject a single sensor extrinsics error by shifting its position and rotating its angle.
        
        Args:
            data (Sensor): The sensor object to inject the error into.
        
        Returns:
            Sensor: The sensor object with the injected error.
        """
        # Shift position by a random amount (e.g., up to 1 meter)
        data.x_offset_error += random.uniform(-1, 1)
        data.y_offset_error += random.uniform(-1, 1)
        # Rotate angle by a random amount (e.g., up to 10 degrees or ~0.17 radians)
        data.angle_offset_error += random.uniform(-0.17, 0.17)
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
        x_shift = random.uniform(-1, 1)
        y_shift = random.uniform(-1, 1)
        angle_shift = random.uniform(-0.17, 0.17) # ~10 degrees in radians

        for sensor_obj in data:
            sensor_obj.x_offset_error += x_shift
            sensor_obj.y_offset_error += y_shift
            sensor_obj.angle_offset_error += angle_shift
        return data

    def inject_localization_error(self, data):
        """
        Inject a localization error by shifting the position and rotating the angle of the localization data.
        
        Args:
            data (dict): A dictionary containing 'x', 'y', and 'yaw' of the localization.
        
        Returns:
            dict: The localization data with the injected error.
        """
        # Shift position by a random amount (e.g., up to 1 meter)
        data['x'] += random.uniform(-1, 1) * self.probability
        data['y'] += random.uniform(-1, 1) * self.probability
        # Rotate angle by a random amount (e.g., up to 10 degrees or ~0.17 radians)
        data['yaw'] += random.uniform(-0.17, 0.17) * self.probability
        return data

    def inject_malicious_removal_error(self, data):
        """
        Inject a malicious removal error by removing a detection with a certain probability.
        
        Args:
            data (list): A list of DetectedObject instances.
        
        Returns:
            list: The data with the injected error.
        """
        # Remove a detection with a certain probability
        if random.random() < self.probability:
            if data:
                data.pop(random.randint(0, len(data) - 1))
        return data

    def inject_malicious_addition_error(self, data):
        """
        Inject a malicious addition error by adding a detection with a certain probability.
        
        Args:
            data (list): The list of DetectedObject instances to inject the error into.
        
        Returns:
            list: The data with the injected error.
        """
        # Add a detection with a certain probability
        if random.random() < self.probability:
            fake_detection = self.create_fake_detection()
            data.append(fake_detection)
        return data

    def inject_malicious_convoy_error(self, data):
        """
        Inject a malicious convoy error by consistently spawning a fake detection within a designated area.
        
        Args:
            data (list): The list of DetectedObject instances to inject the error into.
        
        Returns:
            list: The data with the injected error.
        """
        # Consistently spawn a fake detection within a designated area
        if random.random() < self.probability:
            # Define a designated area for the convoy (e.g., a fixed rectangle)
            convoy_area_x_min, convoy_area_x_max = 20, 40
            convoy_area_y_min, convoy_area_y_max = 20, 40

            fake_detection = self.create_fake_detection()
            
            # Set the fake detection's centroid within the convoy area
            fake_detection.centroid = (random.uniform(convoy_area_x_min, convoy_area_x_max),
                                       random.uniform(convoy_area_y_min, convoy_area_y_max))
            data.append(fake_detection)
        return data

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
            vehicle_type="car", # Or some other appropriate type
            detected_bbox=None, # Will be calculated by property if needed
            centroid=fake_centroid,
            width=fake_width,
            length=fake_length,
            angle=fake_angle,
            expected_error_gaussian=None, # Not relevant for a fake detection
            velocity_vector=fake_velocity_vector,
            error_covariance=fake_error_covariance
        )
        return fake_detection_object