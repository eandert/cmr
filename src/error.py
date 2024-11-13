import random

class ErrorPackage:
    """
    Class representing an error package that injects errors based on the specified error type and probability.
    
    Attributes:
        error_type (str): The type of error to inject.
        probability (float): The probability of injecting the error (0 to 100%).
    """
    
    def __init__(self, error_type, probability):
        """
        Initialize the ErrorPackage with the specified error type and probability.
        
        Args:
            error_type (str): The type of error to inject.
            probability (float): The probability of injecting the error (0 to 100%).
        """
        self.error_type = error_type
        self.probability = probability / 100.0  # Convert percentage to a probability value between 0 and 1

    def inject_error(self, data):
        """
        Inject the specified error into the data based on the probability.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        if random.random() < self.probability:
            if self.error_type == "Single sensor extrinsics":
                return self.inject_single_sensor_extrinsics_error(data)
            elif self.error_type == "Multi sensor extrinsics":
                return self.inject_multi_sensor_extrinsics_error(data)
            elif self.error_type == "Localization":
                return self.inject_localization_error(data)
            elif self.error_type == "Malicious Removal":
                return self.inject_malicious_removal_error(data)
            elif self.error_type == "Malicious Addition":
                return self.inject_malicious_addition_error(data)
            elif self.error_type == "Malicious Convoy":
                return self.inject_malicious_convoy_error(data)
        return data

    def inject_single_sensor_extrinsics_error(self, data):
        """
        Inject a single sensor extrinsics error by increasing the variance.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        # Increase the variance by a random percentage between 0 and 100%
        variance_increase = random.uniform(0, 1)
        data['variance'] *= (1 + variance_increase)
        return data

    def inject_multi_sensor_extrinsics_error(self, data):
        """
        Inject a multi sensor extrinsics error by increasing the variance.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        # Increase the variance by a random percentage between 0 and 100%
        variance_increase = random.uniform(0, 1)
        data['variance'] *= (1 + variance_increase)
        return data

    def inject_localization_error(self, data):
        """
        Inject a localization error by increasing the variance.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        # Increase the variance by a random percentage between 0 and 100%
        variance_increase = random.uniform(0, 1)
        data['variance'] *= (1 + variance_increase)
        return data

    def inject_malicious_removal_error(self, data):
        """
        Inject a malicious removal error by removing a detection with a certain probability.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        # Remove a detection with a certain probability
        if random.random() < self.probability:
            if data['detections']:
                data['detections'].pop(random.randint(0, len(data['detections']) - 1))
        return data

    def inject_malicious_addition_error(self, data):
        """
        Inject a malicious addition error by adding a detection with a certain probability.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        # Add a detection with a certain probability
        if random.random() < self.probability:
            fake_detection = self.create_fake_detection()
            data['detections'].append(fake_detection)
        return data

    def inject_malicious_convoy_error(self, data):
        """
        Inject a malicious convoy error by consistently spawning a fake detection within a designated area.
        
        Args:
            data: The data to inject the error into.
        
        Returns:
            The data with the injected error.
        """
        # Consistently spawn a fake detection within a designated area
        if random.random() < self.probability:
            fake_detection = self.create_fake_detection()
            data['detections'].append(fake_detection)
        return data

    def create_fake_detection(self):
        """
        Create a fake detection.
        
        Returns:
            dict: A dictionary representing a fake detection.
        """
        # Example fake detection
        fake_detection = {
            'id': 'fake',
            'position': (random.uniform(0, 100), random.uniform(0, 100)),
            'velocity': (random.uniform(-10, 10), random.uniform(-10, 10)),
            'variance': random.uniform(0, 1)
        }
        return fake_detection