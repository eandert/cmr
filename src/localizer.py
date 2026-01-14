import utils
import math
from error import ErrorPackage, ErrorType

class Localizer:
    """
    Class representing a localizer with its properties and error models.
    
    Attributes:
        lateral_error_polynomial (Polynomial): Polynomial for lateral error.
        longitudinal_error_polynomial (Polynomial): Polynomial for longitudinal error.
        error_package (ErrorPackage): The error package to apply localization errors.
    """
    
    def __init__(self, lateral_error_coefficients, longitudinal_error_coefficients, error_package):
        """
        Initialize the localizer with its error models.
        
        Args:
            lateral_error_coefficients (list): Coefficients for the lateral error polynomial.
            longitudinal_error_coefficients (list): Coefficients for the longitudinal error polynomial.
            error_package (ErrorPackage): The error package to apply localization errors.
        """
        self.lateral_error_polynomial = utils.Polynomial(lateral_error_coefficients)
        self.longitudinal_error_polynomial = utils.Polynomial(longitudinal_error_coefficients)
        self.error_package = error_package
        
    def lateral_error_at_velocity(self, velocity):
        """
        Get the lateral error at a given velocity.
        
        Args:
            velocity (float): The velocity of the target in meters per second.
        
        Returns:
            float: The lateral error.
        """
        return self.lateral_error_polynomial.evaluate(velocity)
        
    def longitudinal_error_at_velocity(self, velocity):
        """
        Get the longitudinal error at a given velocity.
        
        Args:
            velocity (float): The velocity of the target in meters per second.
        
        Returns:
            float: The longitudinal error.
        """
        return self.longitudinal_error_polynomial.evaluate(velocity)

    def get_localization_pose(self, ground_truth_x, ground_truth_y, ground_truth_yaw, velocity):
        """
        Get the localized pose with potential errors.

        Args:
            ground_truth_x (float): The ground truth x-coordinate.
            ground_truth_y (float): The ground truth y-coordinate.
            ground_truth_yaw (float): The ground truth yaw angle.
            velocity (float): The current velocity.

        Returns:
            tuple: The localized pose as (x, y, yaw) with injected errors.
        """
        lateral_error = self.lateral_error_at_velocity(velocity)
        longitudinal_error = self.longitudinal_error_at_velocity(velocity)

        # Adjust the position based on the errors and its angle
        adjusted_x = ground_truth_x + longitudinal_error * math.cos(ground_truth_yaw) - lateral_error * math.sin(ground_truth_yaw)
        adjusted_y = ground_truth_y + longitudinal_error * math.sin(ground_truth_yaw) + lateral_error * math.cos(ground_truth_yaw)
        adjusted_yaw = ground_truth_yaw # Yaw is not directly affected by lateral/longitudinal error here

        localization_data = {'x': adjusted_x, 'y': adjusted_y, 'yaw': adjusted_yaw}

        # Inject localization error if applicable
        if self.error_package.error_type == ErrorType.LOCALIZATION:
            localization_data = self.error_package.inject_error(localization_data, True)

        return localization_data['x'], localization_data['y'], localization_data['yaw']