import utils

class Localizer:
    """
    Class representing a localizer with its properties and error models.
    
    Attributes:
        lateral_error_polynomial (Polynomial): Polynomial for lateral error.
        longitudinal_error_polynomial (Polynomial): Polynomial for longitudinal error.
    """
    
    def __init__(self, lateral_error_coefficients, longitudinal_error_coefficients):
        """
        Initialize the localizer with its error models.
        
        Args:
            lateral_error_coefficients (list): Coefficients for the lateral error polynomial.
            longitudinal_error_coefficients (list): Coefficients for the longitudinal error polynomial.
        """
        self.lateral_error_polynomial = utils.Polynomial(lateral_error_coefficients)
        self.longitudinal_error_polynomial = utils.Polynomial(longitudinal_error_coefficients)
        
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