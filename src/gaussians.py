import numpy as np
import math

class BivariateGaussian:
    """
    This class is for creating and storing error terms of a sensor using
    mu and covariance so they can be fed into Kalman filters and printed.
    """
    
    def __init__(self, a, b, phi, mu=None, cov=None):
        """
        Initialize the Bivariate Gaussian with given parameters.
        
        Args:
            a (float): The variance along the x-axis.
            b (float): The variance along the y-axis.
            phi (float): The rotation angle of the ellipse in radians.
            mu (array-like, optional): The mean vector. Defaults to [0.0, 0.0].
            cov (array-like, optional): The covariance matrix. Defaults to None.
        """
        if cov is None:
            # Create the bivariate Gaussian matrix
            self.mu = np.array([0.0, 0.0])
            self.covariance = np.array([[a, 0], [0, b]])

            # RΣR^T to rotate the ellipse where Σ is the original covariance matrix
            rotate = np.array([[math.cos(phi), math.sin(phi)],
                               [-math.sin(phi), math.cos(phi)]])
            self.covariance = np.matmul(rotate, self.covariance)
            self.covariance = np.matmul(self.covariance, rotate.transpose())
        else:
            # Create the bivariate Gaussian matrix
            self.mu = mu
            self.covariance = cov

    def calc_self_radius_at_angle(self, angle, num_std_deviations):
        """
        Calculate the radius of the ellipse at a given angle.
        
        Args:
            angle (float): The angle at which to calculate the radius in radians.
            num_std_deviations (float): The number of standard deviations.
        
        Returns:
            float: The radius at the given angle.
        """
        a, b, phi = self.extract_error_ellipse_params(num_std_deviations)
        return self.calculate_radius_at_angle(a, b, phi, angle)

    def eigsorted(self):
        """
        Eigenvalues and eigenvectors of the covariance matrix.
        
        Returns:
            tuple: Sorted eigenvalues and eigenvectors.
        """
        vals, vecs = np.linalg.eigh(self.covariance)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def extract_error_ellipse_params(self, num_std_deviations):
        """
        Extract the parameters of the error ellipse from the bivariate Gaussian.
        
        Args:
            num_std_deviations (float): The number of standard deviations.
        
        Returns:
            tuple: The semi-major axis, semi-minor axis, and rotation angle of the ellipse.
        """
        if np.all((self.covariance == 0.0)):
            return 0, 0, 0.0

        vals, vecs = self.eigsorted()
        phi = np.arctan2(*vecs[:, 0][::-1])

        a = num_std_deviations * vals[0]
        b = num_std_deviations * vals[1]

        return a, b, phi

    def calc_xy_components(self):
        """
        Calculate the x and y components of the ellipse.
        
        Returns:
            tuple: The x and y components.
        """
        x_comp = self.calc_self_radius_at_angle(math.radians(0), 1.0)
        y_comp = self.calc_self_radius_at_angle(math.radians(90), 1.0)
        return x_comp, y_comp

    def union_bivariate_gaussians(self, gaussian_b):
        """
        Combine two bivariate Gaussians by adding their covariances.
        
        Args:
            gaussian_b (BivariateGaussian): The second bivariate Gaussian.
        """
        self.covariance = np.add(self.covariance.transpose(), gaussian_b.covariance.transpose()).transpose()

    @staticmethod
    def intersection_bivariate_gaussians_covariance(covariance_gaussian_a, covariance_gaussian_b):
        """
        Combine two bivariate Gaussians by subtracting their covariances.
        
        Args:
            covariance_gaussian_a (array-like): The covariance matrix of the first Gaussian.
            covariance_gaussian_b (array-like): The covariance matrix of the second Gaussian.
        
        Returns:
            array-like: The resulting covariance matrix.
        """
        covariance_new = np.subtract(covariance_gaussian_a.transpose(), covariance_gaussian_b.transpose()).transpose()
        return covariance_new

    @staticmethod
    def calculate_radius_at_angle(a, b, phi, measurement_angle):
        """
        Calculate the radius of the ellipse at a given angle.
        
        Args:
            a (float): The semi-major axis of the ellipse.
            b (float): The semi-minor axis of the ellipse.
            phi (float): The rotation angle of the ellipse in radians.
            measurement_angle (float): The angle at which to calculate the radius in radians.
        
        Returns:
            float: The radius at the given angle.
        """
        epsilon = 1e-10  # Small value to avoid zero denominator
        denominator = math.sqrt(a**2 * math.sin(phi - measurement_angle)**2 + b**2 * math.cos(phi - measurement_angle)**2 + epsilon)
        if denominator == 0.0:
            print("Warning: calculate_radius_at_angle denom 0! - check localizer definitions")
            return 0.0
        else:
            return (a * b) / denominator