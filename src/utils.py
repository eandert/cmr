import math
import random
import cav
import numpy as np

class VehicleProbabilityManager:
    """
    Manages the probability of vehicles being classified as a specific type (e.g., CAV).
    """

    def __init__(self, probability, type, sumo_type, sensor_packages=None):
        """
        Initializes the VehicleProbabilityManager.

        Args:
            probability (float): The probability of a vehicle being classified as the specified type.
            type (str): The type of vehicle (e.g., "CAV").
            sumo_type (str): The SUMO vehicle type to set (e.g., "CAV_passenger").
            sensor_packages (list): A list of tuples containing the probability, sensors, and sensors' extrinsics.
        """
        self.probability = probability
        self.vehicle_list = []
        self.total_vehicles = 0
        self.checked_vehicle_id_list = []
        self.type = type
        self.sumo_type = sumo_type
        self.vehicle_instances = {}
        self.sensor_packages = sensor_packages if sensor_packages else []

    def update_vehicles(self, traci_instance, vehicle_id_list):
        """
        Updates the list of vehicles and changes the type of new vehicles based on the specified probability.

        Args:
            traci_instance: The TraCI instance to use for interacting with the simulation.
            vehicle_id_list (list): A list of vehicle IDs currently in the simulation.
        """
        # Get the IDs of all vehicles that have appeared in the simulation since the last step
        new_vehicle_id_list = [x for x in vehicle_id_list if x not in self.checked_vehicle_id_list]

        for vehicle_id in new_vehicle_id_list:
            randomnum = random.random()
            if self.probability != 0 and randomnum <= self.probability:
                try:
                    print(f"{self.type} added: ", vehicle_id)
                    self.total_vehicles += 1
                    self.vehicle_list.append(vehicle_id)
                    # Change the vehicle type to the specified SUMO type
                    traci_instance.vehicle.setType(vehicle_id, self.sumo_type)
                    # Create a new instance of the vehicle class with a selected sensor package
                    if self.sensor_packages:
                        sensor_package = self.select_sensor_package()
                        print(f"Selected sensor package for {vehicle_id}: {sensor_package}")
                        self.vehicle_instances[vehicle_id] = cav.CAV(vehicle_id, sensor_package[0], sensor_package[1])
                except Exception as e:
                    print(f"ERROR: Couldn't add {self.type}: ", e)

        self.checked_vehicle_id_list += new_vehicle_id_list

        # Remove vehicles that are no longer in the simulation
        removed_vehicle_id_list = [x for x in self.checked_vehicle_id_list if x not in vehicle_id_list]
        for vehicle_id in removed_vehicle_id_list:
            if vehicle_id in self.vehicle_list:
                self.vehicle_list.remove(vehicle_id)
                self.total_vehicles -= 1
                # Delete the vehicle instance
                if vehicle_id in self.vehicle_instances:
                    del self.vehicle_instances[vehicle_id]

        self.checked_vehicle_id_list = list(vehicle_id_list)

    def select_sensor_package(self):
        """
        Selects a sensor package based on the distribution within the sensor_packages tuple.

        Returns:
            tuple: The selected sensor package (sensors, sensors_extrinsics).
        """
        total_probability = sum(package[0] for package in self.sensor_packages)
        randomnum = random.uniform(0, total_probability)
        cumulative_probability = 0.0

        for probability, sensors, sensors_extrinsics in self.sensor_packages:
            cumulative_probability += probability
            if randomnum <= cumulative_probability:
                return sensors, sensors_extrinsics

        return self.sensor_packages[-1][1], self.sensor_packages[-1][2]  # Fallback to the last package

    def get_active_vehicle_ids(self):
        """
        Returns the list of vehicle instances of active vehicles that are of the specified type.

        Returns:
            list: A list of active vehicle instances that are of the specified type.
        """
        return list(self.vehicle_instances.keys())
    
    def get_active_vehicle_instances(self):
        """
        Returns the list of vehicle instances of active vehicles that are of the specified type.

        Returns:
            list: A list of active vehicle instances that are of the specified type.
        """
        return list(self.vehicle_instances.values())

class Polynomial:
    """
    Represents a polynomial with a list of coefficients.
    """

    def __init__(self, coefficients):
        """
        Initializes the Polynomial.

        Args:
            coefficients (list): A list of coefficients for the polynomial.
        """
        self.coefficients = coefficients  # List of coefficients

    def __str__(self):
        """
        Returns a string representation of the polynomial.

        Returns:
            str: The string representation of the polynomial.
        """
        terms = []
        for power, coeff in enumerate(self.coefficients):
            if coeff != 0:
                terms.append(f"{coeff}*x^{power}")
        return " + ".join(terms)

    def evaluate(self, x):
        """
        Evaluates the polynomial at a given value of x.

        Args:
            x (float): The value at which to evaluate the polynomial.

        Returns:
            float: The result of the polynomial evaluation.
        """
        result = 0
        for power, coeff in enumerate(self.coefficients):
            result += coeff * (x ** power)
        return result
    
def rotate_point(x, y, cx, cy, angle):
    """
    Rotate a point around a center by a given angle.
    
    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        cx (float): The x-coordinate of the center.
        cy (float): The y-coordinate of the center.
        angle (float): The angle in radians.
    
    Returns:
        tuple: The rotated point (x, y).
    """
    # Translate point to origin
    x -= cx
    y -= cy
    # Rotate point in the opposite direction
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle
    # Translate point back
    x_new += cx
    y_new += cy
    return x_new, y_new

''' Kalman filter prediction equasion '''
def kalman_prediction(X_hat_t_1, P_t_1, F_t, B_t, U_t, Q_t):
    X_hat_t = F_t.dot(X_hat_t_1) + (B_t.dot(U_t).reshape(B_t.shape[0], -1))
    P_t = F_t.dot(P_t_1).dot(F_t.transpose()) + Q_t
    return X_hat_t, P_t

''' Kalman filter update equasion '''
def kalman_update(X_hat_t, P_t, Z_t, R_t, H_t):
    K_prime = P_t.dot(H_t.transpose()).dot(kalman_inverse(H_t.dot(P_t).dot(H_t.transpose()) + R_t))
    X_t = X_hat_t + K_prime.dot(Z_t - H_t.dot(X_hat_t))
    P_t = P_t - K_prime.dot(H_t).dot(P_t)
    return X_t, P_t

''' Inverse function that is better than the default numpy one '''
def kalman_inverse(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")
    i = np.eye(a, a)
    return np.linalg.lstsq(m, i, rcond=None)[0]

def ellipsify(covariance, num_std_deviations = 3.0):
    # Eigenvalue and eigenvector computations
    #print ( covariance )
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Use the eigenvalue to figure out which direction is larger
    phi = np.arctan2(*vecs[:, 0][::-1])

    # A and B are radii
    #a, b = num_std_deviations * np.sqrt(vals)

    if not np.any(np.isnan(vals)) and np.all(np.isfinite(vals)):
        a, b = num_std_deviations * np.sqrt(vals)
    elif not np.isnan(vals[0]) and np.isfinite(vals[0]):
        a = num_std_deviations * np.sqrt(vals[0])
        b = 0.0
    elif not np.isnan(vals[1]) and np.isfinite(vals[1]):
        b = num_std_deviations * np.sqrt(vals[1])
        a = 0.0
    else:
        a = 0.0
        b = 0.0

    return a, b, phi