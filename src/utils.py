import math
import random

class VehicleProbabilityManager:
    """
    Manages the probability of vehicles being classified as a specific type (e.g., CAV).
    """

    def __init__(self, probability, type, sumo_type):
        """
        Initializes the VehicleProbabilityManager.

        Args:
            probability (float): The probability of a vehicle being classified as the specified type.
            type (str): The type of vehicle (e.g., "CAV").
            sumo_type (str): The SUMO vehicle type to set (e.g., "CAV_passenger").
        """
        self.probability = probability
        self.vehicle_list = []
        self.total_vehicles = 0
        self.checked_vehicle_id_list = []
        self.type = type
        self.sumo_type = sumo_type

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
                except Exception as e:
                    print(f"ERROR: Couldn't add {self.type}: ", e)

        self.checked_vehicle_id_list += new_vehicle_id_list

    def get_active_vehicles(self, vehicle_ids):
        """
        Returns the list of active vehicles that are of the specified type.

        Args:
            vehicle_ids (list): A list of vehicle IDs currently in the simulation.

        Returns:
            list: A list of active vehicle IDs that are of the specified type.
        """
        return list(set(self.vehicle_list) & set(vehicle_ids))

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