import math

# Define the categories with their respective parameters
CATEGORIES = {
    "Aggressive": {"rho": 0.5, "a_accel_max": 4.10, "a_decel_min": 4.60, "a_decel_max": 8.00},
    "Conservative": {"rho": 1.9, "a_accel_max": 5.90, "a_decel_min": 4.10, "a_decel_max": 9.50},
    "Naturalistic": {"rho": 0.2, "a_accel_max": 1.80, "a_decel_min": 3.60, "a_decel_max": 6.10}
}

def calculate_mse_violations(ego_vehicle, ground_truth_objects, category="Aggressive"):
    """
    Calculate violations of the Minimum Safety Envelope (MSE) for a given ego vehicle using ground truth data.

    Args:
        ego_vehicle (GroundTruthObject): The ground truth object for the ego (subject) vehicle.
        ground_truth_objects (list): A list of GroundTruthObject instances representing other vehicles.
        category (str): The category of driving behavior ("Aggressive", "Conservative", "Naturalistic"). Defaults to "Aggressive".

    Returns:
        int: The number of MSE violations.
    """
    # Get the parameters for the selected category
    params = CATEGORIES[category]

    # Get the position and speed of the ego vehicle
    subject_position = ego_vehicle.centroid
    subject_speed = math.sqrt(ego_vehicle.velocity_vector[0]**2 + ego_vehicle.velocity_vector[1]**2)
    subject_velocity = ego_vehicle.velocity_vector

    violations = 0

    for gt_obj in ground_truth_objects:
        if gt_obj.vehicle_id == ego_vehicle.vehicle_id:
            continue

        # Calculate the longitudinal and lateral distances
        dlong = gt_obj.centroid[0] - subject_position[0]
        dlat = gt_obj.centroid[1] - subject_position[1]

        # Calculate the longitudinal and lateral velocities
        vlong = gt_obj.velocity_vector[0] - subject_velocity[0]
        vlat = gt_obj.velocity_vector[1] - subject_velocity[1]

        # Calculate the minimum safe distances
        min_safe_dlong = subject_speed * params["rho"] + 0.5 * params["a_decel_min"] * params["rho"]**2 + vlong * params["rho"]
        min_safe_dlat = 0.5 * params["a_decel_min"] * params["rho"]**2 + vlat * params["rho"]

        # Check for violations
        if dlong < min_safe_dlong or dlat < min_safe_dlat:
            violations += 1

    return violations

# # Unit tests
# import unittest

# class TestCalculateMSEViolations(unittest.TestCase):
#     def setUp(self):
#         self.ground_truth_objects = [
#             GroundTruthObject("veh1", "veh_passenger", (100, 100), (0, 0), [], 0, 1.8, 5.0),
#             GroundTruthObject("veh2", "veh_passenger", (200, 200), (0, 0), [], 0, 1.8, 5.0),
#             GroundTruthObject("veh3", "veh_passenger", (300, 300), (0, 0), [], 0, 1.8, 5.0)
#         ]
#         self.ego_vehicle = GroundTruthObject("veh1", "veh_passenger", (100, 100), (0, 0), [], 0, 1.8, 5.0)

#     def test_no_violations(self):
#         violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Aggressive")
#         self.assertEqual(violations, 0)

#     def test_one_violation(self):
#         self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
#         violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Aggressive")
#         self.assertEqual(violations, 1)

#     def test_multiple_violations(self):
#         self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
#         self.ground_truth_objects[2].centroid = (110, 100)  # Close to ego vehicle
#         violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Aggressive")
#         self.assertEqual(violations, 2)

#     def test_different_category(self):
#         self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
#         violations = calculate_mse_violations(self.ego_vehicle, self.ground_truth_objects, category="Conservative")
#         self.assertEqual(violations, 1)

# if __name__ == "__main__":
#     unittest.main()