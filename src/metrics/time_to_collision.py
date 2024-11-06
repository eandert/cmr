import math

class GroundTruthObject:
    def __init__(self, vehicle_id, vehicle_type, position, velocity_vector, bounding_box, angle_rad, width, length):
        self.vehicle_id = vehicle_id
        self.type = vehicle_type
        self.bbox = bounding_box
        self.dimensions = [width, length]  # width, length
        self.centroid = [position[0], position[1]]  # 2D location
        self.rotation = angle_rad  # Store angle in radians
        self.velocity_vector = velocity_vector

def calculate_ttc(ego_vehicle, lead_vehicle):
    """
    Calculate the Time to Collision (TTC) between the ego vehicle and a lead vehicle.
    
    Args:
        ego_vehicle (GroundTruthObject): The ground truth object for the ego (subject) vehicle.
        lead_vehicle (GroundTruthObject): The ground truth object for the lead vehicle.
    
    Returns:
        float: The Time to Collision (TTC) in seconds.
    """
    # Calculate the longitudinal distance and relative velocity
    dlong = lead_vehicle.centroid[0] - ego_vehicle.centroid[0]
    vlong = ego_vehicle.velocity_vector[0] - lead_vehicle.velocity_vector[0]

    # Avoid division by zero
    if vlong == 0:
        return float('inf')

    # Calculate TTC
    ttc = dlong / vlong
    return ttc

def calculate_ttc_violations(ego_vehicle, ground_truth_objects, ttc_threshold=1.0):
    """
    Calculate violations of the Time to Collision (TTC) for a given ego vehicle using ground truth data.
    
    Args:
        ego_vehicle (GroundTruthObject): The ground truth object for the ego (subject) vehicle.
        ground_truth_objects (list): A list of GroundTruthObject instances representing other vehicles.
        ttc_threshold (float): The threshold for TTC violations. Defaults to 1.0 seconds.
    
    Returns:
        int: The number of TTC violations.
    """
    violations = 0

    for gt_obj in ground_truth_objects:
        if gt_obj.vehicle_id == ego_vehicle.vehicle_id:
            continue

        ttc = calculate_ttc(ego_vehicle, gt_obj)

        # Check for violations
        if 0 < ttc <= ttc_threshold:
            violations += 1

    return violations

# # Unit tests
# import unittest

# class TestTTCViolationFunctions(unittest.TestCase):
#     def setUp(self):
#         self.ground_truth_objects = [
#             GroundTruthObject("veh1", "veh_passenger", (100, 100), (10, 0), [], 0, 1.8, 5.0),
#             GroundTruthObject("veh2", "veh_passenger", (200, 100), (5, 0), [], 0, 1.8, 5.0),
#             GroundTruthObject("veh3", "veh_passenger", (300, 100), (0, 0), [], 0, 1.8, 5.0)
#         ]
#         self.ego_vehicle = GroundTruthObject("veh1", "veh_passenger", (100, 100), (10, 0), [], 0, 1.8, 5.0)

#     def test_no_violations(self):
#         violations = calculate_ttc_violations(self.ego_vehicle, self.ground_truth_objects, ttc_threshold=1.0)
#         self.assertEqual(violations, 0)

#     def test_one_violation(self):
#         self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
#         self.ground_truth_objects[1].velocity_vector = (5, 0)  # Slower than ego vehicle
#         violations = calculate_ttc_violations(self.ego_vehicle, self.ground_truth_objects, ttc_threshold=1.0)
#         self.assertEqual(violations, 1)

#     def test_multiple_violations(self):
#         self.ground_truth_objects[1].centroid = (105, 100)  # Close to ego vehicle
#         self.ground_truth_objects[1].velocity_vector = (5, 0)  # Slower than ego vehicle
#         self.ground_truth_objects[2].centroid = (110, 100)  # Close to ego vehicle
#         self.ground_truth_objects[2].velocity_vector = (0, 0)  # Stationary
#         violations = calculate_ttc_violations(self.ego_vehicle, self.ground_truth_objects, ttc_threshold=1.0)
#         self.assertEqual(violations, 2)

#     def test_different_threshold(self):
#         violations = calculate_ttc_violations(self.ego_vehicle, self.ground_truth_objects, ttc_threshold=2.0)
#         self.assertEqual(violations, 0)

# if __name__ == "__main__":
#     unittest.main()