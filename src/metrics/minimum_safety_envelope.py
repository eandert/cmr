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

    # Get the position, speed, and angle of the ego vehicle
    subject_position = ego_vehicle.centroid
    subject_speed = math.sqrt(ego_vehicle.velocity_vector[0]**2 + ego_vehicle.velocity_vector[1]**2)
    subject_angle_rad = ego_vehicle.angle
    subject_velocity = ego_vehicle.velocity_vector

    violations = 0

    for gt_obj in ground_truth_objects:
        if gt_obj.vehicle_id == ego_vehicle.vehicle_id:
            continue

        # Calculate the relative position
        rel_position_x = gt_obj.centroid[0] - subject_position[0]
        rel_position_y = gt_obj.centroid[1] - subject_position[1]

        # Rotate the relative position to align with the ego vehicle's orientation
        dlong = rel_position_x * math.cos(subject_angle_rad) + rel_position_y * math.sin(subject_angle_rad)
        dlat = -rel_position_x * math.sin(subject_angle_rad) + rel_position_y * math.cos(subject_angle_rad)

        # Calculate the relative velocity
        rel_velocity_x = gt_obj.velocity_vector[0] - subject_velocity[0]
        rel_velocity_y = gt_obj.velocity_vector[1] - subject_velocity[1]

        # Rotate the relative velocity to align with the ego vehicle's orientation
        vlong = rel_velocity_x * math.cos(subject_angle_rad) + rel_velocity_y * math.sin(subject_angle_rad)
        vlat = -rel_velocity_x * math.sin(subject_angle_rad) + rel_velocity_y * math.cos(subject_angle_rad)

        # Calculate the minimum safe distances
        min_safe_dlong = subject_speed * params["rho"] + 0.5 * params["a_decel_min"] * params["rho"]**2 + vlong * params["rho"]
        min_safe_dlat = 0.5 * params["a_decel_min"] * params["rho"]**2 + vlat * params["rho"]

        # Check for violations
        if dlong < min_safe_dlong or dlat < min_safe_dlat:
            violations += 1

    return violations