import math

def calculate_ttc(ego_vehicle, lead_vehicle):
    """
    Calculate the Time to Collision (TTC) between the ego vehicle and a lead vehicle.
    
    Args:
        ego_vehicle (GroundTruthObject): The ground truth object for the ego (subject) vehicle.
        lead_vehicle (GroundTruthObject): The ground truth object for the lead vehicle.
    
    Returns:
        float: The Time to Collision (TTC) in seconds.
    """
    # Calculate the relative position
    rel_position_x = lead_vehicle.centroid[0] - ego_vehicle.centroid[0]
    rel_position_y = lead_vehicle.centroid[1] - ego_vehicle.centroid[1]

    # Rotate the relative position to align with the ego vehicle's orientation
    ego_angle_rad = ego_vehicle.angle
    dlong = rel_position_x * math.cos(ego_angle_rad) + rel_position_y * math.sin(ego_angle_rad)

    # Calculate the relative velocity
    rel_velocity_x = lead_vehicle.velocity_vector[0] - ego_vehicle.velocity_vector[0]
    rel_velocity_y = lead_vehicle.velocity_vector[1] - ego_vehicle.velocity_vector[1]

    # Rotate the relative velocity to align with the ego vehicle's orientation
    vlong = rel_velocity_x * math.cos(ego_angle_rad) + rel_velocity_y * math.sin(ego_angle_rad)

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