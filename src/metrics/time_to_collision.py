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

def calculate_ttc_violations(ego_vehicle, ground_truth_objects, ttc_threshold=1.0, av_ids=None):
    """
    Calculate violations of the Time to Collision (TTC) for a given ego vehicle using ground truth data.
    
    Args:
        ego_vehicle (GroundTruthObject): The ground truth object for the ego (subject) vehicle.
        ground_truth_objects (list): A list of GroundTruthObject instances representing other vehicles.
        ttc_threshold (float): The threshold for TTC violations. Defaults to 1.0 seconds.
        av_ids (set, optional): Set of vehicle IDs that are AVs. If provided, returns split metrics.
    
    Returns:
        int or tuple: If av_ids is None, returns total violations count.
                      If av_ids is provided, returns (total, av_violations, non_av_violations).
    """
    violations = 0
    av_violations = 0
    non_av_violations = 0

    for gt_obj in ground_truth_objects:
        if gt_obj.vehicle_id == ego_vehicle.vehicle_id:
            continue

        ttc = calculate_ttc(ego_vehicle, gt_obj)

        # Check for violations
        if 0 < ttc <= ttc_threshold:
            violations += 1
            if av_ids is not None:
                if gt_obj.vehicle_id in av_ids:
                    av_violations += 1
                else:
                    non_av_violations += 1

    if av_ids is not None:
        return violations, av_violations, non_av_violations
    return violations