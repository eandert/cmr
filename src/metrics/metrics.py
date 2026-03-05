# FILE: metrics.py

from metrics import minimum_safety_envelope, time_to_collision, amota

def calculate_violations(cav_gt_obj, detectable_ground_truth, detected_objects, category="Aggressive", calc_amota=True, av_ids=None):
    """
    Calculate various violations for the ego vehicle, including true positives, false positives, false negatives, and AMOTA.

    Args:
        cav_gt_obj (GroundTruthObject): The ground truth object for the ego vehicle.
        detectable_ground_truth (dict): A dictionary of detectable ground truth objects.
        detected_objects (list): A list of detected objects.
        category (str): The category of driving behavior ("Aggressive", "Conservative", "Naturalistic"). Defaults to "Aggressive".
        calc_amota (bool): Whether to calculate AMOTA score. Defaults to True.
        av_ids (set, optional): Set of vehicle IDs that are AVs. If provided, returns split metrics for AV vs non-AV.

    Returns:
        dict: A dictionary containing the total MSE violations for ground truth and sensor, TTC violations for ground truth and sensor,
              true positives, false positives, false negatives for both MSE and TTC metrics, and AMOTA score.
              If av_ids is provided, also includes _av and _non_av suffixed versions of violation metrics.
    """
    violations = {
        "total_mse_violations_gt": 0,
        "total_mse_violations_sensor": 0,
        "total_ttc_violations_gt": 0,
        "total_ttc_violations_sensor": 0,
        "true_positives_mse": 0,
        "false_positives_mse": 0,
        "false_negatives_mse": 0,
        "true_positives_ttc": 0,
        "false_positives_ttc": 0,
        "false_negatives_ttc": 0,
        "amota": 0.0
    }
    
    # Add AV/non-AV split metrics if av_ids is provided
    if av_ids is not None:
        violations.update({
            "total_mse_violations_gt_av": 0,
            "total_mse_violations_gt_non_av": 0,
            "total_mse_violations_sensor_av": 0,
            "total_mse_violations_sensor_non_av": 0,
            "total_ttc_violations_gt_av": 0,
            "total_ttc_violations_gt_non_av": 0,
            "total_ttc_violations_sensor_av": 0,
            "total_ttc_violations_sensor_non_av": 0,
        })

    # Calculate ground truth MSE violations for the ego vehicle
    if av_ids is not None:
        mse_violations_gt, mse_gt_av, mse_gt_non_av = minimum_safety_envelope.calculate_mse_violations(
            cav_gt_obj, list(detectable_ground_truth.values()), category, av_ids)
        violations["total_mse_violations_gt"] += mse_violations_gt
        violations["total_mse_violations_gt_av"] += mse_gt_av
        violations["total_mse_violations_gt_non_av"] += mse_gt_non_av
    else:
        mse_violations_gt = minimum_safety_envelope.calculate_mse_violations(
            cav_gt_obj, list(detectable_ground_truth.values()), category)
        violations["total_mse_violations_gt"] += mse_violations_gt

    # Calculate sensor based MSE violations for the ego vehicle
    if av_ids is not None:
        mse_violations_sensor, mse_sensor_av, mse_sensor_non_av = minimum_safety_envelope.calculate_mse_violations(
            cav_gt_obj, detected_objects, category, av_ids)
        violations["total_mse_violations_sensor"] += mse_violations_sensor
        violations["total_mse_violations_sensor_av"] += mse_sensor_av
        violations["total_mse_violations_sensor_non_av"] += mse_sensor_non_av
    else:
        mse_violations_sensor = minimum_safety_envelope.calculate_mse_violations(cav_gt_obj, detected_objects, category)
        violations["total_mse_violations_sensor"] += mse_violations_sensor

    # Calculate the time to collision for the ego vehicle with gt
    if av_ids is not None:
        ttc_violations_gt, ttc_gt_av, ttc_gt_non_av = time_to_collision.calculate_ttc_violations(
            cav_gt_obj, list(detectable_ground_truth.values()), av_ids=av_ids)
        violations["total_ttc_violations_gt"] += ttc_violations_gt
        violations["total_ttc_violations_gt_av"] += ttc_gt_av
        violations["total_ttc_violations_gt_non_av"] += ttc_gt_non_av
    else:
        ttc_violations_gt = time_to_collision.calculate_ttc_violations(cav_gt_obj, list(detectable_ground_truth.values()))
        violations["total_ttc_violations_gt"] += ttc_violations_gt

    # Calculate the time to collision for the ego vehicle with sensor
    if av_ids is not None:
        ttc_violations_sensor, ttc_sensor_av, ttc_sensor_non_av = time_to_collision.calculate_ttc_violations(
            cav_gt_obj, detected_objects, av_ids=av_ids)
        violations["total_ttc_violations_sensor"] += ttc_violations_sensor
        violations["total_ttc_violations_sensor_av"] += ttc_sensor_av
        violations["total_ttc_violations_sensor_non_av"] += ttc_sensor_non_av
    else:
        ttc_violations_sensor = time_to_collision.calculate_ttc_violations(cav_gt_obj, detected_objects)
        violations["total_ttc_violations_sensor"] += ttc_violations_sensor

    # Estimate true positives, false positives, and false negatives for MSE
    num_detected_mse = mse_violations_sensor
    num_ground_truth_mse = mse_violations_gt
    estimated_tp_mse = min(num_detected_mse, num_ground_truth_mse)
    estimated_fp_mse = num_detected_mse - estimated_tp_mse
    estimated_fn_mse = num_ground_truth_mse - estimated_tp_mse

    # Ensure values are not less than zero
    estimated_tp_mse = max(0, estimated_tp_mse)
    estimated_fp_mse = max(0, estimated_fp_mse)
    estimated_fn_mse = max(0, estimated_fn_mse)

    violations["true_positives_mse"] += estimated_tp_mse
    violations["false_positives_mse"] += estimated_fp_mse
    violations["false_negatives_mse"] += estimated_fn_mse

    # Estimate true positives, false positives, and false negatives for TTC
    num_detected_ttc = ttc_violations_sensor
    num_ground_truth_ttc = ttc_violations_gt
    estimated_tp_ttc = min(num_detected_ttc, num_ground_truth_ttc)
    estimated_fp_ttc = num_detected_ttc - estimated_tp_ttc
    estimated_fn_ttc = num_ground_truth_ttc - estimated_tp_ttc

    # Ensure values are not less than zero
    estimated_tp_ttc = max(0, estimated_tp_ttc)
    estimated_fp_ttc = max(0, estimated_fp_ttc)
    estimated_fn_ttc = max(0, estimated_fn_ttc)

    violations["true_positives_ttc"] += estimated_tp_ttc
    violations["false_positives_ttc"] += estimated_fp_ttc
    violations["false_negatives_ttc"] += estimated_fn_ttc

    # Calculate AMOTA score
    if calc_amota:
        violations["amota"] = amota.calculate_amota(detected_objects, list(detectable_ground_truth.values()))
        violations["amotp"] = amota.calculate_amotp(detected_objects, list(detectable_ground_truth.values()))
    else:
        violations["amota"] = 0.0
        violations["amotp"] = 0.0

    return violations