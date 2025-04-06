import traci
import ground_truth
import utils
import sensor
import localizer
from config import sensor_type, detector_type
from metrics import metrics, minimum_safety_envelope, time_to_collision, amota
import time
import sensor_fusion
import csv

# Start SUMO in server mode
sumoBinary = "sumo-gui"  # or "sumo" if you don't need the GUI
sumoCmd = [
    sumoBinary,
    "-c", "C:/home/rylens/cmr/maps/single/osm.sumocfg",
    "--step-length", "0.1"  # Set the step size to 0.1 seconds
]
traci.start(sumoCmd)

# Define some sensor detector combos
os1_pointpillars = sensor.Sensor(
                sensor_type=sensor_type.SensorType.OS1_128,
                detector_type=detector_type.DetectorType.POINT_PILLARS
            )
camera_yolo = sensor.Sensor(
                sensor_type=sensor_type.SensorType.CAMERA,
                detector_type=detector_type.DetectorType.YOLO
            )
os1_perfect = sensor.Sensor(
                sensor_type=sensor_type.SensorType.OS1_128,
                detector_type=detector_type.DetectorType.PERFECT
            )

# Example sensors/detector combos and their extrinsics
sensors = [
    os1_pointpillars,
    camera_yolo
]
sensors_extrinsics = [
    (0.0, 0.0, 0.0),  # (x, y, yaw) relative to the CAV
    (-1.0, 0.0, 0.0)
]
sensor_package_autoware = [(.5, sensors, sensors_extrinsics),
                  (.5, sensors, sensors_extrinsics)]

# Example localizer
# Coefficients for the lateral and longitudinal error polynomials
lateral_error_coefficients = [0.01, 0.001]  # Example coefficients 10cm at 0m/s, 20 cm at 10m/s
longitudinal_error_coefficients = [0.01, 0.001]  # Example coefficients 10cm at 0m/s, 20 cm at 10m/s

# Instantiate the Localizer class
lidar_slam = localizer.Localizer(lateral_error_coefficients, longitudinal_error_coefficients)

# Example sensors/detector combos and their extrinsics
sensors = [
    os1_perfect
]
sensors_extrinsics = [
    (0.0, 0.0, 0.0)
]
sensor_packages_perfect = [(1, sensors, sensors_extrinsics, lidar_slam)]

# Initialize VehicleProbabilityManager for CAVs
cav_manager = utils.VehicleProbabilityManager(probability=0.1, type="CAV", sumo_type="CAV_passenger", sensor_packages=sensor_packages_perfect)

# Initialize TrafficLightProbabilityManager for traffic lights
tfl_manager = utils.TrafficLightProbabilityManager(probability=1, traci_instance=traci, sensor_packages=sensor_packages_perfect)
cis_id_list = tfl_manager.get_tracked_traffic_light_ids()

# This is for doing the global sensor fusion
global_fusion = sensor_fusion.Fusion(sensor_fusion.max_id)

# Track the IDs of all polygons
polygon_ids = []

# Track total 
total_mse_violations_gt = 0
total_ttc_violations_gt = 0
total_mse_violations_sensor = 0
total_ttc_violations_sensor = 0

# Timing variables
total_simulation_step_time = 0
total_cis_detection_time = 0 
total_cis_detection_time = 0
total_global_fusion_time = 0
total_random_overhead_time = 0
iterations = 0

# Metrics accumulation
total_cav_metrics = {
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
total_global_metrics = {
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
total_active_cavs = 0

# Debug variables
visualize_ground_truth = False
visualize_cis_detections = False
visualize_cav_detections = False
visualize_global_fusion = False
do_global_fusion = True

# Simulation loop
step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    start_time = time.time()
    traci.simulationStep()
    simulation_step_time = time.time() - start_time

    print(f"Step: {step}")

    random_overhead_start_time = time.time()
    # Remove all polygons from the last frame
    for polygon_id in polygon_ids:
        traci.polygon.remove(polygon_id)
    polygon_ids = []

    # Get the IDs of all vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    # Update CAVs if step >= 1
    if step >= 1:
        cav_manager.update_vehicles(traci, vehicle_ids)
    else:
        update_vehicles_time = 0

    # Create ground truth data for all vehicles
    ground_truth_global = ground_truth.create_ground_truth_for_vehicle_from_list(traci, vehicle_ids)

    # Draw the bounding boxes in SUMO for debugging
    if visualize_ground_truth:
        for gt_obj in ground_truth_global:
            if gt_obj.draw_bounding_box_in_sumo(traci, layer=10):
                polygon_ids.append(f"bbox_{gt_obj.vehicle_id}")
            gt_obj.draw_position_vector_in_sumo(traci, layer=11)
            polygon_ids.append(f"vector_{gt_obj.vehicle_id}")

    # Get active CAVs and their ground truth objects
    cav_id_list = cav_manager.get_active_vehicle_ids()
    total_active_cavs += len(cav_id_list)
    
    # Simulation time now
    simulation_time_now = traci.simulation.getTime()

    # Initialize a dictionary to store unique detectable ground truth objects
    unique_detectable_ground_truth = {}
    random_overhead_time = time.time() - random_overhead_start_time

    # Iterate over all CISs to get their detection sets -----------------------------------
    cis_detection_start_time = time.time()
    for cis_id, cis_instance in zip(cis_id_list, cav_manager.get_active_vehicle_instances()):
        # Get the ground truth object for the CIS
        cis_gt_obj = ground_truth.create_ground_truth_for_traffic_light_by_id(traci, cis_id)
        fusion_result, detected_objects, detectable_ground_truth = cis_instance.create_detection_sets(cis_gt_obj, ground_truth_global, simulation_time_now)

        # Add detectable ground truth objects to the dictionary
        unique_detectable_ground_truth.update(detectable_ground_truth)

        if visualize_cis_detections:
            for each in detected_objects:
                if each.draw_bounding_box_in_sumo(traci, color=(255, 255, 0, 100), layer=9):
                    polygon_ids.append(f"bbox_{each.vehicle_id}")

        if do_global_fusion:
            global_fusion.processDetectionFrame(simulation_time_now, detected_objects, .3)
    cis_detection_time = time.time() - cis_detection_start_time
    
    # Iterate over all CAVs to get their detection sets -----------------------------------
    cav_detection_start_time = time.time()
    for cav_id, cav_instance in zip(cav_id_list, cav_manager.get_active_vehicle_instances()):
        # Get the ground truth object for the CAV
        cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci, cav_id)
        fusion_result, detected_objects, detectable_ground_truth = cav_instance.create_detection_sets(cav_gt_obj, ground_truth_global, simulation_time_now)

        # Add detectable ground truth objects to the dictionary
        unique_detectable_ground_truth.update(detectable_ground_truth)

        if visualize_cav_detections:
            for each in detected_objects:
                if each.draw_bounding_box_in_sumo(traci, color=(255, 255, 0, 255), layer=9):
                    polygon_ids.append(f"bbox_{each.vehicle_id}")

        # Calculate violations for the CAV
        cav_metrics = metrics.calculate_violations(cav_gt_obj, detectable_ground_truth, detected_objects, category="Aggressive")
        for key in total_cav_metrics:
            total_cav_metrics[key] += cav_metrics[key]

        if do_global_fusion:
            global_fusion.processDetectionFrame(simulation_time_now, detected_objects, .3)
    cav_detection_time = time.time() - cav_detection_start_time

    global_fusion_start_time = time.time()
    if do_global_fusion:
        global_fusion_result, global_detection_result, _, _ = global_fusion.fuseDetectionFrame(simulation_time_now)
        for each in global_detection_result:
            if each.draw_bounding_box_in_sumo(traci, color=(255, 100, 0, 255), layer=9):
                polygon_ids.append(f"bbox_{each.vehicle_id}")

        # Convert the dictionary of all detectable ground truth objects to a list
        all_detectable_ground_truth = list(unique_detectable_ground_truth.values())

        # Calculate AMOTA using unique detectable ground truth
        amota_score = amota.calculate_amota(global_detection_result, all_detectable_ground_truth)
        print(f"AMOTA Score: {amota_score:.4f}")

        # Add the AMOTA score to the global metrics here because we only want it one time
        total_global_metrics["amota"] += amota_score

        # Calculate violations for the global fusion for each CAV
        for cav_id in cav_id_list:
            # Get the ground truth object for the CAV
            cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci, cav_id)
            global_metrics = metrics.calculate_violations(cav_gt_obj, unique_detectable_ground_truth, global_detection_result, category="Aggressive", calc_amota=False)
            for key in total_global_metrics:
                total_global_metrics[key] += global_metrics[key]

        # TODO: Use global_detection_result for input to the motion planner

    global_fusion_time = time.time() - global_fusion_start_time

    # Update total times
    total_simulation_step_time += simulation_step_time
    total_cis_detection_time += cis_detection_time
    total_cis_detection_time += cav_detection_time
    total_global_fusion_time += global_fusion_time
    total_random_overhead_time += random_overhead_time
    iterations += 1

    # Print average times
    print(f"Average Simulation Step Time: {total_simulation_step_time / iterations:.4f} seconds")
    print(f"Average CIS Detection Time: {total_cis_detection_time / iterations:.4f} seconds")
    print(f"Average CAV Detection Time: {total_cis_detection_time / iterations:.4f} seconds")
    print(f"Average Global Fusion Time: {total_global_fusion_time / iterations:.4f} seconds")
    print(f"Average Random Overhead Time: {total_random_overhead_time / iterations:.4f} seconds")

    print(f"Total MSE Violations Ground Truth: {total_cav_metrics['total_mse_violations_gt']}")
    print(f"Total MSE Violations Onboard Sensing: {total_cav_metrics['total_mse_violations_sensor']}")
    print(f"Total TTC Violations Ground Truth: {total_cav_metrics['total_ttc_violations_gt']}")
    print(f"Total TTC Violations Onboard Sensing: {total_cav_metrics['total_ttc_violations_sensor']}")
    print(f"Total True Positives MSE: {total_cav_metrics['true_positives_mse']}")
    print(f"Total False Positives MSE: {total_cav_metrics['false_positives_mse']}")
    print(f"Total False Negatives MSE: {total_cav_metrics['false_negatives_mse']}")
    print(f"Total True Positives TTC: {total_cav_metrics['true_positives_ttc']}")
    print(f"Total False Positives TTC: {total_cav_metrics['false_positives_ttc']}")
    print(f"Total False Negatives TTC: {total_cav_metrics['false_negatives_ttc']}")
    print(f"Total AMOTA: {total_cav_metrics['amota']}")
    
    step += 1

# Write the totals to a CSV file
with open('metrics_totals.csv', 'w', newline='') as csvfile:
    fieldnames = ['Metric', 'Total']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'Metric': 'Total Steps', 'Total': iterations})
    writer.writerow({'Metric': 'Total Active CAVs', 'Total': total_active_cavs})
    for key, value in total_cav_metrics.items():
        writer.writerow({'Metric': f'CAV {key}', 'Total': value})
    for key, value in total_global_metrics.items():
        writer.writerow({'Metric': f'Global {key}', 'Total': value})

print(f"Total MSE Violations Ground Truth: {total_cav_metrics['total_mse_violations_gt']}")
print(f"Total MSE Violations Onboard Sensing: {total_cav_metrics['total_mse_violations_sensor']}")
print(f"Total TTC Violations Ground Truth: {total_cav_metrics['total_ttc_violations_gt']}")
print(f"Total TTC Violations Onboard Sensing: {total_cav_metrics['total_ttc_violations_sensor']}")
print(f"Total True Positives MSE: {total_cav_metrics['true_positives_mse']}")
print(f"Total False Positives MSE: {total_cav_metrics['false_positives_mse']}")
print(f"Total False Negatives MSE: {total_cav_metrics['false_negatives_mse']}")
print(f"Total True Positives TTC: {total_cav_metrics['true_positives_ttc']}")
print(f"Total False Positives TTC: {total_cav_metrics['false_positives_ttc']}")
print(f"Total False Negatives TTC: {total_cav_metrics['false_negatives_ttc']}")
print(f"Total AMOTA: {total_cav_metrics['amota']}")

traci.close()
