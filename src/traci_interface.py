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
from error import ErrorPackage, ErrorType
from config_templates import get_perfect_config, get_normal_sensors_config
import os
from concurrent.futures import ThreadPoolExecutor # Added for parallel processing

# Function to run the simulation
def run_simulation(config):
    # Instantiate ErrorPackage
    if config["error_type"] == "None":
        # Create a dummy ErrorPackage that injects no errors
        error_package_instance = ErrorPackage(ErrorType.SINGLE_SENSOR_EXTRINSICS, 0)
    else:
        error_package_instance = ErrorPackage(ErrorType(config["error_type"]), config["error_level"])

    # Start SUMO in server mode
    sumoBinary = "sumo"  # or "sumo-gui" if you need the GUI
    sumoCmd = [
        sumoBinary,
        "-c", os.path.join(os.path.dirname(__file__), "..", config["sumo_config_path"]), # Updated path
        "--step-length", str(config["step_length"])  # Set the step size from config
    ]
    traci.start(sumoCmd)

    # Define sensor detector combos based on config
    sensors = []
    for i in range(len(config["sensor_types"])):
        _sensor_type = getattr(sensor_type.SensorType, config["sensor_types"][i])
        _detector_type = getattr(detector_type.DetectorType, config["detector_types"][i])
        sensors.append(sensor.Sensor(
            sensor_type=_sensor_type,
            detector_type=_detector_type
        ))

    # Instantiate the Localizer class
    lidar_slam = localizer.Localizer(config["lateral_error_coefficients"], config["longitudinal_error_coefficients"], error_package_instance)

    # Prepare sensor packages for VehicleProbabilityManager and TrafficLightProbabilityManager
    sensor_packages_for_managers = [(1, sensors, config["sensor_extrinsics"], lidar_slam)]

    # Initialize VehicleProbabilityManager for CAVs
    cav_manager = utils.VehicleProbabilityManager(probability=config["cav_probability"], type="CAV", sumo_type="CAV_passenger", sensor_packages=sensor_packages_for_managers, error_package=error_package_instance)

    # Initialize TrafficLightProbabilityManager for traffic lights
    tfl_manager = utils.TrafficLightProbabilityManager(probability=config["tfl_probability"], traci_instance=traci, sensor_packages=sensor_packages_for_managers, error_package=error_package_instance)
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
    total_cav_detection_time = 0
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
    total_cis_metrics = { # New metrics dictionary for CIS
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
    cav_amota_frames_count = 0 # New counter for frames where CAV AMOTA is calculated
    cis_amota_frames_count = 0 # New counter for frames where CIS AMOTA is calculated
    global_amota_frames_count = 0 # New counter for frames where Global AMOTA is calculated

    # Debug variables
    visualize_ground_truth = config["visualize_ground_truth"]
    visualize_cis_detections = config["visualize_cis_detections"]
    visualize_cav_detections = config["visualize_cav_detections"]
    visualize_global_fusion = config["visualize_global_fusion"]
    do_global_fusion = config["do_global_fusion"]

    # Simulation loop
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        start_time = time.time()
        traci.simulationStep()
        simulation_step_time = time.time() - start_time

        if step % 100 == 0:
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

        # Create a cache dictionary for ground truth objects to avoid recreation
        # Key: vehicle_id, Value: GroundTruthObject
        ground_truth_cache = {gt.vehicle_id: gt for gt in ground_truth_global}

        # Build spatial index for fast range queries (optimization #2)
        ground_truth_spatial_index, _ = ground_truth.build_ground_truth_spatial_index(ground_truth_global)

        # Draw the bounding boxes in SUMO for debugging
        if visualize_ground_truth:
            for gt_obj in ground_truth_global:
                polygon_bbox_id = gt_obj.draw_bounding_box_in_sumo(traci, layer=10, polygon_id_prefix="gt")
                if polygon_bbox_id:
                    polygon_ids.append(polygon_bbox_id)
                
                # Position vector also needs a unique ID
                polygon_vector_id = gt_obj.draw_position_vector_in_sumo(traci, layer=11, polygon_id_prefix="gt")
                if polygon_vector_id:
                    polygon_ids.append(polygon_vector_id)

        # Get active CAVs and their ground truth objects
        cav_id_list = cav_manager.get_active_vehicle_ids()
        total_active_cavs += len(cav_id_list)
        
        # Simulation time now
        simulation_time_now = traci.simulation.getTime()

        # Initialize a dictionary to store unique detectable ground truth objects
        unique_detectable_ground_truth = {}
        random_overhead_time = time.time() - random_overhead_start_time

        # Use ThreadPoolExecutor for parallel processing of sensor packages
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # List to hold futures for CIS processing
            cis_futures = []
            # Cache to store CIS ground truth objects to avoid recreation
            cis_gt_cache = {}
            cis_detection_start_time = time.time()
            for cis_id, cis_instance in zip(cis_id_list, tfl_manager.get_tracked_traffic_light_instances()):
                # Get the ground truth object for the CIS (ego vehicle for the CIS)
                cis_gt_obj = ground_truth.create_ground_truth_for_traffic_light_by_id(traci, cis_id)
                # Cache it for reuse in metrics calculation
                cis_gt_cache[cis_id] = cis_gt_obj
                
                # Submit the task to the executor
                future = executor.submit(process_sensor_package, cis_id, cis_instance, cis_gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now, config["cleanup_time"])
                cis_futures.append(future)

            # Process results from CIS futures as they complete
            for future in cis_futures:
                fusion_result, detected_objects_for_metrics, detectable_ground_truth, cis_id = future.result()
                
                # Add detectable ground truth objects to the dictionary
                unique_detectable_ground_truth.update(detectable_ground_truth)

                if visualize_cis_detections:
                    for each in detected_objects_for_metrics:
                        polygon_bbox_id = each.draw_bounding_box_in_sumo(traci, color=(255, 255, 0, 100), layer=9, sensor_package_id=cis_id)
                        if polygon_bbox_id:
                            polygon_ids.append(polygon_bbox_id)

                # Calculate violations for the CIS - reuse cached ground truth object
                cis_gt_obj_for_metrics = cis_gt_cache.get(cis_id)
                if cis_gt_obj_for_metrics is None:
                    # Fallback to recreation if not in cache (shouldn't happen, but safe)
                    cis_gt_obj_for_metrics = ground_truth.create_ground_truth_for_traffic_light_by_id(traci, cis_id)
                cis_metrics = metrics.calculate_violations(cis_gt_obj_for_metrics, detectable_ground_truth, detected_objects_for_metrics, category="Aggressive")

                # Accumulate all metrics except 'amota'
                for key, value in cis_metrics.items():
                    if key != "amota":
                        total_cis_metrics[key] += value
                
                # Only accumulate amota if it's not zero and we have a valid ground truth to compare against
                if cis_metrics["amota"] != 0.0 and len(detectable_ground_truth) > 0:
                    total_cis_metrics["amota"] += cis_metrics["amota"]
                    cis_amota_frames_count += 1

                if do_global_fusion:
                    # The `detected_objects_for_metrics` here should be the aggregated and ego-filtered list from the CIS
                    global_fusion.processDetectionFrame(simulation_time_now, detected_objects_for_metrics, config["cleanup_time"])
            cis_detection_time = time.time() - cis_detection_start_time
            
            # Iterate over all CAVs to get their detection sets -----------------------------------
            cav_futures = []
            cav_detection_start_time = time.time()
            for cav_id, cav_instance in zip(cav_id_list, cav_manager.get_active_vehicle_instances()):
                # Get the ground truth object for the CAV - try cache first, then create if needed
                # Note: This object may be modified during detection (localization adjustments),
                # so we'll use the original from cache for metrics calculation
                cav_gt_obj = ground_truth_cache.get(cav_id)
                if cav_gt_obj is None:
                    # Fallback to creation if not in cache (shouldn't happen for active CAVs, but safe)
                    cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci, cav_id)
                
                # Submit the task to the executor
                future = executor.submit(process_sensor_package, cav_id, cav_instance, cav_gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now, config["cleanup_time"])
                cav_futures.append(future)

            # Process results from CAV futures as they complete
            for future in cav_futures:
                fusion_result, detected_objects_for_metrics, detectable_ground_truth, cav_id = future.result()

                # Add detectable ground truth objects to the dictionary
                unique_detectable_ground_truth.update(detectable_ground_truth)

                if visualize_cav_detections:
                    for each in detected_objects_for_metrics:
                        polygon_bbox_id = each.draw_bounding_box_in_sumo(traci, color=(255, 255, 0, 255), layer=9, sensor_package_id=cav_id)
                        if polygon_bbox_id:
                            polygon_ids.append(polygon_bbox_id)

                # Calculate violations for the CAV - use original ground truth from cache (not the modified one)
                # The ground truth object may have been modified during detection processing, so we use the original
                cav_gt_obj_for_metrics = ground_truth_cache.get(cav_id)
                if cav_gt_obj_for_metrics is None:
                    # Fallback to recreation if not in cache (shouldn't happen for active CAVs, but safe)
                    cav_gt_obj_for_metrics = ground_truth.create_ground_truth_for_vehicle_by_id(traci, cav_id)
                cav_metrics = metrics.calculate_violations(cav_gt_obj_for_metrics, detectable_ground_truth, detected_objects_for_metrics, category="Aggressive")

                # Accumulate all metrics except 'amota'
                for key, value in cav_metrics.items():
                    if key != "amota":
                        total_cav_metrics[key] += value
                
                # Only accumulate amota if it's not zero and we have a valid ground truth to compare against
                if cav_metrics["amota"] != 0.0 and len(detectable_ground_truth) > 0:
                    total_cav_metrics["amota"] += cav_metrics["amota"]
                    cav_amota_frames_count += 1

                if do_global_fusion:
                    global_fusion.processDetectionFrame(simulation_time_now, detected_objects_for_metrics, config["cleanup_time"])
            cav_detection_time = time.time() - cav_detection_start_time

        global_fusion_start_time = time.time()
        if do_global_fusion:
            global_fusion_result, global_detection_result, _, _ = global_fusion.fuseDetectionFrame(simulation_time_now)
            for each in global_detection_result:
                polygon_bbox_id = each.draw_bounding_box_in_sumo(traci, color=(255, 100, 0, 255), layer=9, sensor_package_id="global_fusion")
                if polygon_bbox_id:
                    polygon_ids.append(polygon_bbox_id)

            # Convert the dictionary of all detectable ground truth objects to a list
            all_detectable_ground_truth = list(unique_detectable_ground_truth.values())

            # Calculate AMOTA using unique detectable ground truth
            amota_score = amota.calculate_amota(global_detection_result, all_detectable_ground_truth)

            # Add the AMOTA score to the global metrics here because we only want it one time
            if amota_score != 0.0:
                total_global_metrics["amota"] += amota_score
                global_amota_frames_count += 1

            # Calculate violations for the global fusion for each CAV
            for cav_id in cav_id_list:
                # Get the ground truth object for the CAV - reuse from cache
                cav_gt_obj = ground_truth_cache.get(cav_id)
                if cav_gt_obj is None:
                    # Fallback to creation if not in cache (shouldn't happen for active CAVs, but safe)
                    cav_gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci, cav_id)
                global_metrics = metrics.calculate_violations(cav_gt_obj, unique_detectable_ground_truth, global_detection_result, category="Aggressive", calc_amota=False)
                for key in total_global_metrics:
                    total_global_metrics[key] += global_metrics[key]

            # TODO: Use global_detection_result for input to the motion planner

        global_fusion_time = time.time() - global_fusion_start_time

        # Update total times
        total_simulation_step_time += simulation_step_time
        total_cis_detection_time += cis_detection_time
        total_cav_detection_time += cav_detection_time
        total_global_fusion_time += global_fusion_time
        total_random_overhead_time += random_overhead_time
        iterations += 1

        if step % 100 == 0:
            # Print average times
            print(f"Average Simulation Step Time: {total_simulation_step_time / iterations:.4f} seconds")
            print(f"Average CIS Detection Time: {total_cis_detection_time / iterations:.4f} seconds")
            print(f"Average CAV Detection Time: {total_cav_detection_time / iterations:.4f} seconds")
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
            
            # Print Average AMOTA (for individual CAVs)
            if cav_amota_frames_count > 0:
                print(f"Average AMOTA: {total_cav_metrics['amota'] / cav_amota_frames_count:.4f}")
            else:
                print("Average AMOTA: N/A (no relevant ground truth for CAVs)")

            # Print CIS Metrics
            print(f"\nTotal CIS MSE Violations Ground Truth: {total_cis_metrics['total_mse_violations_gt']}")
            print(f"Total CIS MSE Violations Onboard Sensing: {total_cis_metrics['total_mse_violations_sensor']}")
            print(f"Total CIS TTC Violations Ground Truth: {total_cis_metrics['total_ttc_violations_gt']}")
            print(f"Total CIS TTC Violations Onboard Sensing: {total_cis_metrics['total_ttc_violations_sensor']}")
            print(f"Total CIS True Positives MSE: {total_cis_metrics['true_positives_mse']}")
            print(f"Total CIS False Positives MSE: {total_cis_metrics['false_positives_mse']}")
            print(f"Total CIS False Negatives MSE: {total_cis_metrics['false_negatives_mse']}")
            print(f"Total CIS True Positives TTC: {total_cis_metrics['true_positives_ttc']}")
            print(f"Total CIS False Positives TTC: {total_cis_metrics['false_positives_ttc']}")
            print(f"Total CIS False Negatives TTC: {total_cis_metrics['false_negatives_ttc']}")

            # Print Average AMOTA (for CIS)
            if cis_amota_frames_count > 0:
                print(f"Average CIS AMOTA: {total_cis_metrics['amota'] / cis_amota_frames_count:.4f}")
            else:
                print("Average CIS AMOTA: N/A (no relevant ground truth for CIS)")
            
            # Print Average Global AMOTA if global fusion is enabled
            if do_global_fusion:
                if global_amota_frames_count > 0:
                    print(f"Average Global AMOTA: {total_global_metrics['amota'] / global_amota_frames_count:.4f}")
                else:
                    print("Average Global AMOTA: N/A (no relevant ground truth)")

        step += 1

    traci.close()

    return {
        "iterations": iterations,
        "total_active_cavs": total_active_cavs,
        "total_cav_metrics": total_cav_metrics,
        "total_cis_metrics": total_cis_metrics, # Return CIS metrics
        "total_global_metrics": total_global_metrics
    }

# New helper function for parallel processing
def process_sensor_package(package_id, sensor_package_instance, gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now, cleanup_time):
    """
    Helper function to process a single sensor package (CAV or CIS) in a separate thread.
    """
    # Get the ground truth object for the ego vehicle (already created in main thread)
    # The gt_obj passed here is already specific to the ego vehicle.

    fusion_result, detected_objects_for_metrics, detectable_ground_truth = \
        sensor_package_instance.create_detection_sets(gt_obj, ground_truth_global, ground_truth_spatial_index, simulation_time_now)
    
    return fusion_result, detected_objects_for_metrics, detectable_ground_truth, package_id

# Define the error types and levels
error_types = [
    "None"
]
error_levels = [0]  # Levels from 0 to 10

# Open the CSV file for writing
with open('metrics_totals.csv', 'w', newline='') as csvfile:
    fieldnames = ['Run', 'Error Type', 'Error Level', 'Metric', 'Total']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Sweep through error types and levels
    run_id = 1
    # Use the perfect config template initially
    config = get_perfect_config()
    # Uncomment the line below to use the normal sensors config
    # config = get_normal_sensors_config()

    for i in range(1):
        print(f"Running simulation {run_id}/10 for error type '{config['error_type']}' at level {config['error_level']}...")
        result = run_simulation(config)

        # Write the results to the CSV file after each run
        writer.writerow({'Run': run_id, 'Error Type': config["error_type"], 'Error Level': config["error_level"], 'Metric': 'Total Steps', 'Total': result['iterations']})
        writer.writerow({'Run': run_id, 'Error Type': config["error_type"], 'Error Level': config["error_level"], 'Metric': 'Total Active CAVs', 'Total': result['total_active_cavs']})
        for key, value in result['total_cav_metrics'].items():
            writer.writerow({'Run': run_id, 'Error Type': config["error_type"], 'Error Level': config["error_level"], 'Metric': f'CAV {key}', 'Total': value})
        for key, value in result['total_cis_metrics'].items(): # Write CIS metrics
            writer.writerow({'Run': run_id, 'Error Type': config["error_type"], 'Error Level': config["error_level"], 'Metric': f'CIS {key}', 'Total': value})
        for key, value in result['total_global_metrics'].items():
            writer.writerow({'Run': run_id, 'Error Type': config["error_type"], 'Error Level': config["error_level"], 'Metric': f'Global {key}', 'Total': value})

        run_id += 1

print("Simulation completed and results written to metrics_totals.csv")