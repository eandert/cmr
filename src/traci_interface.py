import traci
import ground_truth
import utils
import sensor
import cav
from config import sensor_type, detector_type
from metrics import minimum_safety_envelope, time_to_collision

# Start SUMO in server mode
sumoBinary = "sumo-gui"  # or "sumo" if you don't need the GUI
sumoCmd = [sumoBinary, "-c", "C:/Users/edwar/Documents/GitHub/cmr/maps/tempe_2x3/osm.sumocfg"]
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

# Example sensors/detector combos and their extrinsics
sensors = [
    os1_pointpillars,
    camera_yolo
]
sensors_extrinsics = [
    (0.0, 0.0, 0.0),  # (x, y, yaw) relative to the CAV
    (-1.0, 0.0, 0.0)
]
sensor_package = [(.5, sensors, sensors_extrinsics),
                  (.5, sensors, sensors_extrinsics)]

# Initialize VehicleProbabilityManager for CAVs
cav_manager = utils.VehicleProbabilityManager(probability=0.1, type="CAV", sumo_type="CAV_passenger", sensor_packages=sensor_package)

# Track the IDs of all polygons
polygon_ids = []

# Track total 
total_mse_violations_gt = 0
total_ttc_violations_gt = 0
total_mse_violations_sensor = 0
total_ttc_violations_sensor = 0

# Simulation loop
step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    print(f"Step: {step}")

    # Remove all polygons from the last frame
    for polygon_id in polygon_ids:
        traci.polygon.remove(polygon_id)
    polygon_ids = []

    # Get the IDs of all vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    # Update CAVs if step >= 1
    if step >= 1:
        cav_manager.update_vehicles(traci, vehicle_ids)

    # Create ground truth data for all vehicles
    ground_truth_objects = ground_truth.create_ground_truth_from_list(traci, vehicle_ids)

    # Draw the bounding boxes in SUMO for debugging
    polygon_ids = []
    for gt_obj in ground_truth_objects:
        gt_obj.draw_bounding_box_in_sumo(traci, layer=10)
        # gt_obj.draw_position_vector_in_sumo(traci, layer=11)
        polygon_ids.append(f"bbox_{gt_obj.vehicle_id}")
        # polygon_ids.append(f"vector_{gt_obj.vehicle_id}")

    # Get active CAVs
    cav_id_list = cav_manager.get_active_vehicle_ids()
    ground_truth_objects = [ground_truth.create_ground_truth_by_id(traci, veh_id) for veh_id in vehicle_ids]
    time = traci.simulation.getTime()
    
    for cav_id, cav_instance in zip(cav_id_list, cav_manager.get_active_vehicle_instances()):
        # Get the ground truth object for the CAV
        cav_gt_obj = ground_truth.create_ground_truth_by_id(traci, cav_id)
        fusion_result, viz_result = cav_instance.create_detection_sets(cav_gt_obj, ground_truth_objects, time)

        for each in viz_result:
            each.draw_bounding_box_in_sumo(traci, layer=12)
            polygon_ids.append(f"bbox_{each.vehicle_id}")

        # Unomment for debug
        # cav_instance.draw_detected_objects(traci, polygon_ids)

        # TODO(eandert): Compare the filtered grount truth objects to the detected objects using AMOTA

        # # Calculate ground truth MSE violations for the ego vehicle
        # violations = minimum_safety_envelope.calculate_mse_violations(cav_gt_obj, ground_truth_objects_filtered, category="Aggressive")
        # total_mse_violations_gt += violations
        # # print(f"MSE Violations for vehicle {cav_gt_obj.vehicle_id}: {violations}")

        # # Calculate sensor based MSE violations for the ego vehicle
        # violations = minimum_safety_envelope.calculate_mse_violations(cav_gt_obj, detected_objects, category="Aggressive")
        # total_mse_violations_sensor += violations

        # # Calculate the time to collision for the ego vehicle with gt
        # ttc_violations = time_to_collision.calculate_ttc_violations(cav_gt_obj, ground_truth_objects_filtered)
        # total_ttc_violations_gt += ttc_violations
        # # print(f"TTC Violations for vehicle {cav_gt_obj.vehicle_id}: {ttc_violations}")

        # # Calculate the time to collision for the ego vehicle with sensor
        # ttc_violations = time_to_collision.calculate_ttc_violations(cav_gt_obj, detected_objects)
        # total_ttc_violations_sensor += ttc_violations

    print(f"Total MSE Violations GT: {total_mse_violations_gt}")
    print(f"Total MSE Violations Sensor: {total_mse_violations_sensor}")
    print(f"Total TTC Violations: {total_ttc_violations_gt}")
    print(f"Total TTC Violations Sensor: {total_ttc_violations_sensor}")
    
    step += 1

print(f"Total MSE Violations GT: {total_mse_violations_gt}")
print(f"Total MSE Violations Sensor: {total_mse_violations_sensor}")
print(f"Total TTC Violations: {total_ttc_violations_gt}")
print(f"Total TTC Violations Sensor: {total_ttc_violations_sensor}")

traci.close()