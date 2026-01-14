import math
import utils
import sensor
import ground_truth
import sensor_fusion
import random
import numpy
from error import ErrorPackage, ErrorType

class SensorPackage:
    """
    Class representing a Connected Sensor either a CAV or CIS.
    
    Attributes:
        sensor_package_id (str): The ID of the SensorPackage.
        ground_truth_obj (GroundTruthObject): The ground truth object for the SensorPackage.
        sensors (list): A list of sensor objects for the SensorPackage.
        sensors_extrinsics (list): A list of extrinsic parameters for each sensor.
        sensor_detection_sets (list): A list of detection sets for each sensor.
        detection_set_timestamp (float): The timestamp of the detection set.
        localizer (Localizer): The localizer object for the SensorPackage.
    """
    
    def __init__(self, sensor_package_id, sensors, sensors_extrinsics, localizer, error_package):
        """
        Initialize the SensorPackage with its ID, sensors, sensors' extrinsics, and localizer.
        
        Args:
            sensor_package_id (str): The ID of the SensorPackage.
            sensors (list): A list of sensor objects for the SensorPackage.
            sensors_extrinsics (list): A list of extrinsic parameters for each sensor.
            localizer (Localizer): The localizer object for the SensorPackage.
            error_package (ErrorPackage): The error package to apply errors.
        """
        self.sensor_package_id = sensor_package_id
        self.ground_truth_obj = None
        self.sensors = []
        self.sensors_extrinsics = sensors_extrinsics
        self.sensor_detection_sets = []
        self.detection_set_timestamp = None
        self.integer_id = utils.extract_id(sensor_package_id)
        self.sensor_fusion = sensor_fusion.Fusion(self.integer_id)
        self.localizer = localizer
        self.error = error_package
        self.roll1 = random.random()
        self.roll2 = random.random()
        self.error_x = None
        self.error_y = None

        # Assign initial extrinsics to each sensor object
        for i, sensor_obj in enumerate(sensors):
            sensor_obj.x_extrinsics_original = sensors_extrinsics[i][0]
            sensor_obj.y_extrinsics_original = sensors_extrinsics[i][1]
            sensor_obj.angle_extrinsics_original = sensors_extrinsics[i][2]
            self.sensors.append(sensor_obj)

        # Apply single or multi-sensor extrinsics errors
        if self.error.error_type == ErrorType.SINGLE_SENSOR_EXTRINSICS and random.random() < self.error.probability:
            # Apply error to a single random sensor
            target_sensor_idx = random.randint(0, len(self.sensors) - 1)
            self.sensors[target_sensor_idx] = self.error.inject_single_sensor_extrinsics_error(self.sensors[target_sensor_idx])
        elif self.error.error_type == ErrorType.MULTI_SENSOR_EXTRINSICS and random.random() < self.error.probability:
            # Apply error to all sensors
            self.sensors = self.error.inject_multi_sensor_extrinsics_error(self.sensors)
        
    def set_sensor_poses(self):
        """
        Set the sensor poses for the SensorPackage based on the SensorPackage's pose and the sensors' extrinsics.
        
        Returns:
            list: A list of sensor poses as (x, y, yaw).
        """
        
        cav_x, cav_y, cav_yaw = self.ground_truth_obj.centroid[0], self.ground_truth_obj.centroid[1], self.ground_truth_obj.angle
        sensor_poses = []

        for sensor_obj in self.sensors:
            # Use the sensor's original extrinsics combined with any injected errors
            sensor_x_rel = sensor_obj.x_extrinsics_original + sensor_obj.x_offset_error
            sensor_y_rel = sensor_obj.y_extrinsics_original + sensor_obj.y_offset_error
            sensor_yaw_rel = sensor_obj.angle_extrinsics_original + sensor_obj.angle_offset_error

            # Calculate the sensor's absolute position
            sensor_x = cav_x + sensor_x_rel * math.cos(cav_yaw) - sensor_y_rel * math.sin(cav_yaw)
            sensor_y = cav_y + sensor_x_rel * math.sin(cav_yaw) + sensor_y_rel * math.cos(cav_yaw)
            sensor_yaw = cav_yaw + sensor_yaw_rel

            sensor_poses.append((sensor_x, sensor_y, sensor_yaw))
            # print(f"  SensorPackage {self.sensor_package_id} Sensor {sensor_obj.sensor_type_id} actual pose: (x={sensor_x:.2f}, y={sensor_y:.2f}, yaw={math.degrees(sensor_yaw):.2f} deg)") # Debugging print

        return sensor_poses

    def create_detection_sets(self, ego_ground_truth, ground_truth_objects, ground_truth_spatial_index, time):
        """
        Create detection sets for each sensor in the SensorPackage using the sensor poses.
        
        Args:
            ego_ground_truth (GroundTruthObject): The ground truth object for the ego vehicle.
            ground_truth_objects (list): A list of GroundTruthObject instances.
            ground_truth_spatial_index (cKDTree): Spatial index for fast range queries (can be None).
            time (float): The timestamp of the detection set.
        """
        self.ground_truth_obj = ego_ground_truth

        # Get the velocity of the ego vehicle
        velocity = math.sqrt(ego_ground_truth.velocity_vector[0]**2 + ego_ground_truth.velocity_vector[1]**2)

        # Get the localized pose with potential errors
        adjusted_x, adjusted_y, adjusted_yaw = self.localizer.get_localization_pose(
            ego_ground_truth.centroid[0],
            ego_ground_truth.centroid[1],
            ego_ground_truth.angle,
            velocity
        )

        # Update the ego vehicle's position and angle
        self.ground_truth_obj.centroid = [adjusted_x, adjusted_y]
        self.ground_truth_obj.angle = adjusted_yaw

        sensor_poses = self.set_sensor_poses()

        self.sensor_detection_sets = []
        all_sensors_detected_ground_truths = [] # Collect all detected GT from all sensors
        all_sensors_detected_objects = [] # Collect all detected objects from all sensors
        detectable_ground_truth = {} # Initialize once per frame for the entire SensorPackage

        for sensor_obj, sensor_pose in zip(self.sensors, sensor_poses):
            # Filter ground truth by range so that we don't do unnecessary calculations
            # Use spatial index if available for much faster filtering
            if ground_truth_spatial_index is not None:
                ground_truth_objects_filtered = ground_truth.filter_ground_truth_by_range_indexed(
                    ground_truth_spatial_index, ground_truth_objects, sensor_pose[0], sensor_pose[1], sensor_obj.max_range)
            else:
                # Fallback to linear search if spatial index is not available
                ground_truth_objects_filtered = ground_truth.filter_ground_truth_by_range(ground_truth_objects, sensor_pose[0], sensor_pose[1], sensor_obj.max_range)

            # Create detected objects
            detected_objects, detected_ground_truths = sensor.create_detected_bounding_boxes(sensor_obj, sensor_pose, ground_truth_objects_filtered, self.sensor_package_id)
            # print(f"  Sensor {sensor_obj.sensor_type_id}: Detected ground truths count: {len(detected_ground_truths)}") # Debugging print

            all_sensors_detected_ground_truths.extend(detected_ground_truths) # Collect for merging
            all_sensors_detected_objects.extend(detected_objects) # Collect for merging

            # Inject malicious removal, addition, or convoy errors if applicable
            if self.error.error_type == ErrorType.MALICIOUS_REMOVAL:
                detected_objects = self.error.inject_error(detected_objects, True)
            elif self.error.error_type == ErrorType.MALICIOUS_ADDITION:
                detected_objects = self.error.inject_error(detected_objects, True)
            elif self.error.error_type == ErrorType.MALICIOUS_CONVOY:
                detected_objects = self.error.inject_error(detected_objects, True)

            self.sensor_detection_sets.append(detected_objects)

            # Process and match the detections from each sensor (filtered to exclude ego)
            filtered_detected_objects_for_fusion = [obj for obj in detected_objects if obj.vehicle_id != self.sensor_package_id]
            self.sensor_fusion.processDetectionFrame(time, filtered_detected_objects_for_fusion, .3)
        
        # After all sensors have processed, populate the final detectable_ground_truth for this SensorPackage
        for gt in all_sensors_detected_ground_truths:
            if gt.vehicle_id != self.sensor_package_id:
                detectable_ground_truth[gt.vehicle_id] = gt

        # Filter out ego vehicle from the overall detected objects for metrics
        final_detected_objects_for_metrics = [obj for obj in all_sensors_detected_objects if obj.vehicle_id != self.sensor_package_id]

        self.detection_set_timestamp = time

        # Kick off the actual fusion process
        result, visualization, _, _ = self.sensor_fusion.fuseDetectionFrame(time)

        return result, final_detected_objects_for_metrics, detectable_ground_truth

    def draw_detected_objects(self, traci_instance, polygon_ids):
        """
        Draw all detected objects for each sensor in the SensorPackage using the TraCI instance.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            polygon_ids (list): A list to store the IDs of the drawn polygons.
        """
        for detected_objects in self.sensor_detection_sets:
            for detected_obj in detected_objects:
                if detected_obj.draw_bounding_box_in_sumo(traci_instance):
                    polygon_ids.append(f"bbox_{detected_obj.vehicle_id}")
