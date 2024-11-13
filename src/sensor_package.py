import math
import utils
import sensor
import ground_truth
import sensor_fusion

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
    
    def __init__(self, sensor_package_id, sensors, sensors_extrinsics, localizer, error_package = None):
        """
        Initialize the SensorPackage with its ID, sensors, sensors' extrinsics, and localizer.
        
        Args:
            sensor_package_id (str): The ID of the SensorPackage.
            sensors (list): A list of sensor objects for the SensorPackage.
            sensors_extrinsics (list): A list of extrinsic parameters for each sensor.
            localizer (Localizer): The localizer object for the SensorPackage.
        """
        self.sensor_package_id = sensor_package_id
        self.ground_truth_obj = None
        self.sensors = sensors
        self.sensors_extrinsics = sensors_extrinsics
        self.sensor_detection_sets = []
        self.detection_set_timestamp = None
        self.integer_id = utils.extract_id(sensor_package_id)
        self.sensor_fusion = sensor_fusion.Fusion(self.integer_id)
        self.localizer = localizer

    def set_sensor_poses(self):
        """
        Set the sensor poses for the SensorPackage based on the SensorPackage's pose and the sensors' extrinsics.
        
        Returns:
            list: A list of sensor poses as (x, y, yaw).
        """
        cav_x, cav_y, cav_yaw = self.ground_truth_obj.centroid[0], self.ground_truth_obj.centroid[1], self.ground_truth_obj.angle
        sensor_poses = []

        for extrinsics in self.sensors_extrinsics:
            sensor_x_rel, sensor_y_rel, sensor_yaw_rel = extrinsics

            # Calculate the sensor's absolute position
            sensor_x = cav_x + sensor_x_rel * math.cos(cav_yaw) - sensor_y_rel * math.sin(cav_yaw)
            sensor_y = cav_y + sensor_x_rel * math.sin(cav_yaw) + sensor_y_rel * math.cos(cav_yaw)
            sensor_yaw = cav_yaw + sensor_yaw_rel

            sensor_poses.append((sensor_x, sensor_y, sensor_yaw))

        return sensor_poses

    def create_detection_sets(self, ego_ground_truth, ground_truth_objects, time):
        """
        Create detection sets for each sensor in the SensorPackage using the sensor poses.
        
        Args:
            ego_ground_truth (GroundTruthObject): The ground truth object for the ego vehicle.
            ground_truth_objects (list): A list of GroundTruthObject instances.
            time (float): The timestamp of the detection set.
        """
        self.ground_truth_obj = ego_ground_truth

        # Get the velocity of the ego vehicle
        velocity = math.sqrt(ego_ground_truth.velocity_vector[0]**2 + ego_ground_truth.velocity_vector[1]**2)

        # Calculate the lateral and longitudinal errors
        lateral_error = self.localizer.lateral_error_at_velocity(velocity)
        longitudinal_error = self.localizer.longitudinal_error_at_velocity(velocity)

        # Adjust the position of the ego vehicle based on the errors and its angle
        adjusted_x = ego_ground_truth.centroid[0] + longitudinal_error * math.cos(ego_ground_truth.angle) - lateral_error * math.sin(ego_ground_truth.angle)
        adjusted_y = ego_ground_truth.centroid[1] + longitudinal_error * math.sin(ego_ground_truth.angle) + lateral_error * math.cos(ego_ground_truth.angle)

        # Update the ego vehicle's position
        self.ground_truth_obj.centroid = [adjusted_x, adjusted_y]

        sensor_poses = self.set_sensor_poses()

        self.sensor_detection_sets = []
        detectable_ground_truth = {}

        for sensor_obj, sensor_pose in zip(self.sensors, sensor_poses):
            # Filter ground truth by range so that we don't do unnecessary calculations
            ground_truth_objects_filtered = ground_truth.filter_ground_truth_by_range(ground_truth_objects, sensor_pose[0], sensor_pose[1], sensor_obj.max_range)

            # Create detected objects
            detected_objects, detected_ground_truths = sensor.create_detected_bounding_boxes(sensor_obj, sensor_pose, ground_truth_objects_filtered)

            self.sensor_detection_sets.append(detected_objects)

            # Add detected ground truths to the set
            for gt in detected_ground_truths:
                detectable_ground_truth[utils.extract_id(gt.vehicle_id)] = gt

            # Process and match the detections from each sensor
            self.sensor_fusion.processDetectionFrame(time, detected_objects, 2)

        self.detection_set_timestamp = time

        # Kick off the actual fusion process
        result, vizualization, _, _ = self.sensor_fusion.fuseDetectionFrame(time)

        return result, vizualization, detectable_ground_truth

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