import math
import copy
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
        has_error (bool): Whether this CAV has permanent error flag (set at creation).
    """
    
    def __init__(self, sensor_package_id, sensors, sensors_extrinsics, localizer, error_package, has_error=False):
        """
        Initialize the SensorPackage with its ID, sensors, sensors' extrinsics, and localizer.
        
        Args:
            sensor_package_id (str): The ID of the SensorPackage.
            sensors (list): A list of sensor objects for the SensorPackage.
            sensors_extrinsics (list): A list of extrinsic parameters for each sensor.
            localizer (Localizer): The localizer object for the SensorPackage.
            error_package (ErrorPackage): The error package to apply errors.
            has_error (bool): Whether this CAV has errors (set ONCE at creation, permanent).
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
        self.has_error = has_error  # Permanent error flag for this CAV
        self.roll1 = random.random()
        self.roll2 = random.random()
        self.error_x = None
        self.error_y = None

        # Deep copy and assign initial extrinsics to each sensor object
        # Each SensorPackage needs its own sensor instances to have independent error states
        for i, sensor_obj in enumerate(sensors):
            # Create a deep copy so each SensorPackage has independent sensors
            sensor_copy = copy.deepcopy(sensor_obj)
            sensor_copy.x_extrinsics_original = sensors_extrinsics[i][0]
            sensor_copy.y_extrinsics_original = sensors_extrinsics[i][1]
            sensor_copy.angle_extrinsics_original = sensors_extrinsics[i][2]
            # Reset error offsets for this fresh copy
            sensor_copy.x_offset_error = 0.0
            sensor_copy.y_offset_error = 0.0
            sensor_copy.angle_offset_error = 0.0
            self.sensors.append(sensor_copy)

        # Apply single or multi-sensor extrinsics errors ONCE at creation (if this CAV has errors)
        if self.error.is_active and self.has_error:
            if self.error.error_type == ErrorType.SINGLE_SENSOR_EXTRINSICS:
                # Apply error to a single random sensor (permanent calibration error)
                target_sensor_idx = random.randint(0, len(self.sensors) - 1)
                self.sensors[target_sensor_idx] = self.error.inject_single_sensor_extrinsics_error(self.sensors[target_sensor_idx])
            elif self.error.error_type == ErrorType.MULTI_SENSOR_EXTRINSICS:
                # Apply error to all sensors (permanent calibration error)
                self.sensors = self.error.inject_multi_sensor_extrinsics_error(self.sensors)
        
    def set_sensor_poses(self, true_ego_pose, believed_ego_pose):
        """
        Set the sensor poses for the SensorPackage based on true and believed ego poses.
        
        This handles both extrinsics errors AND localization errors:
        - Extrinsics error: sensor is at wrong position relative to ego
        - Localization error: ego's believed position differs from true position
        
        Args:
            true_ego_pose (tuple): Where the ego vehicle actually is (x, y, yaw)
            believed_ego_pose (tuple): Where localization thinks the ego is (x, y, yaw)
        
        Returns:
            tuple: (actual_poses, believed_poses) where each is a list of (x, y, yaw).
                   - actual_poses: Where sensors physically are (true ego + actual extrinsics)
                   - believed_poses: Where software thinks sensors are (believed ego + believed extrinsics)
        """
        true_x, true_y, true_yaw = true_ego_pose
        believed_x, believed_y, believed_yaw = believed_ego_pose
        
        actual_poses = []
        believed_poses = []

        for sensor_obj in self.sensors:
            # ACTUAL position: TRUE ego pose + actual extrinsics (with calibration errors)
            # This is where the sensor PHYSICALLY is
            actual_x_rel = sensor_obj.x_extrinsics_original + sensor_obj.x_offset_error
            actual_y_rel = sensor_obj.y_extrinsics_original + sensor_obj.y_offset_error
            actual_yaw_rel = sensor_obj.angle_extrinsics_original + sensor_obj.angle_offset_error

            actual_sensor_x = true_x + actual_x_rel * math.cos(true_yaw) - actual_y_rel * math.sin(true_yaw)
            actual_sensor_y = true_y + actual_x_rel * math.sin(true_yaw) + actual_y_rel * math.cos(true_yaw)
            actual_sensor_yaw = true_yaw + actual_yaw_rel
            actual_poses.append((actual_sensor_x, actual_sensor_y, actual_sensor_yaw))

            # BELIEVED position: BELIEVED ego pose + original extrinsics (no calibration errors)
            # This is where the software THINKS the sensor is
            believed_x_rel = sensor_obj.x_extrinsics_original
            believed_y_rel = sensor_obj.y_extrinsics_original
            believed_yaw_rel = sensor_obj.angle_extrinsics_original

            believed_sensor_x = believed_x + believed_x_rel * math.cos(believed_yaw) - believed_y_rel * math.sin(believed_yaw)
            believed_sensor_y = believed_y + believed_x_rel * math.sin(believed_yaw) + believed_y_rel * math.cos(believed_yaw)
            believed_sensor_yaw = believed_yaw + believed_yaw_rel
            believed_poses.append((believed_sensor_x, believed_sensor_y, believed_sensor_yaw))

        return actual_poses, believed_poses

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

        # TRUE ego pose - where the vehicle actually is
        true_ego_pose = (
            ego_ground_truth.centroid[0],
            ego_ground_truth.centroid[1],
            ego_ground_truth.angle
        )

        # Get the velocity of the ego vehicle
        velocity = math.sqrt(ego_ground_truth.velocity_vector[0]**2 + ego_ground_truth.velocity_vector[1]**2)

        # Get the BELIEVED ego pose with potential localization errors
        # This is where the vehicle THINKS it is (may differ from truth if has_error=True)
        believed_x, believed_y, believed_yaw = self.localizer.get_localization_pose(
            ego_ground_truth.centroid[0],
            ego_ground_truth.centroid[1],
            ego_ground_truth.angle,
            velocity,
            has_error=self.has_error
        )
        believed_ego_pose = (believed_x, believed_y, believed_yaw)

        # Calculate sensor poses using both true and believed ego poses
        # - actual_poses: where sensors physically are (for measurement)
        # - believed_poses: where software thinks sensors are (for transform back)
        actual_poses, believed_poses = self.set_sensor_poses(true_ego_pose, believed_ego_pose)

        # GET LOCALIZATION COVARIANCE (GPEM Section VII-A)
        # Combine localization uncertainty with perception uncertainty
        loc_covariance = self.localizer.get_localization_covariance(velocity, believed_yaw)

        self.sensor_detection_sets = []
        all_sensors_detected_ground_truths = [] # Collect all detected GT from all sensors
        all_sensors_detected_objects = [] # Collect all detected objects from all sensors
        detectable_ground_truth = {} # Initialize once per frame for the entire SensorPackage

        for sensor_obj, actual_pose, believed_pose in zip(self.sensors, actual_poses, believed_poses):
            # Filter ground truth by range so that we don't do unnecessary calculations
            # Use spatial index if available for much faster filtering
            # Use ACTUAL pose for filtering (what the sensor can physically see)
            if ground_truth_spatial_index is not None:
                ground_truth_objects_filtered = ground_truth.filter_ground_truth_by_range_indexed(
                    ground_truth_spatial_index, ground_truth_objects, actual_pose[0], actual_pose[1], sensor_obj.max_range)
            else:
                # Fallback to linear search if spatial index is not available
                ground_truth_objects_filtered = ground_truth.filter_ground_truth_by_range(ground_truth_objects, actual_pose[0], actual_pose[1], sensor_obj.max_range)

            # Create detected objects - pass both actual and believed poses for proper extrinsics error simulation
            detected_objects, detected_ground_truths = sensor.create_detected_bounding_boxes(
                sensor_obj, actual_pose, believed_pose, ground_truth_objects_filtered, self.sensor_package_id)
            
            # ADD LOCALIZATION COVARIANCE TO PERCEPTION COVARIANCE (Eq 18: Σ_total = Σ_loc + Σ_perc)
            for det in detected_objects:
                if det.error_covariance is not None:
                    # Equation 18 in the paper: Add the two matrices
                    det.error_covariance = det.error_covariance + loc_covariance
            
            all_sensors_detected_ground_truths.extend(detected_ground_truths) # Collect for merging
            all_sensors_detected_objects.extend(detected_objects) # Collect for merging

            # Inject detection-level errors if this CAV has errors
            if self.has_error and self.error.is_active:
                if self.error.error_type == ErrorType.MALICIOUS_REMOVAL:
                    detected_objects = self.error.inject_malicious_removal_error(detected_objects)
                elif self.error.error_type == ErrorType.MALICIOUS_ADDITION:
                    detected_objects = self.error.inject_malicious_addition_error(detected_objects)
                elif self.error.error_type == ErrorType.MALICIOUS_CONVOY:
                    detected_objects = self.error.inject_malicious_convoy_error(detected_objects)
                elif self.error.error_type == ErrorType.HIGH_MISS_RATE:
                    detected_objects = self.error.inject_high_miss_rate_error(detected_objects)
                elif self.error.error_type == ErrorType.DETECTION_LAG:
                    # Lag requires current step - convert time to step (assuming 0.1s per step)
                    current_step = int(time * 10)
                    detected_objects = self.error.inject_detection_lag_error(
                        detected_objects, self.sensor_package_id, current_step)

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

        # ADD SELF-LOCALIZATION TO SHARED DETECTIONS
        # This allows other CAVs to benefit from our accurate localized position
        # The self-localization uses localizer covariance (NOT detector covariance)
        self_localization_detection = self._create_self_localization_detection(
            ego_ground_truth, believed_ego_pose, velocity, loc_covariance
        )
        if self_localization_detection is not None:
            final_detected_objects_for_metrics.append(self_localization_detection)

        self.detection_set_timestamp = time

        # Kick off the actual fusion process
        result, visualization, _, _ = self.sensor_fusion.fuseDetectionFrame(time)

        return result, final_detected_objects_for_metrics, detectable_ground_truth

    def _create_self_localization_detection(self, ego_ground_truth, believed_ego_pose, velocity, loc_covariance):
        """
        Create a DetectedObject representing this CAV's own localized position.
        
        This detection uses ONLY localizer covariance (no detector covariance added).
        When shared to global fusion, other CAVs can use our accurate self-reported
        position, which is especially valuable at high CAV penetration rates.
        
        Args:
            ego_ground_truth (GroundTruthObject): Ground truth for the ego vehicle.
            believed_ego_pose (tuple): The believed (localized) pose (x, y, yaw).
            velocity (float): Current velocity in m/s.
            loc_covariance (np.ndarray): 2x2 localizer covariance matrix.
            
        Returns:
            DetectedObject: Self-localization detection with localizer covariance,
                           or None if unable to create.
        """
        import gaussians
        
        believed_x, believed_y, believed_yaw = believed_ego_pose
        
        # Get localizer standard deviations for the Gaussian
        loc_long_std = self.localizer.get_longitudinal_localization_std(velocity)
        loc_lat_std = self.localizer.get_lateral_localization_std(velocity)
        
        # Create the expected error Gaussian with localizer uncertainty
        # (oriented along the vehicle heading)
        loc_gaussian = gaussians.BivariateGaussian(loc_long_std**2, loc_lat_std**2, believed_yaw)
        
        # Create the DetectedObject for self-localization
        # Use the believed (localized) position, NOT ground truth
        self_detection = sensor.DetectedObject(
            vehicle_id=self.sensor_package_id,
            vehicle_type=ego_ground_truth.type,
            detected_bbox=None,  # No bbox from localization
            centroid=(believed_x, believed_y),
            width=ego_ground_truth.dimensions[0],
            length=ego_ground_truth.dimensions[1],
            angle=believed_yaw,
            expected_error_gaussian=loc_gaussian,
            velocity_vector=ego_ground_truth.velocity_vector,
            error_covariance=loc_covariance,  # Pure localizer covariance, no detector added
            width_std=0.1,  # Low uncertainty on own dimensions
            length_std=0.1
        )
        
        return self_detection

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
