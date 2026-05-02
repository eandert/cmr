import math
import random
import numpy as np

# Project specific imports
import utils
import gaussians
from config.detector_type import DetectorType
from error_models import get_error_model

class Sensor:
    """
    Class representing a sensor with its properties and error models.
    
    Attributes:
        sensor_type_id (int): The ID of the sensor type.
        detector_type_id (int): The ID of the detector type.
        max_range (int): The maximum range of the sensor in meters.
        horizontal_fov (float): The horizontal field of view of the sensor in radians.
        center_angle (float): The center angle of the sensor in radians.
        centroid_radial_error_polynomial (Polynomial): Polynomial for centroid radial error.
        centroid_distance_error_polynomial (Polynomial): Polynomial for centroid distance error.
        bounding_box_error_polynomial (Polynomial): Polynomial for bounding box error.
        detection_probability_polynomial (Polynomial): Polynomial for detection probability.
    """
    
    def __init__(self, sensor_type, detector_type, use_gpem_model=False, use_quadratic=False, detector_max_range=None, enable_realistic_fp=False):
        """
        Initialize the sensor with its properties and error models.
        
        Args:
            sensor_type (SensorType): The type of the sensor.
            detector_type (DetectorType): The type of the detector.
            use_gpem_model (bool): If True, use GPEM parameterized covariance. If False, use static averaged covariance.
            use_quadratic (bool): If True and use_gpem_model=True, use quadratic regression instead of linear.
            detector_max_range (float): Max detection range in meters. None uses default (60m).
            enable_realistic_fp (bool): If True, inject Poisson-sampled false positives drawn from
                the polar calibration bins. Default False — no change to existing behaviour.
        """
        self.sensor_type_id = sensor_type.id
        self.detector_type_id = detector_type.id
        self.detector_type = detector_type  # Keep reference for error model lookup
        self.max_range = sensor_type.max_range
        self.horizontal_fov = sensor_type.horizontal_fov
        self.center_angle = sensor_type.center_angle
        self.centroid_radial_error_polynomial = utils.Polynomial(detector_type.centroid_radial_error_polynomial)
        self.centroid_distance_error_polynomial = utils.Polynomial(detector_type.centroid_distance_error_polynomial)
        self.bounding_box_error_polynomial = utils.Polynomial(detector_type.bounding_box_error_polynomial)
        self.detection_probability_polynomial = utils.Polynomial(detector_type.detection_probability_polynomial)
        
        # Load regression-tested error model if available
        self.error_model = None
        if hasattr(detector_type, 'error_model_name') and detector_type.error_model_name:
            self.error_model = get_error_model(detector_type.error_model_name, use_gpem_model=use_gpem_model,
                                               use_quadratic=use_quadratic, max_range=detector_max_range)
            # Override sensor max_range with error model's max_range for consistent filtering
            # This ensures ground truth filtering matches the detector's actual range
            self.max_range = self.error_model.max_range
        
        self.enable_realistic_fp = enable_realistic_fp
        self.fp_per_frame_cap = None       # set via config["fp_per_frame_cap"]
        self.fp_score_threshold = 0.3      # only inject FPs that survive the detection threshold

        # Initialize sensor's original extrinsics relative to the SensorPackage
        self.x_extrinsics_original = 0.0
        self.y_extrinsics_original = 0.0
        self.angle_extrinsics_original = 0.0
        # Initialize sensor's extrinsics error offsets
        self.x_offset_error = 0.0
        self.y_offset_error = 0.0
        self.angle_offset_error = 0.0

    def check_in_range_and_fov(self, target_angle, distance):
        """
        Check if a target is within the sensor's range and field of view.
        
        Args:
            target_angle (float): The angle of the target in radians.
            distance (float): The distance to the target in meters.
        
        Returns:
            bool: True if the target is within range and field of view, False otherwise.
        """
        anglediff = ((self.center_angle - target_angle + math.pi + (2 * math.pi)) % (2 * math.pi)) - math.pi
        return abs(anglediff) <= self.horizontal_fov and distance <= self.max_range
        
    def centroid_radial_error_at_distance(self, distance):
        """
        Get the centroid radial error at a given distance.
        
        Args:
            distance (float): The distance to the target in meters.
        
        Returns:
            float: The centroid radial error.
        """
        return self.centroid_radial_error_polynomial.evaluate(distance)
        
    def centroid_distance_error_at_distance(self, distance):
        """
        Get the centroid distance error at a given distance.
        
        Args:
            distance (float): The distance to the target in meters.
        
        Returns:
            float: The centroid distance error.
        """
        return self.centroid_distance_error_polynomial.evaluate(distance)
    
    def bounding_box_error_at_distance(self, distance):
        """
        Get the bounding box error at a given distance.
        
        Args:
            distance (float): The distance to the target in meters.
        
        Returns:
            float: The bounding box error.
        """
        return self.bounding_box_error_polynomial.evaluate(distance)
        
    def detection_probability_at_distance(self, distance, angle_deg=None):
        """
        Get the detection probability at a given distance (and optionally angle).

        Args:
            distance (float): The distance to the target in meters.
            angle_deg (float, optional): Angle to target in degrees (-180 to 180) for polar lookup.

        Returns:
            float: The detection probability.
        """
        if self.error_model:
            return self.error_model.detection_probability(distance, angle_deg)
        return self.detection_probability_polynomial.evaluate(distance)
    
    def uses_regression_error_model(self):
        """Check if this sensor uses a regression-tested error model."""
        return self.error_model is not None


class DetectedObject:
    def __init__(self, vehicle_id, vehicle_type, detected_bbox, centroid, width, length, angle, expected_error_gaussian, velocity_vector=None, error_covariance=None, width_std=0.5, length_std=0.5, yaw_variance=None):
        self.vehicle_id = vehicle_id
        self.type = vehicle_type
        self.detected_bbox = detected_bbox
        self.centroid = centroid
        self.dimensions = [width, length]  # width, length
        self.angle = angle
        self.velocity_vector = velocity_vector
        self.expected_error_gaussian = expected_error_gaussian
        self.error_covariance = error_covariance
        self.width_std = width_std
        self.length_std = length_std
        self.yaw_variance = yaw_variance  # MSE for heading (bias² + var), from GPEM

    @property
    def detected_bbox_corners(self):
        angle = self.angle
        half_width = self.dimensions[0] / 2
        half_length = self.dimensions[1] / 2

        # Center of the bounding box
        cx, cy = self.centroid

        # Standard convention: length is along heading axis, width is perpendicular
        # Use robust rotate_point to avoid manual math errors
        bbox = [
            utils.rotate_point(cx - half_length, cy - half_width, cx, cy, angle),
            utils.rotate_point(cx + half_length, cy - half_width, cx, cy, angle),
            utils.rotate_point(cx + half_length, cy + half_width, cx, cy, angle),
            utils.rotate_point(cx - half_length, cy + half_width, cx, cy, angle)
        ]
        return bbox

    def __str__(self):
        return (f"Type: {self.type}, Detected BBox: {self.detected_bbox}, "
                f"Dimensions: {self.dimensions}, Rotation: {self.angle}, "
                f"Velocity Vector: {self.velocity_vector}")
    
    def draw_bounding_box_in_sumo(self, traci_instance, color=(255, 255, 0, 255), layer=10, sensor_package_id="global"):
        """
        Draw the bounding box of this detected object in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the bounding box in RGBA format. Defaults to yellow.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 7.
            sensor_package_id (str): The ID of the sensor package that detected this object. Used to create unique polygon IDs.
        """
        bbox = self.detected_bbox
        vehicle_id = self.vehicle_id

        # Ensure the bounding box coordinates are valid
        if not all(isinstance(point, tuple) and len(point) == 2 for point in bbox):
            raise ValueError(f"Invalid bounding box coordinates: {bbox}")

        # Draw the bounding box as a polygon in SUMO
        try:
            polygon_unique_id = f"bbox_{sensor_package_id}_{vehicle_id}"
            traci_instance.polygon.add(
                polygonID=polygon_unique_id,
                shape=bbox,
                color=color,
                fill=True,
                layer=layer
            )
            return polygon_unique_id
        except Exception as e:
            print(f"Error drawing bounding box for vehicle {vehicle_id}: {e}")
            return None

    def draw_position_vector_in_sumo(self, traci_instance, color=(0, 255, 255, 255), length=11, layer=8, polygon_id_prefix=""):
        """
        Draw the position vector from the centroid and angle of the bounding box in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the position vector in RGBA format. Defaults to green.
            length (float): The length of the position vector. Defaults to 5.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 11.
            polygon_id_prefix (str): Prefix to make the polygon ID unique across different types of visualizations.
        """
        cx, cy = self.centroid
        angle_rad = self.angle  # Use the stored angle in radians
        end_x = cx + length * math.cos(angle_rad)
        end_y = cy + length * math.sin(angle_rad)

        # Draw the position vector as a thin polygon in SUMO
        polygon_vector_id = f"vector_{polygon_id_prefix}_{self.vehicle_id}"
        traci_instance.polygon.add(
            polygonID=polygon_vector_id,
            shape=[(cx, cy), (end_x, end_y)],
            color=color,
            fill=False,
            layer=layer
        )
        return polygon_vector_id # Return the ID for management

def create_detected_bounding_boxes(sensor, actual_pose, believed_pose, ground_truth_objects, sensor_package_id):
    """
    Create a set of detected bounding boxes based on the sensor properties and ground truth objects.
    
    Extrinsics error simulation:
    - actual_pose: Where the sensor physically is (with calibration errors)
    - believed_pose: Where the sensor's software thinks it is (original calibration)
    
    When there's an extrinsics error:
    1. Sensor measures from its actual physical position
    2. But transforms back to global using believed position (wrong calibration)
    3. This causes systematic position errors in detections
    
    Args:
        sensor (Sensor): The sensor object.
        actual_pose (tuple): The sensor's actual physical pose as (x, y, yaw) with extrinsics errors.
        believed_pose (tuple): The sensor's believed pose as (x, y, yaw) without extrinsics errors.
        ground_truth_objects (list): A list of GroundTruthObject instances.
        sensor_package_id (str): The ID of the sensor package.
    
    Returns:
        tuple: A tuple containing a list of DetectedObject instances and a list of corresponding ground truth objects.
    """
    detected_objects = []
    detected_ground_truths = []
    
    # ACTUAL position - where sensor physically is
    actual_x = actual_pose[0]
    actual_y = actual_pose[1]
    actual_yaw = actual_pose[2]
    
    # BELIEVED position - where sensor's software thinks it is
    believed_x = believed_pose[0]
    believed_y = believed_pose[1]
    believed_yaw = believed_pose[2]

    detection_id = 0

    for gt_obj in ground_truth_objects:
        # Explicitly skip the ego vehicle itself
        if gt_obj.vehicle_id == sensor_package_id:
            continue

        # Calculate the relative position of the ground truth object to the ACTUAL sensor position
        # (This is what the sensor physically measures)
        dx = gt_obj.centroid[0] - actual_x
        dy = gt_obj.centroid[1] - actual_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_obj = math.atan2(dy, dx) - actual_yaw

        # Normalize the angle to the range [-pi, pi]
        angle_to_obj = (angle_to_obj + math.pi) % (2 * math.pi) - math.pi

        # Check if the object is within the sensor's range and field of view but not ourself (distance > 0)
        if sensor.check_in_range_and_fov(angle_to_obj, distance) and distance > 0.001:
            # Calculate the detection probability (pass angle for polar lookup)
            angle_to_obj_deg = math.degrees(angle_to_obj)
            detection_probability = sensor.detection_probability_at_distance(distance, angle_to_obj_deg)
            probability = random.random()

            # Use this to determine if the object is detected
            if probability <= detection_probability:
                detected_ground_truths.append(gt_obj) # Only add to detected_ground_truths if actually detected

                # If it's a PERFECT detector, set errors to zero
                if sensor.detector_type_id == DetectorType.PERFECT.id:
                    actual_error = (0.0, 0.0)
                    width_error = 0.0
                    length_error = 0.0
                    yaw_error = 0.0
                    width_std = 1e-6
                    length_std = 1e-6
                    expected_error_gaussian = gaussians.BivariateGaussian(0.000001, 0.000001, angle_to_obj) # Near zero covariance
                elif sensor.uses_regression_error_model():
                    # Use regression-tested error model (distance-binned distributions)
                    error_model = sensor.error_model
                    
                    # Sample all errors including position, dimensions, and yaw
                    errors = error_model.sample_all_errors(distance, angle_to_obj)
                    actual_error = (errors['x_error'], errors['y_error'])
                    width_error = errors['width_error']
                    length_error = errors['length_error']
                    yaw_error = errors['yaw_error']
                    
                    # Store dimension stds for Kalman filter
                    width_std = errors['width_std']
                    length_std = errors['length_std']
                    
                    # Build covariance from the predicted stds
                    # ROTATION: Bring sensor-relative covariance into global frame
                    # distal_std is along the ray to the object (angle_to_obj in sensor frame)
                    # To get global distal axis, add sensor heading (believed_yaw)
                    expected_error_gaussian = gaussians.BivariateGaussian(
                        errors['distal_std']**2, errors['perp_std']**2, believed_yaw + angle_to_obj
                    )
                else:
                    # Use polynomial-based error model
                    radial_error = sensor.centroid_radial_error_at_distance(distance)
                    distal_error = sensor.centroid_distance_error_at_distance(distance)
                    bbox_error = sensor.bounding_box_error_at_distance(distance)

                    # SAMPLING: Sample in distal/radial frame and rotate to sensor frame
                    # (actual_error is added to local_x/y later)
                    actual_radial_samp = np.random.normal(0, radial_error)
                    actual_distal_samp = np.random.normal(0, distal_error)
                    actual_error_x = actual_distal_samp * math.cos(angle_to_obj) - actual_radial_samp * math.sin(angle_to_obj)
                    actual_error_y = actual_distal_samp * math.sin(angle_to_obj) + actual_radial_samp * math.cos(angle_to_obj)
                    actual_error = (actual_error_x, actual_error_y)

                    # COVARIANCE: Build global covariance for the Kalman filter
                    # distal_std is along the ray to the object (angle_to_obj in sensor frame)
                    # To get global distal axis, add sensor heading (believed_yaw)
                    expected_error_gaussian = gaussians.BivariateGaussian(
                        distal_error**2, radial_error**2, believed_yaw + angle_to_obj
                    )

                    # Calculate the new bounding box error
                    width_error, length_error = calculateBBoxError(bbox_error)
                    yaw_error = 0.0  # No yaw error for polynomial model
                    width_std = bbox_error
                    length_std = bbox_error

                # EXTRINSICS ERROR HANDLING:
                # Simulate proper extrinsics calibration error:
                # 1. Sensor is physically at ACTUAL position (with calibration errors)
                # 2. Sensor measures relative position from its ACTUAL physical location
                # 3. But sensor's software transforms back using BELIEVED position (wrong calibration)
                # This causes systematic position errors when calibration is wrong.
                
                # Step 1: Transform to sensor-local frame using ACTUAL orientation
                # (dx, dy already calculated relative to ACTUAL sensor position above)
                # actual_yaw is now in standard math convention
                cos_actual = math.cos(-actual_yaw)
                sin_actual = math.sin(-actual_yaw)
                local_x = dx * cos_actual - dy * sin_actual
                local_y = dx * sin_actual + dy * cos_actual
                
                # Step 2: Add sensing noise in local frame (radial/perpendicular errors)
                local_x += actual_error[0]
                local_y += actual_error[1]
                
                # Step 3: Transform back to global using BELIEVED sensor pose (wrong calibration!)
                # This is where extrinsics errors manifest - sensor uses wrong pose in transform
                # believed_yaw is now in standard math convention
                cos_believed = math.cos(believed_yaw)
                sin_believed = math.sin(believed_yaw)
                new_centroid_x = believed_x + local_x * cos_believed - local_y * sin_believed
                new_centroid_y = believed_y + local_x * sin_believed + local_y * cos_believed
                
                # Apply yaw error to the detected angle
                detected_angle = gt_obj.angle + yaw_error

                # Calculate the size of the bounding box in ± width and ± length
                detected_width = gt_obj.dimensions[0] + width_error
                detected_length = gt_obj.dimensions[1] + length_error
                half_width = detected_width / 2
                half_length = detected_length / 2

                # Compute the global coordinates of the bounding box corners (axis-aligned at 0 rad)
                # At 0 radians (East), length is along x and width is along y.
                bbox_x_min = new_centroid_x - half_length
                bbox_x_max = new_centroid_x + half_length
                bbox_y_min = new_centroid_y - half_width
                bbox_y_max = new_centroid_y + half_width
                
                # Rotate the bounding box based on the detected angle (with error)
                detected_bbox = [
                    utils.rotate_point(bbox_x_min, bbox_y_min, new_centroid_x, new_centroid_y, detected_angle),
                    utils.rotate_point(bbox_x_max, bbox_y_min, new_centroid_x, new_centroid_y, detected_angle),
                    utils.rotate_point(bbox_x_max, bbox_y_max, new_centroid_x, new_centroid_y, detected_angle),
                    utils.rotate_point(bbox_x_min, bbox_y_max, new_centroid_x, new_centroid_y, detected_angle)
                ]

                tp_det = DetectedObject(
                    vehicle_id=f"{gt_obj.vehicle_id}_{detection_id}",
                    vehicle_type=gt_obj.type,
                    detected_bbox=detected_bbox,
                    centroid=(new_centroid_x, new_centroid_y),
                    width=detected_width,
                    length=detected_length,
                    angle=detected_angle,
                    expected_error_gaussian=expected_error_gaussian,
                    velocity_vector=gt_obj.velocity_vector,
                    error_covariance=expected_error_gaussian.covariance,
                    width_std=width_std,
                    length_std=length_std
                )
                tp_det.p_tp = 0.9
                detected_objects.append(tp_det)

                detection_id += 1

    # Inject realistic false positives (off by default; enable via sensor.enable_realistic_fp)
    if sensor.enable_realistic_fp and sensor.error_model is not None:
        for fp in sensor.error_model.sample_false_positives(
            believed_x, believed_y, believed_yaw,
            max_fp_per_frame=sensor.fp_per_frame_cap,
            score_threshold=sensor.fp_score_threshold,
        ):
            gx, gy = fp["x"], fp["y"]
            w, l   = fp["width"], fp["length"]
            angle  = fp["angle_rad"]  # global angle sensor→fp, used as heading prior

            half_w = w / 2
            half_l = l / 2
            detected_bbox = [
                utils.rotate_point(gx - half_l, gy - half_w, gx, gy, angle),
                utils.rotate_point(gx + half_l, gy - half_w, gx, gy, angle),
                utils.rotate_point(gx + half_l, gy + half_w, gx, gy, angle),
                utils.rotate_point(gx - half_l, gy + half_w, gx, gy, angle),
            ]

            # Position covariance: distal axis along sensor→fp ray
            fp_cov = gaussians.BivariateGaussian(0.5**2, 0.3**2, angle)

            det_obj = DetectedObject(
                vehicle_id=f"fp_{detection_id}",
                vehicle_type=fp["class_label"],
                detected_bbox=detected_bbox,
                centroid=(gx, gy),
                width=w,
                length=l,
                angle=angle,
                expected_error_gaussian=fp_cov,
                velocity_vector=(0.0, 0.0),
                error_covariance=fp_cov.covariance,
                width_std=0.5,
                length_std=0.5,
            )
            det_obj.det_score = fp["score"]
            det_obj.p_tp = fp["p_tp"]  # Bayesian P(TP|score,range,angle) from calibration
            detected_objects.append(det_obj)
            detection_id += 1

    return detected_objects, detected_ground_truths

def calculateBBoxError(bbox_error):
    # Calculate our expected errors in width and length
    width_error = np.random.normal(0, bbox_error, 1)[0]
    length_error = np.random.normal(0, bbox_error, 1)[0]
    return (width_error, length_error)