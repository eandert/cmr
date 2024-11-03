import math
import random
import numpy as np

# Project specific imports
import utils
import gaussians

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
    
    def __init__(self, sensor_type, detector_type):
        """
        Initialize the sensor with its properties and error models.
        
        Args:
            sensor_type (SensorType): The type of the sensor.
            detector_type (DetectorType): The type of the detector.
        """
        self.sensor_type_id = sensor_type.id
        self.detector_type_id = detector_type.id
        self.max_range = sensor_type.max_range
        self.horizontal_fov = sensor_type.horizontal_fov
        self.center_angle = sensor_type.center_angle
        self.centroid_radial_error_polynomial = utils.Polynomial(detector_type.centroid_radial_error_polynomial)
        self.centroid_distance_error_polynomial = utils.Polynomial(detector_type.centroid_distance_error_polynomial)
        self.bounding_box_error_polynomial = utils.Polynomial(detector_type.bounding_box_error_polynomial)
        self.detection_probability_polynomial = utils.Polynomial(detector_type.detection_probability_polynomial)
        
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
        
    def detection_probability_at_distance(self, distance):
        """
        Get the detection probability at a given distance.
        
        Args:
            distance (float): The distance to the target in meters.
        
        Returns:
            float: The detection probability.
        """
        return self.detection_probability_polynomial.evaluate(distance)


class DetectedObject:
    def __init__(self, vehicle_id, vehicle_type, detected_bbox, centroid, width, length, angle, expected_error_gaussian, velocity_vector=None):
        self.vehicle_id = vehicle_id
        self.type = vehicle_type
        self.detected_bbox = detected_bbox
        self.centroid = centroid
        self.dimensions = [width, length]  # width, length
        self.rotation = angle
        self.velocity_vector = velocity_vector
        self.expected_error_gaussian = expected_error_gaussian

    def __str__(self):
        return (f"Type: {self.type}, Detected BBox: {self.detected_bbox}, "
                f"Dimensions: {self.dimensions}, Rotation: {self.rotation}, "
                f"Velocity Vector: {self.velocity_vector}")
    
    def draw_bounding_box_in_sumo(self, traci_instance, color=(255, 255, 0, 255), layer=10):
        """
        Draw the bounding box of this detected object in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the bounding box in RGBA format. Defaults to yellow.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 7.
        """
        bbox = self.detected_bbox
        vehicle_id = self.vehicle_id

        # Ensure the bounding box coordinates are valid
        if not all(isinstance(point, tuple) and len(point) == 2 for point in bbox):
            raise ValueError(f"Invalid bounding box coordinates: {bbox}")

        # Draw the bounding box as a polygon in SUMO
        try:
            traci_instance.polygon.add(
                polygonID=f"bbox_{vehicle_id}",
                shape=bbox,
                color=color,
                fill=True,
                layer=layer
            )
            return True
        except Exception as e:
            print(f"Error drawing bounding box for vehicle {vehicle_id}: {e}")
            return False

    def draw_position_vector_in_sumo(self, traci_instance, color=(0, 255, 255, 255), length=11, layer=8):
        """
        Draw the position vector from the centroid and angle of the bounding box in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the position vector in RGBA format. Defaults to green.
            length (float): The length of the position vector. Defaults to 5.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 11.
        """
        cx, cy = self.centroid
        angle_rad = self.rotation  # Use the stored angle in radians
        end_x = cx + length * math.cos(angle_rad)
        end_y = cy + length * math.sin(angle_rad)

        # Draw the position vector as a thin polygon in SUMO
        traci_instance.polygon.add(
            polygonID=f"vector_{self.vehicle_id}",
            shape=[(cx, cy), (end_x, end_y)],
            color=color,
            fill=False,
            layer=layer
        )

def create_detected_bounding_boxes(sensor, sensor_pose, ground_truth_objects):
    """
    Create a set of detected bounding boxes based on the sensor properties and ground truth objects.
    
    Args:
        sensor (Sensor): The sensor object.
        sensor_pose (tuple): The sensor pose as (x, y, yaw).
        ground_truth_objects (list): A list of GroundTruthObject instances.
    
    Returns:
        list: A list of DetectedObject instances.
    """
    detected_objects = []
    sensor_x, sensor_y, sensor_yaw = sensor_pose

    detection_id = 0

    for gt_obj in ground_truth_objects:
        # Calculate the relative position of the ground truth object to the sensor
        dx = gt_obj.location[0] - sensor_x
        dy = gt_obj.location[1] - sensor_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_obj = math.atan2(dy, dx) - sensor_yaw

        # Normalize the angle to the range [-pi, pi]
        angle_to_obj = (angle_to_obj + math.pi) % (2 * math.pi) - math.pi

        # Check if the object is within the sensor's range and field of view but not ourself (distance > 0)
        if sensor.check_in_range_and_fov(angle_to_obj, distance) and distance > 0.001:
            # Calculate the detection probability
            detection_probability = sensor.detection_probability_at_distance(distance)
            probability = random.random()
            # Use this to determine if the object is detected
            # print(f"Detection Probability: {detection_probability}, Random Probability: {probability}")
            if probability <= detection_probability:
                # print(f"Object Detected: {gt_obj.vehicle_id}")
                # Calculate the detected bounding box with errors
                radial_error = sensor.centroid_radial_error_at_distance(distance)
                distance_error = sensor.centroid_distance_error_at_distance(distance)
                bbox_error = sensor.bounding_box_error_at_distance(distance)

                expected_error_gaussian, actual_error = calculateErrorGaussian(angle_to_obj, radial_error, distance_error)

                # Move the centroid using the actual error
                new_centroid_x = gt_obj.location[0] + actual_error[0]
                new_centroid_y = gt_obj.location[1] + actual_error[1]

                # Calculate the new bounding box error
                width_error, length_error = calculateBBoxError(bbox_error)

                # Calculate the size of the bounding box in ± width and ± length
                width = (gt_obj.dimensions[0] + width_error) / 2
                length = (gt_obj.dimensions[1] + length_error) / 2

                # Compute the global coordinates of the bounding box corners
                bbox_x_min = new_centroid_x - width
                bbox_x_max = new_centroid_x + width
                bbox_y_min = new_centroid_y - length
                bbox_y_max = new_centroid_y + length
                
                # Rotate the bounding box based on the vehicle's angle
                detected_bbox = [
                    utils.rotate_point(bbox_x_min, bbox_y_min, new_centroid_x, new_centroid_y, gt_obj.rotation),
                    utils.rotate_point(bbox_x_max, bbox_y_min, new_centroid_x, new_centroid_y, gt_obj.rotation),
                    utils.rotate_point(bbox_x_max, bbox_y_max, new_centroid_x, new_centroid_y, gt_obj.rotation),
                    utils.rotate_point(bbox_x_min, bbox_y_max, new_centroid_x, new_centroid_y, gt_obj.rotation)
                ]

                detected_objects.append(DetectedObject(
                    vehicle_id=f"{gt_obj.vehicle_id}_{detection_id}",
                    vehicle_type=gt_obj.type,
                    detected_bbox=detected_bbox,
                    centroid=(new_centroid_x, new_centroid_y),
                    width=gt_obj.dimensions[0],
                    length=gt_obj.dimensions[1],
                    angle=gt_obj.rotation,
                    expected_error_gaussian=expected_error_gaussian,
                    velocity_vector=gt_obj.velocity_vector
                ))

                detection_id += 1

    return detected_objects

def calculateErrorGaussian(target_line_angle, radial_error, distal_error):
    # Calculate our expected elipse error bounds
    elipse_angle_expected = target_line_angle
    
    expected_error_gaussian = gaussians.BivariateGaussian(distal_error**2,
                                                radial_error**2,
                                                elipse_angle_expected)

    # Calculate our expected errors in x,y coordinates
    actualRadialError = np.random.normal(0, radial_error, 1)[0]
    actualDistanceError = np.random.normal(0, distal_error, 1)[0]
    actual_error_gaussian = gaussians.BivariateGaussian(actualDistanceError**2,
                                                actualRadialError**2,
                                                elipse_angle_expected)
    actual_error = actual_error_gaussian.calc_xy_components()
    return expected_error_gaussian, actual_error

def calculateBBoxError(bbox_error):
    # Calculate our expected errors in width and length
    width_error = np.random.normal(0, bbox_error, 1)[0]
    length_error = np.random.normal(0, bbox_error, 1)[0]
    return (width_error, length_error)