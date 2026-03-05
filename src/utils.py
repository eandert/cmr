import math
import random
import sensor_package
import re
import numpy as np
from error import ErrorPackage, ErrorType

# TraCI constants for subscriptions (import lazily to avoid circular imports)
_TC_CONSTANTS = None

def _get_tc_constants():
    """Lazily import traci.constants to avoid circular imports."""
    global _TC_CONSTANTS
    if _TC_CONSTANTS is None:
        import traci.constants as tc
        _TC_CONSTANTS = tc
    return _TC_CONSTANTS


def subscribe_to_vehicle(traci_instance, veh_id):
    """
    Subscribe to a vehicle's properties for efficient batched retrieval.
    This reduces TraCI overhead significantly.
    """
    tc = _get_tc_constants()
    try:
        traci_instance.vehicle.subscribe(veh_id, [
            tc.VAR_POSITION,
            tc.VAR_SPEED,
            tc.VAR_ANGLE
        ])
    except Exception:
        pass  # Vehicle might not exist yet


def unsubscribe_from_vehicle(traci_instance, veh_id):
    """Unsubscribe from a vehicle's properties."""
    # Only unsubscribe if vehicle still exists in simulation
    try:
        if veh_id in traci_instance.vehicle.getIDList():
            traci_instance.vehicle.unsubscribe(veh_id)
    except Exception:
        pass  # Vehicle might already be gone

# ============================================================================
# FAST ROTATED RECTANGLE IoU IMPLEMENTATION (replaces Shapely)
# ============================================================================

def get_rotated_box_corners(cx, cy, w, h, angle):
    """
    Get the 4 corners of a rotated bounding box.
    
    Args:
        cx, cy: Center coordinates
        w, h: Width and length (swapped from standard convention?)
        angle: Rotation angle in radians
    
    Returns:
        numpy array of shape (4, 2) containing corner coordinates
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Half dimensions
    # Assuming w is width (perpendicular to heading) and h is length (along heading)
    # The matching logic passes [width, length] to this function
    hw = w / 2.0
    hh = h / 2.0
    
    # Corners relative to center (before rotation)
    # If angle is heading (East=0), then length (h) should be along x-axis
    # But this function treats w as x-axis dimension and h as y-axis dimension at 0 rotation?
    # NO: Standard math box at 0 rotation has width along x and height along y.
    # IF we want length along heading (x-axis at 0), we need to swap w and h here
    # OR the caller must pass [length, width].
    
    # The caller (sensor_fusion.py) passes [width, length].
    # So w=width, h=length.
    # At 0 rotation (East), length should be along X.
    # Currently: w is along X, h is along Y. This means at 0 rotation, the box is "wide" not "long".
    # This is 90 degrees off.
    
    # FIX: Swap dimensions here to align length with heading (x-axis at 0)
    # Let's map w (width) to Y-axis and h (length) to X-axis
    
    # Corners relative to center:
    # x (along heading) = +/- length/2
    # y (perpendicular) = +/- width/2
    
    corners_rel = np.array([
        [-hh, -hw], # -length/2, -width/2
        [hh, -hw],  # +length/2, -width/2
        [hh, hw],   # +length/2, +width/2
        [-hh, hw]   # -length/2, +width/2
    ])
    
    # Rotation matrix
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate and translate corners
    corners = corners_rel @ rot_matrix.T + np.array([cx, cy])
    
    return corners


def polygon_area_signed(vertices):
    """
    Calculate the signed area of a polygon using the shoelace formula.
    Positive area = counter-clockwise winding
    Negative area = clockwise winding
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    
    vertices = np.asarray(vertices, dtype=np.float64)
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Signed area (positive for CCW, negative for CW)
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def polygon_area(vertices):
    """
    Calculate the absolute area of a polygon using the shoelace formula.
    
    Args:
        vertices: numpy array of shape (n, 2) containing polygon vertices
    
    Returns:
        Area of the polygon (always positive)
    """
    return abs(polygon_area_signed(vertices))


def ensure_ccw(vertices):
    """
    Ensure polygon vertices are in counter-clockwise order.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 3:
        return vertices
    
    # Check winding order using signed area
    signed_area = polygon_area_signed(vertices)
    
    # If clockwise (negative area), reverse the order
    if signed_area < 0:
        return vertices[::-1].copy()
    return vertices


def line_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two line segments.
    
    Args:
        p1, p2: Endpoints of first line segment
        p3, p4: Endpoints of second line segment
    
    Returns:
        Intersection point or None if no intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return np.array([x, y])


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Sutherland-Hodgman polygon clipping algorithm.
    Clips the subject polygon against the clip polygon.
    Both polygons are normalized to CCW order before clipping.
    
    Args:
        subject_polygon: numpy array of shape (n, 2) - polygon to be clipped
        clip_polygon: numpy array of shape (m, 2) - clipping polygon
    
    Returns:
        numpy array of clipped polygon vertices, or empty array if no intersection
    """
    def inside_edge(point, edge_start, edge_end):
        """Check if point is on the left side of the edge (inside for CCW polygon)."""
        return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - \
               (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) >= -1e-10
    
    # Ensure both polygons are CCW
    subject = ensure_ccw(subject_polygon)
    clip = ensure_ccw(clip_polygon)
    
    if len(subject) == 0 or len(clip) == 0:
        return np.array([])
    
    output = list(subject)
    
    for i in range(len(clip)):
        if len(output) == 0:
            return np.array([])
        
        input_list = output
        output = []
        
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        
        for j in range(len(input_list)):
            current = np.asarray(input_list[j], dtype=np.float64)
            previous = np.asarray(input_list[j - 1], dtype=np.float64)
            
            current_inside = inside_edge(current, edge_start, edge_end)
            previous_inside = inside_edge(previous, edge_start, edge_end)
            
            if current_inside:
                if not previous_inside:
                    intersection = line_intersection(previous, current, edge_start, edge_end)
                    if intersection is not None:
                        output.append(intersection)
                output.append(current)
            elif previous_inside:
                intersection = line_intersection(previous, current, edge_start, edge_end)
                if intersection is not None:
                    output.append(intersection)
    
    if len(output) < 3:
        return np.array([])
    
    return np.array(output, dtype=np.float64)


def rotated_box_iou(box_a, box_b):
    """
    Calculate IoU between two rotated bounding boxes.
    
    Args:
        box_a: tuple/list of (cx, cy, w, h, angle)
        box_b: tuple/list of (cx, cy, w, h, angle)
    
    Returns:
        IoU value between 0 and 1
    """
    try:
        # Get corners of both boxes
        corners_a = get_rotated_box_corners(box_a[0], box_a[1], box_a[2], box_a[3], box_a[4])
        corners_b = get_rotated_box_corners(box_b[0], box_b[1], box_b[2], box_b[3], box_b[4])
        
        # Calculate areas of both boxes
        area_a = box_a[2] * box_a[3]  # w * h
        area_b = box_b[2] * box_b[3]
        
        if area_a <= 0 or area_b <= 0:
            return 0.0
        
        # Find intersection polygon using Sutherland-Hodgman algorithm
        intersection = sutherland_hodgman_clip(corners_a, corners_b)
        
        if len(intersection) < 3:
            return 0.0
        
        # Calculate intersection area
        intersection_area = polygon_area(intersection)
        
        # Calculate union area
        union_area = area_a + area_b - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    except Exception:
        return 0.0


def computeDistanceBBox(a, b):
    """
    Compute distance metric (1 - IoU) between two rotated bounding boxes.
    This is a fast pure-numpy implementation that replaces the Shapely version.
    
    Args:
        a: tuple/array of (cx, cy, w, h, angle) for first box
        b: tuple/array of (cx, cy, w, h, angle) for second box
    
    Returns:
        Distance value between 0 (identical) and 1 (no overlap)
    """
    iou = rotated_box_iou(a, b)
    
    if iou <= 0:
        return 1.0
    else:
        return 1.0 - iou


def mahalanobis_distance(detection_pos, predicted_pos, S_inv):
    """
    Compute Mahalanobis distance between a detection and predicted track position.
    
    The Mahalanobis distance accounts for the covariance structure of the prediction
    uncertainty, providing a statistically-principled distance measure.
    
    d = sqrt((z - z_pred)^T * S^-1 * (z - z_pred))
    
    Args:
        detection_pos: [x, y] position of detection
        predicted_pos: [x, y] predicted position of track
        S_inv: 2x2 inverse of innovation covariance matrix S = H*P*H^T + R
    
    Returns:
        Mahalanobis distance (chi-squared distributed with 2 DOF if Gaussian)
    """
    residual = np.array([detection_pos[0] - predicted_pos[0], 
                         detection_pos[1] - predicted_pos[1]]).reshape(2, 1)
    d_squared_matrix = residual.T @ S_inv @ residual
    d_squared = float(d_squared_matrix[0, 0])
    return math.sqrt(max(0, d_squared))


def compute_innovation_covariance(P_pred, R):
    """
    Compute the innovation (residual) covariance matrix S = H*P*H^T + R.
    
    For position-only measurement (H extracts x,y from state), this simplifies
    to S = P_pred[0:2, 0:2] + R for the position submatrix.
    
    Args:
        P_pred: Predicted state covariance (at least 2x2 for position)
        R: Measurement noise covariance (2x2)
    
    Returns:
        S: 2x2 innovation covariance matrix
        S_inv: 2x2 inverse of S
    """
    # Extract position covariance (top-left 2x2)
    if P_pred.shape[0] >= 2:
        P_pos = P_pred[0:2, 0:2]
    else:
        P_pos = np.eye(2)
    
    S = P_pos + R
    
    # Ensure S is invertible by adding small regularization if needed
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Regularize if singular
        S_inv = np.linalg.inv(S + 1e-6 * np.eye(2))
    
    return S, S_inv


def compute_hybrid_cost(detection_pos, detection_cov, predicted_pos, P_pred, 
                        det_bbox, pred_bbox, 
                        mahal_weight=0.5, iou_weight=0.5,
                        mahal_gate=9.21):
    """
    Compute hybrid cost combining Mahalanobis distance and IOU for track association.
    
    This balances statistical distance (from Kalman covariance) with geometric
    overlap (bounding box IOU), providing robust matching even when boxes
    don't overlap but are statistically close.
    
    Args:
        detection_pos: [x, y] detection position
        detection_cov: 2x2 measurement covariance from detection
        predicted_pos: [x, y] predicted track position
        P_pred: Predicted state covariance matrix
        det_bbox: [x, y, w, h, angle] detection bounding box
        pred_bbox: [x, y, w, h, angle] predicted track bounding box
        mahal_weight: Weight for normalized Mahalanobis cost (0-1)
        iou_weight: Weight for IOU cost (0-1)
        mahal_gate: Mahalanobis distance gate threshold (default 9.21 = 99% chi-sq with 2 DOF)
    
    Returns:
        cost: Combined cost value (0 = perfect match, inf = gated out)
        mahal_dist: Raw Mahalanobis distance for debugging
        iou: IOU value for debugging
    """
    # Compute innovation covariance: S = P_pred + R (measurement noise)
    S, S_inv = compute_innovation_covariance(P_pred[0:2, 0:2] if P_pred.shape[0] >= 2 else P_pred, 
                                              detection_cov)
    
    # Mahalanobis distance
    mahal_dist = mahalanobis_distance(detection_pos, predicted_pos, S_inv)
    
    # Gate check: reject if Mahalanobis distance exceeds threshold
    # 9.21 corresponds to 99% confidence for chi-squared with 2 DOF
    # Return large finite value (not inf) for scipy's linear_sum_assignment
    if mahal_dist > mahal_gate:
        return 1e9, mahal_dist, 0.0
    
    # IOU (0 to 1, higher is better)
    iou = rotated_box_iou(det_bbox, pred_bbox)
    
    # Normalize Mahalanobis to 0-1 range using gate threshold
    mahal_normalized = mahal_dist / mahal_gate
    
    # IOU cost (1 - IOU, so 0 = perfect overlap)
    iou_cost = 1.0 - iou
    
    # Combined cost
    cost = mahal_weight * mahal_normalized + iou_weight * iou_cost
    
    return cost, mahal_dist, iou


def computeDistanceEuclidean(a, b):
    """
    Compute Euclidean-based distance if point a is inside rotated box b.
    Pure numpy implementation replacing Shapely.
    
    Args:
        a: tuple/array of (x, y, w, h, angle) - point is at (x, y)
        b: tuple/array of (cx, cy, w, h, angle) - box centered at (cx, cy)
    
    Returns:
        Normalized distance if point inside box, else 1.0
    """
    # Get corners of box b
    corners_b = get_rotated_box_corners(b[0], b[1], b[2], b[3], b[4])
    
    # Check if point a is inside polygon b using ray casting
    point = np.array([a[0], a[1]])
    
    if point_in_polygon(point, corners_b):
        distance = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        return distance / 100.0
    else:
        return 1.0


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: numpy array of shape (2,) - the point to check
        polygon: numpy array of shape (n, 2) - polygon vertices
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    n = len(polygon)
    inside = False
    
    x, y = point
    j = n - 1
    
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


# ============================================================================
# VEHICLE AND TRAFFIC LIGHT MANAGERS
# ============================================================================

class VehicleProbabilityManager:
    """
    Manages the probability of vehicles being classified as a specific type (e.g., CAV).
    """

    def __init__(self, probability, type, sumo_type, sensor_packages=None, error_package=None):
        """
        Initializes the VehicleProbabilityManager.

        Args:
            probability (float): The probability of a vehicle being classified as the specified type.
            type (str): The type of vehicle (e.g., "CAV").
            sumo_type (str): The SUMO vehicle type to set (e.g., "CAV_passenger").
            sensor_packages (list): A list of tuples containing the probability, sensors, and sensors' extrinsics.
            error_package (ErrorPackage): The error package to apply errors.
        """
        self.probability = probability
        self.vehicle_list = []
        self.total_vehicles = 0
        self.checked_vehicle_id_list = []
        self.type = type
        self.sumo_type = sumo_type
        self.vehicle_instances = {}
        self.sensor_packages = sensor_packages if sensor_packages else []
        self.error_package = error_package

    def update_vehicles(self, traci_instance, vehicle_id_list):
        """
        Updates the list of vehicles and changes the type of new vehicles based on the specified probability.

        Args:
            traci_instance: The TraCI instance to use for interacting with the simulation.
            vehicle_id_list (list): A list of vehicle IDs currently in the simulation.
        """
        # Get the IDs of all vehicles that have appeared in the simulation since the last step
        new_vehicle_id_list = [x for x in vehicle_id_list if x not in self.checked_vehicle_id_list]

        for vehicle_id in new_vehicle_id_list:
            # Subscribe to all new vehicles for efficient data retrieval
            subscribe_to_vehicle(traci_instance, vehicle_id)
            
            randomnum = random.random()
            if self.probability != 0 and randomnum <= self.probability:
                try:
                    self.total_vehicles += 1
                    self.vehicle_list.append(vehicle_id)
                    # Change the vehicle type to the specified SUMO type
                    traci_instance.vehicle.setType(vehicle_id, self.sumo_type)
                    # Create a new instance of the vehicle class with a selected sensor package
                    if self.sensor_packages:
                        sp = self.select_sensor_package()
                        # Determine ONCE if this CAV should have errors (10% probability)
                        # This flag is permanent for the lifetime of this CAV
                        from error import ErrorPackage
                        has_error = ErrorPackage.should_cav_have_error() if self.error_package else False
                        self.vehicle_instances[vehicle_id] = sensor_package.SensorPackage(
                            vehicle_id, sp[0], sp[1], sp[2], self.error_package, has_error=has_error)
                except Exception as e:
                    print(f"ERROR: Couldn't add {self.type}: ", e)

        self.checked_vehicle_id_list += new_vehicle_id_list

        # Remove vehicles that are no longer in the simulation
        removed_vehicle_id_list = [x for x in self.checked_vehicle_id_list if x not in vehicle_id_list]
        for vehicle_id in removed_vehicle_id_list:
            # Note: Don't unsubscribe - SUMO automatically cleans up subscriptions when vehicles leave
            # Calling unsubscribe on departed vehicles causes TraCI errors
            
            if vehicle_id in self.vehicle_list:
                self.vehicle_list.remove(vehicle_id)
                self.total_vehicles -= 1
                # Delete the vehicle instance
                if vehicle_id in self.vehicle_instances:
                    del self.vehicle_instances[vehicle_id]

        self.checked_vehicle_id_list = list(vehicle_id_list)

    def select_sensor_package(self):
        """
        Selects a sensor package based on the distribution within the sensor_packages tuple.

        Returns:
            tuple: The selected sensor package (sensors, sensors_extrinsics, localizer).
        """
        total_probability = sum(package[0] for package in self.sensor_packages)
        randomnum = random.uniform(0, total_probability)
        cumulative_probability = 0.0

        for probability, sensors, sensors_extrinsics, localizer_instance in self.sensor_packages:
            cumulative_probability += probability
            if randomnum <= cumulative_probability:
                # Pass the error_package to the Localizer constructor if needed
                if hasattr(localizer_instance, '__init__') and 'error_package' in localizer_instance.__init__.__code__.co_varnames:
                    # Create a new Localizer instance with the error_package
                    localizer_instance = localizer_instance.__class__(
                        localizer_instance.lateral_error_polynomial.coefficients, 
                        localizer_instance.longitudinal_error_polynomial.coefficients, 
                        self.error_package)
                return sensors, sensors_extrinsics, localizer_instance

        # Fallback to the last package, ensuring error_package is passed to Localizer if applicable
        last_package = self.sensor_packages[-1]
        sensors, sensors_extrinsics, localizer_instance = last_package[1], last_package[2], last_package[3]
        if hasattr(localizer_instance, '__init__') and 'error_package' in localizer_instance.__init__.__code__.co_varnames:
            localizer_instance = localizer_instance.__class__(
                localizer_instance.lateral_error_polynomial.coefficients, 
                localizer_instance.longitudinal_error_polynomial.coefficients, 
                self.error_package)
        return sensors, sensors_extrinsics, localizer_instance

    def get_active_vehicle_ids(self):
        """
        Returns the list of vehicle instances of active vehicles that are of the specified type.

        Returns:
            list: A list of active vehicle instances that are of the specified type.
        """
        return list(self.vehicle_instances.keys())
    
    def get_active_vehicle_instances(self):
        """
        Returns the list of vehicle instances of active vehicles that are of the specified type.

        Returns:
            list: A list of active vehicle instances that are of the specified type.
        """
        return list(self.vehicle_instances.values())


class TrafficLightProbabilityManager:
    """
    Manages the probability of traffic lights being outfitted with cameras.
    """

    def __init__(self, probability, traci_instance, sensor_packages=None, error_package=None):
        """
        Initializes the TrafficLightProbabilityManager.

        Args:
            probability (float): The probability of a traffic light being outfitted with an RSU.
            sensor_packages (list): A list of tuples containing the probability, sensors, and sensors' extrinsics.
            error_package (ErrorPackage): The error package to apply errors.
        """
        self.probability = probability
        self.tracked_tfls = {}
        self.tracked_tfl_total = 0
        self.tracked_tfl_total_possible = 0
        self.sensor_packages = sensor_packages if sensor_packages else []
        self.position = None
        self.error_package = error_package

        # Initialize the traffic lights once
        self.initialize_traffic_lights(traci_instance)

    def initialize_traffic_lights(self, traci_instance):
        """
        Initializes the list of traffic lights and adds cameras based on the specified probability.
        """
        # Get the list of traffic light IDs
        tfl = traci_instance.trafficlight.getIDList()

        for light in tfl:
            if light.find("joined") == -1:
                self.tracked_tfl_total_possible += 1
                # Add to list based on probability
                if random.random() <= self.probability:
                    self.position = traci_instance.junction.getPosition(light.replace("GS_", "", 1))
                    if self.sensor_packages:
                        sp = self.select_sensor_package()
                        self.tracked_tfls[light] = sensor_package.SensorPackage(light, sp[0], sp[1], sp[2], self.error_package)
                    else:
                        self.tracked_tfls[light] = None
                    self.tracked_tfl_total += 1

    def select_sensor_package(self):
        """
        Selects a sensor package based on the distribution within the sensor_packages tuple.

        Returns:
            tuple: The selected sensor package (sensors, sensors_extrinsics, localizer).
        """
        total_probability = sum(package[0] for package in self.sensor_packages)
        randomnum = random.uniform(0, total_probability)
        cumulative_probability = 0.0

        for probability, sensors, sensors_extrinsics, localizer_instance in self.sensor_packages:
            cumulative_probability += probability
            if randomnum <= cumulative_probability:
                # Pass the error_package to the Localizer constructor if needed
                if hasattr(localizer_instance, '__init__') and 'error_package' in localizer_instance.__init__.__code__.co_varnames:
                    localizer_instance = localizer_instance.__class__(
                        localizer_instance.lateral_error_polynomial.coefficients, 
                        localizer_instance.longitudinal_error_polynomial.coefficients, 
                        self.error_package)
                return sensors, sensors_extrinsics, localizer_instance

        # Fallback to the last package, ensuring error_package is passed to Localizer if applicable
        last_package = self.sensor_packages[-1]
        sensors, sensors_extrinsics, localizer_instance = last_package[1], last_package[2], last_package[3]
        if hasattr(localizer_instance, '__init__') and 'error_package' in localizer_instance.__init__.__code__.co_varnames:
            localizer_instance = localizer_instance.__class__(
                localizer_instance.lateral_error_polynomial.coefficients, 
                localizer_instance.longitudinal_error_polynomial.coefficients, 
                self.error_package)
        return sensors, sensors_extrinsics, localizer_instance

    def get_tracked_traffic_light_ids(self):
        """
        Returns the list of tracked traffic light IDs.

        Returns:
            list: A list of tracked traffic light IDs.
        """
        return list(self.tracked_tfls.keys())

    def get_tracked_traffic_light_instances(self):
        """
        Returns the list of tracked traffic light instances.

        Returns:
            list: A list of tracked traffic light instances.
        """
        return list(self.tracked_tfls.values())

    def get_tracked_traffic_light_total(self):
        """
        Returns the total number of tracked traffic lights.

        Returns:
            int: The total number of tracked traffic lights.
        """
        return self.tracked_tfl_total

    def get_tracked_traffic_light_total_possible(self):
        """
        Returns the total number of possible tracked traffic lights.

        Returns:
            int: The total number of possible tracked traffic lights.
        """
        return self.tracked_tfl_total_possible


# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

class Polynomial:
    """
    Represents a polynomial with a list of coefficients.
    """

    def __init__(self, coefficients):
        """
        Initializes the Polynomial.

        Args:
            coefficients (list): A list of coefficients for the polynomial.
        """
        self.coefficients = coefficients  # List of coefficients

    def __str__(self):
        """
        Returns a string representation of the polynomial.

        Returns:
            str: The string representation of the polynomial.
        """
        terms = []
        for power, coeff in enumerate(self.coefficients):
            if coeff != 0:
                terms.append(f"{coeff}*x^{power}")
        return " + ".join(terms)

    def evaluate(self, x):
        """
        Evaluates the polynomial at a given value of x.

        Args:
            x (float): The value at which to evaluate the polynomial.

        Returns:
            float: The result of the polynomial evaluation.
        """
        result = 0
        for power, coeff in enumerate(self.coefficients):
            result += coeff * (x ** power)
        return result

    
def rotate_point(x, y, cx, cy, angle):
    """
    Rotate a point around a center by a given angle.
    
    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        cx (float): The x-coordinate of the center.
        cy (float): The y-coordinate of the center.
        angle (float): The angle in radians.
    
    Returns:
        tuple: The rotated point (x, y).
    """
    # Translate point to origin
    x -= cx
    y -= cy
    # Rotate point in the opposite direction
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle
    # Translate point back
    x_new += cx
    y_new += cy
    return x_new, y_new


def kalman_prediction(X_hat_t_1, P_t_1, F_t, B_t, U_t, Q_t):
    """Kalman filter prediction equation."""
    X_hat_t = F_t.dot(X_hat_t_1) + (B_t.dot(U_t).reshape(B_t.shape[0], -1))
    P_t = F_t.dot(P_t_1).dot(F_t.transpose()) + Q_t
    return X_hat_t, P_t


def kalman_batch_update(X_hat_t, P_t, measurements, H_t_func):
    """
    Information filter batch update for multiple measurements.
    
    This is order-independent: all measurements contribute equally regardless of
    processing order. This prevents low-covariance measurements from dominating
    and causing overconfidence before other measurements are considered.
    
    Uses the information filter formulation:
        Y = P^-1  (information matrix)
        y = P^-1 * x  (information vector)
        
    Batch update (additive):
        Y_new = Y_prior + sum(H_i^T * R_i^-1 * H_i)
        y_new = y_prior + sum(H_i^T * R_i^-1 * z_i)
        
    Then convert back:
        P_new = Y_new^-1
        x_new = P_new * y_new
    
    Args:
        X_hat_t: Prior state estimate (n x 1)
        P_t: Prior covariance (n x n)
        measurements: List of (Z_t, R_t, h_t_type) tuples where:
            - Z_t: measurement vector (m x 1)
            - R_t: measurement covariance (m x m)
            - h_t_type: type parameter for H_t_func
        H_t_func: Function that returns H matrix given h_t_type
    
    Returns:
        X_new, P_new: Updated state and covariance
    """
    if not measurements:
        return X_hat_t, P_t
    
    # Check for NaN/Inf in prior
    if np.any(np.isnan(P_t)) or np.any(np.isinf(P_t)):
        return X_hat_t, P_t
    if np.any(np.isnan(X_hat_t)) or np.any(np.isinf(X_hat_t)):
        return X_hat_t, P_t
    
    n = X_hat_t.shape[0]
    
    # Convert to information form
    # Add regularization to ensure P is invertible
    min_eigenval = 1e-10
    P_reg = P_t + min_eigenval * np.eye(n)
    
    try:
        Y_prior = kalman_inverse(P_reg)  # Information matrix
        y_prior = Y_prior.dot(X_hat_t)   # Information vector
    except (np.linalg.LinAlgError, ValueError):
        # If inversion fails, fall back to sequential updates
        X_new, P_new = X_hat_t, P_t
        for Z_t, R_t, h_t_type in measurements:
            X_new, P_new = kalman_update(X_new, P_new, Z_t, R_t, H_t_func(h_t_type))
        return X_new, P_new
    
    # Accumulate information from all measurements (order-independent)
    Y_accum = np.zeros_like(Y_prior)
    y_accum = np.zeros_like(y_prior)
    
    for Z_t, R_t, h_t_type in measurements:
        # Skip invalid measurements
        if np.any(np.isnan(Z_t)) or np.any(np.isnan(R_t)):
            continue
        if np.any(np.isinf(Z_t)) or np.any(np.isinf(R_t)):
            continue
        
        H_t = H_t_func(h_t_type)
        
        # Add regularization to R for invertibility
        R_reg = R_t + min_eigenval * np.eye(R_t.shape[0])
        
        try:
            R_inv = kalman_inverse(R_reg)
        except (np.linalg.LinAlgError, ValueError):
            continue  # Skip this measurement if R is singular
        
        # Accumulate information: H^T * R^-1 * H and H^T * R^-1 * z
        HtRinv = H_t.T.dot(R_inv)
        Y_accum += HtRinv.dot(H_t)
        y_accum += HtRinv.dot(Z_t)
    
    # Combine prior and measurement information
    Y_new = Y_prior + Y_accum
    y_new = y_prior + y_accum
    
    # Convert back to state space form
    try:
        P_new = kalman_inverse(Y_new)
        X_new = P_new.dot(y_new)
    except (np.linalg.LinAlgError, ValueError):
        # If inversion fails, fall back to sequential updates
        X_new, P_new = X_hat_t, P_t
        for Z_t, R_t, h_t_type in measurements:
            X_new, P_new = kalman_update(X_new, P_new, Z_t, R_t, H_t_func(h_t_type))
        return X_new, P_new
    
    # Ensure covariance stays symmetric and positive definite
    P_new = (P_new + P_new.T) / 2
    
    return X_new, P_new


def kalman_update(X_hat_t, P_t, Z_t, R_t, H_t):
    """Kalman filter update equation with numerical stability protection."""
    # Check for NaN/Inf in inputs
    if np.any(np.isnan(P_t)) or np.any(np.isnan(R_t)) or np.any(np.isnan(Z_t)):
        # Skip update if inputs contain NaN - return prediction as-is
        return X_hat_t, P_t
    
    if np.any(np.isinf(P_t)) or np.any(np.isinf(R_t)) or np.any(np.isinf(Z_t)):
        # Skip update if inputs contain Inf - return prediction as-is
        return X_hat_t, P_t
    
    # Calculate innovation covariance
    S = H_t.dot(P_t).dot(H_t.transpose()) + R_t
    
    # Add small regularization to prevent singularity
    # Reduced from 1e-6 to 1e-11 to avoid limiting high-precision sensor benefits (GPEM)
    min_eigenval = 1e-11
    S = S + min_eigenval * np.eye(S.shape[0])
    
    try:
        K_prime = P_t.dot(H_t.transpose()).dot(kalman_inverse(S))
    except (np.linalg.LinAlgError, ValueError):
        # If inversion fails, skip the update
        return X_hat_t, P_t
    
    X_t = X_hat_t + K_prime.dot(Z_t - H_t.dot(X_hat_t))
    P_t_new = P_t - K_prime.dot(H_t).dot(P_t)
    
    # Ensure covariance stays positive definite
    P_t_new = (P_t_new + P_t_new.T) / 2  # Symmetrize
    
    return X_t, P_t_new


def kalman_inverse(m):
    """Inverse function that is better than the default numpy one."""
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")
    
    # Check for NaN/Inf
    if np.any(np.isnan(m)) or np.any(np.isinf(m)):
        raise ValueError("Matrix contains NaN or Inf")
    
    i = np.eye(a, a)
    try:
        result = np.linalg.lstsq(m, i, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if lstsq fails
        result = np.linalg.pinv(m)
    
    return result


def ellipsify(covariance, num_std_deviations=3.0):
    """Convert covariance matrix to ellipse parameters."""
    # Defensive: ensure covariance is a 2x2 numpy array
    if covariance is None:
        return 0.0, 0.0, 0.0
    cov = np.asarray(covariance, dtype='float')
    if cov.ndim != 2 or cov.shape[0] != 2 or cov.shape[1] != 2:
        # If 3x3 or larger, extract the 2x2 position submatrix
        if cov.ndim == 2 and cov.shape[0] >= 2 and cov.shape[1] >= 2:
            cov = cov[0:2, 0:2]
        else:
            return 0.0, 0.0, 0.0
    # Eigenvalue and eigenvector computations
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Use the eigenvalue to figure out which direction is larger
    phi = np.arctan2(*vecs[:, 0][::-1])

    if not np.any(np.isnan(vals)) and np.all(np.isfinite(vals)):
        a, b = num_std_deviations * np.sqrt(vals)
    elif not np.isnan(vals[0]) and np.isfinite(vals[0]):
        a = num_std_deviations * np.sqrt(vals[0])
        b = 0.0
    elif not np.isnan(vals[1]) and np.isfinite(vals[1]):
        b = num_std_deviations * np.sqrt(vals[1])
        a = 0.0
    else:
        a = 0.0
        b = 0.0

    return a, b, phi


def calculateRadiusAtAngle(a, b, phi, theta):
    """Calculate the radius of an ellipse at a given angle.
    Standard polar form: r = ab / sqrt((b*cos(theta-phi))^2 + (a*sin(theta-phi))^2)
    """
    cos_diff = math.cos(theta - phi)
    sin_diff = math.sin(theta - phi)
    
    if a == 0 and b == 0:
        return 0.0
    
    denominator = (b * cos_diff)**2 + (a * sin_diff)**2
    if denominator <= 0:
        return 0.0
    
    return (a * b) / math.sqrt(denominator)


def extract_id(string_id):
    """
    Extracts the integer value from the cav_id string.
    
    Args:
        cav_id (str): The ID of the CAV.
    
    Returns:
        int: The extracted integer value.
    """
    match = re.search(r'\d+', string_id)
    return int(match.group()) if match else None
