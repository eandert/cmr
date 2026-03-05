import math
import utils
import numpy as np
from scipy.spatial import cKDTree

# Cache for vehicle static properties (optimization #6)
# These properties (type, length, width) don't change during simulation
_vehicle_static_cache = {}

# Cache for traffic light positions (they never move)
_traffic_light_position_cache = {}

def clear_vehicle_static_cache():
    """
    Clear the vehicle static properties cache.
    Call this when vehicles are removed from the simulation.
    """
    global _vehicle_static_cache
    _vehicle_static_cache.clear()

def cleanup_vehicle_cache(active_vehicle_ids):
    """
    Remove cached entries for vehicles that are no longer in the simulation.
    This prevents memory leaks from vehicles that have left.
    
    Args:
        active_vehicle_ids (set or list): Set of currently active vehicle IDs.
    """
    global _vehicle_static_cache
    active_set = set(active_vehicle_ids)
    # Remove entries for vehicles that are no longer active
    keys_to_remove = [veh_id for veh_id in _vehicle_static_cache.keys() if veh_id not in active_set]
    for veh_id in keys_to_remove:
        del _vehicle_static_cache[veh_id]


def clear_traffic_light_cache():
    """Clear the traffic light position cache."""
    global _traffic_light_position_cache
    _traffic_light_position_cache.clear()


class GroundTruthObject:
    def __init__(self, vehicle_id, vehicle_type, position, velocity_vector, bounding_box, angle_rad, width, length):
        self.vehicle_id = vehicle_id
        self.type = vehicle_type
        self.bbox = bounding_box
        self.dimensions = [width, length]  # width, length
        self.centroid = [position[0], position[1]]  # 2D location
        self.angle = angle_rad  # Store angle in radians
        self.velocity_vector = velocity_vector

    def __str__(self):
        return (f"Type: {self.type}, BBox: {self.bbox}, "
                f"Dimensions: {self.dimensions}, Location: {self.centroid}, "
                f"Rotation: {math.degrees(self.angle)}°, Velocity Vector: {self.velocity_vector}")

    def draw_bounding_box_in_sumo(self, traci_instance, color=(255, 0, 0, 255), layer=10, polygon_id_prefix=""):
        """
        Draw the bounding box of this ground truth object in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the bounding box in RGBA format. Defaults to red.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 10.
            polygon_id_prefix (str): Prefix to make the polygon ID unique across different types of visualizations.
        """
        bbox = self.bbox
        vehicle_id = self.vehicle_id

        # Draw the bounding box as a polygon in SUMO
        try:
            polygon_unique_id = f"bbox_{polygon_id_prefix}_{vehicle_id}"
            traci_instance.polygon.add(
                polygonID=polygon_unique_id,
                shape=bbox,
                color=color,
                fill=True,
                layer=layer
            )
            return polygon_unique_id # Return the ID for management
        except Exception as e:
            print(f"Error drawing bounding box for vehicle {vehicle_id}: {e}") # Debug print
            return None

    def draw_position_vector_in_sumo(self, traci_instance, color=(0, 255, 0, 255), length=5, layer=11, polygon_id_prefix=""):
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
        angle_rad = self.angle # Already in standard math convention
        vehicle_length = self.dimensions[1] # Get vehicle length

        # Start from the center
        # end point is forward in the direction of heading
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
        return polygon_vector_id


def setup_vehicle_subscriptions(traci_instance):
    """
    Set up TraCI subscriptions for efficient batched vehicle data retrieval.
    Call this once at simulation start.
    """
    import traci.constants as tc
    # Subscribe to simulation-level vehicle ID list
    traci_instance.simulation.subscribe([tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS])


def subscribe_to_vehicle(traci_instance, veh_id):
    """
    Subscribe to a vehicle's dynamic properties for efficient retrieval.
    """
    import traci.constants as tc
    traci_instance.vehicle.subscribe(veh_id, [
        tc.VAR_POSITION,
        tc.VAR_SPEED,
        tc.VAR_ANGLE,
        tc.VAR_LENGTH,
        tc.VAR_WIDTH,
        tc.VAR_TYPE
    ])


def get_all_vehicle_data_batched(traci_instance, vehicle_ids):
    """
    Get all vehicle data using batched TraCI calls.
    This is MUCH faster than individual getPosition/getSpeed/getAngle calls.
    
    Args:
        traci_instance: The TraCI instance.
        vehicle_ids: List of vehicle IDs.
    
    Returns:
        dict: {veh_id: {'position': (x, y), 'speed': float, 'angle': float, 'length': float, 'width': float, 'type': str}}
    """
    import traci.constants as tc
    
    if not vehicle_ids:
        return {}
    
    result = {}
    
    # Get subscription results for all vehicles
    for veh_id in vehicle_ids:
        try:
            sub_results = traci_instance.vehicle.getSubscriptionResults(veh_id)
            if sub_results:
                result[veh_id] = {
                    'position': sub_results.get(tc.VAR_POSITION, (0, 0)),
                    'speed': sub_results.get(tc.VAR_SPEED, 0),
                    'angle': sub_results.get(tc.VAR_ANGLE, 0),
                    'length': sub_results.get(tc.VAR_LENGTH, 5),
                    'width': sub_results.get(tc.VAR_WIDTH, 2),
                    'type': sub_results.get(tc.VAR_TYPE, 'unknown')
                }
            else:
                # Fallback to individual calls if subscription failed
                result[veh_id] = {
                    'position': traci_instance.vehicle.getPosition(veh_id),
                    'speed': traci_instance.vehicle.getSpeed(veh_id),
                    'angle': traci_instance.vehicle.getAngle(veh_id),
                    'length': traci_instance.vehicle.getLength(veh_id),
                    'width': traci_instance.vehicle.getWidth(veh_id),
                    'type': traci_instance.vehicle.getTypeID(veh_id)
                }
        except Exception:
            # Vehicle might have left simulation
            pass
    
    return result


def create_ground_truth_for_vehicle_by_id(traci_instance, veh_id):
    """
    Create ground truth data for a single vehicle ID.
    Optimized with caching and subscription results.
    
    Args:
        traci_instance: The TraCI instance to use for retrieving vehicle data.
        veh_id (str): The vehicle ID.
    
    Returns:
        GroundTruthObject: An instance of GroundTruthObject.
    """
    import traci.constants as tc
    
    # Try to get data from subscription results first (much faster)
    sub_results = None
    try:
        sub_results = traci_instance.vehicle.getSubscriptionResults(veh_id)
    except Exception:
        pass
    
    if sub_results and tc.VAR_POSITION in sub_results:
        # Use subscription data (fast path)
        front_bumper_position = sub_results[tc.VAR_POSITION]
        speed = sub_results.get(tc.VAR_SPEED, 0)
        angle_deg = sub_results.get(tc.VAR_ANGLE, 0)
    else:
        # Fall back to individual TraCI calls (slow path)
        front_bumper_position = traci_instance.vehicle.getPosition(veh_id)
        speed = traci_instance.vehicle.getSpeed(veh_id)
        angle_deg = traci_instance.vehicle.getAngle(veh_id)
    
    # Compute the velocity vector from speed and angle
    # Standard math angle: 0 is East, 90 is North
    angle_rad = math.radians(90 - angle_deg)
    velocity_vector = (speed * math.cos(angle_rad), speed * math.sin(angle_rad))
    
    # Optimization #6: Cache static vehicle properties (type, length, width)
    # These don't change during simulation, so we cache them to avoid repeated TraCI calls
    if veh_id not in _vehicle_static_cache:
        length = traci_instance.vehicle.getLength(veh_id)
        width = traci_instance.vehicle.getWidth(veh_id)
        vehicle_type = traci_instance.vehicle.getTypeID(veh_id)
        _vehicle_static_cache[veh_id] = (length, width, vehicle_type)
    else:
        length, width, vehicle_type = _vehicle_static_cache[veh_id]

    # Adjust the position to the center of the vehicle
    cx, cy = front_bumper_position
    # Note: bumper position is already provided by SUMO. 
    # If we need center, we should offset by half length in the opposite direction of heading.
    half_length = length / 2
    cx = front_bumper_position[0] - half_length * math.cos(angle_rad)
    cy = front_bumper_position[1] - half_length * math.sin(angle_rad)
    
    # Compute the global coordinates of the bounding box corners (axis-aligned at 0 rad)
    # At 0 radians (East), length is along x and width is along y.
    half_width = width / 2
    bbox_x_min = cx - half_length
    bbox_x_max = cx + half_length
    bbox_y_min = cy - half_width
    bbox_y_max = cy + half_width
    
    # Rotate the bounding box based on the vehicle's angle
    bbox_rotated = [
        utils.rotate_point(bbox_x_min, bbox_y_min, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_max, bbox_y_min, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_max, bbox_y_max, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_min, bbox_y_max, cx, cy, angle_rad)
    ]
    
    # Create and return an instance of GroundTruthObject
    # Use standard math angle convention
    return GroundTruthObject(veh_id, vehicle_type, (cx, cy), velocity_vector, bbox_rotated, angle_rad, width, length)


def create_ground_truth_from_data(veh_id, position, speed, angle_deg, length, width, vehicle_type):
    """
    Create ground truth from pre-fetched data (for batched operations).
    This avoids any TraCI calls.
    """
    # Compute the velocity vector from speed and angle
    # Standard math angle: 0 is East, 90 is North
    angle_rad = math.radians(90 - angle_deg)
    velocity_vector = (speed * math.cos(angle_rad), speed * math.sin(angle_rad))
    
    # Adjust the position to the center of the vehicle
    half_length = length / 2
    cx = position[0] - half_length * math.cos(angle_rad)
    cy = position[1] - half_length * math.sin(angle_rad)
    
    # Compute the global coordinates of the bounding box corners
    half_width = width / 2
    bbox_x_min = cx - half_length
    bbox_x_max = cx + half_length
    bbox_y_min = cy - half_width
    bbox_y_max = cy + half_width
    
    # Rotate the bounding box based on the vehicle's angle
    bbox_rotated = [
        utils.rotate_point(bbox_x_min, bbox_y_min, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_max, bbox_y_min, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_max, bbox_y_max, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_min, bbox_y_max, cx, cy, angle_rad)
    ]
    
    return GroundTruthObject(veh_id, vehicle_type, (cx, cy), velocity_vector, bbox_rotated, angle_rad, width, length)


def create_ground_truth_for_traffic_light_by_id(traci_instance, tl_id):
    """
    Create ground truth data for a single traffic light ID.
    Optimized with position caching (traffic lights don't move).
    
    Args:
        traci_instance: The TraCI instance to use for retrieving traffic light data.
        tl_id (str): The traffic light ID.
    
    Returns:
        GroundTruthObject: An instance of GroundTruthObject.
    """
    global _traffic_light_position_cache
    
    # Cache traffic light positions - they never change
    if tl_id not in _traffic_light_position_cache:
        position = traci_instance.junction.getPosition(tl_id.replace("GS_", "", 1))
        _traffic_light_position_cache[tl_id] = position
    else:
        position = _traffic_light_position_cache[tl_id]
    
    # Traffic lights do not have speed, angle, length, width, or type
    speed = 0
    angle_rad = 0
    velocity_vector = (0, 0)
    length = 100
    width = 100
    vehicle_type = "traffic_light"
    
    # Create an arbitrary bounding box of 100x100
    cx, cy = position
    bbox_rotated = [
        (cx - width / 2, cy - length / 2),
        (cx + width / 2, cy - length / 2),
        (cx + width / 2, cy + length / 2),
        (cx - width / 2, cy + length / 2)
    ]
    
    # Create and return an instance of GroundTruthObject
    return GroundTruthObject(tl_id, vehicle_type, position, velocity_vector, bbox_rotated, angle_rad, width, length)


def create_ground_truth_for_traffic_light_from_list(traci_instance, traffic_light_ids):
    """
    Create ground truth data for the given traffic light IDs.
    
    Args:
        traci_instance: The TraCI instance to use for retrieving vehicle data.
        traffic_light_ids (list): A list of traffic light IDs.
    
    Returns:
        list: A list of GroundTruthObject instances.
    """
    return [create_ground_truth_for_traffic_light_by_id(traci_instance, tl_id) for tl_id in traffic_light_ids]


def create_ground_truth_for_vehicle_from_list(traci_instance, vehicle_ids):
    """
    Create ground truth data for the given vehicle IDs.
    Optimized with caching and list comprehension (optimization #6).
    
    Args:
        traci_instance: The TraCI instance to use for retrieving vehicle data.
        vehicle_ids (list): A list of vehicle IDs.
    
    Returns:
        list: A list of GroundTruthObject instances.
    """
    # Optimization #6: Use list comprehension for faster list creation
    # Also cleanup cache for vehicles that are no longer active
    if vehicle_ids:
        cleanup_vehicle_cache(vehicle_ids)
        ground_truth = [create_ground_truth_for_vehicle_by_id(traci_instance, veh_id) for veh_id in vehicle_ids]
    else:
        ground_truth = []

    return ground_truth


def create_ground_truth_for_vehicle_from_list_batched(traci_instance, vehicle_ids):
    """
    Create ground truth data for vehicles using batched TraCI data.
    This is faster when subscriptions are set up.
    
    Args:
        traci_instance: The TraCI instance.
        vehicle_ids: List of vehicle IDs.
    
    Returns:
        list: List of GroundTruthObject instances.
    """
    if not vehicle_ids:
        return []
    
    cleanup_vehicle_cache(vehicle_ids)
    
    # Get all vehicle data in a batch
    all_data = get_all_vehicle_data_batched(traci_instance, vehicle_ids)
    
    ground_truth = []
    for veh_id in vehicle_ids:
        if veh_id in all_data:
            data = all_data[veh_id]
            gt = create_ground_truth_from_data(
                veh_id,
                data['position'],
                data['speed'],
                data['angle'],
                data['length'],
                data['width'],
                data['type']
            )
            ground_truth.append(gt)
    
    return ground_truth


def filter_ground_truth_by_range(ground_truth_objects, x, y, range):
    """
    Filter ground truth objects by their distance to a given (x, y) position.
    Uses linear search - for better performance with spatial indexing, use filter_ground_truth_by_range_indexed.
    
    Args:
        ground_truth_objects (list): A list of GroundTruthObject instances.
        x (float): The x-coordinate of the position.
        y (float): The y-coordinate of the position.
        range (float): The range within which to filter the ground truth objects.
    
    Returns:
        list: A subset of GroundTruthObject instances within the specified range.
    """
    filtered_objects = []
    for obj in ground_truth_objects:
        distance = math.sqrt((obj.centroid[0] - x)**2 + (obj.centroid[1] - y)**2)
        if distance <= range:
            filtered_objects.append(obj)
    return filtered_objects


def build_ground_truth_spatial_index(ground_truth_objects):
    """
    Build a spatial index (cKDTree) for fast range queries on ground truth objects.
    
    Args:
        ground_truth_objects (list): A list of GroundTruthObject instances.
    
    Returns:
        tuple: (cKDTree, list) - The spatial index tree and the original list of objects.
               Returns (None, ground_truth_objects) if the list is empty.
    """
    if not ground_truth_objects:
        return None, ground_truth_objects
    
    # Extract positions as numpy array for the spatial index
    positions = np.array([[gt.centroid[0], gt.centroid[1]] for gt in ground_truth_objects])
    tree = cKDTree(positions)
    return tree, ground_truth_objects


def filter_ground_truth_by_range_indexed(spatial_index, ground_truth_objects, x, y, max_range):
    """
    Filter ground truth objects by their distance to a given (x, y) position using spatial indexing.
    This is much faster than linear search for large numbers of objects.
    
    Args:
        spatial_index (cKDTree): The spatial index tree from build_ground_truth_spatial_index.
        ground_truth_objects (list): A list of GroundTruthObject instances (same order as when index was built).
        x (float): The x-coordinate of the position.
        y (float): The y-coordinate of the position.
        max_range (float): The maximum range within which to filter the ground truth objects.
    
    Returns:
        list: A subset of GroundTruthObject instances within the specified range.
    """
    if spatial_index is None or not ground_truth_objects:
        return []
    
    # Query the spatial index for all points within max_range
    query_point = np.array([x, y])
    indices = spatial_index.query_ball_point(query_point, max_range)
    
    # Return the corresponding ground truth objects
    return [ground_truth_objects[i] for i in indices]
