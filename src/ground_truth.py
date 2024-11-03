import math
import utils

class GroundTruthObject:
    def __init__(self, vehicle_id, vehicle_type, position, velocity_vector, bounding_box, angle_rad, width, length):
        self.vehicle_id = vehicle_id
        self.type = vehicle_type
        self.bbox = bounding_box
        self.dimensions = [width, length]  # width, length
        self.location = [position[0], position[1]]  # 2D location
        self.rotation = angle_rad  # Store angle in radians
        self.velocity_vector = velocity_vector

    def __str__(self):
        return (f"Type: {self.type}, BBox: {self.bbox}, "
                f"Dimensions: {self.dimensions}, Location: {self.location}, "
                f"Rotation: {math.degrees(self.rotation)}°, Velocity Vector: {self.velocity_vector}")

    def draw_bounding_box_in_sumo(self, traci_instance, color=(255, 0, 0, 255), layer=10):
        """
        Draw the bounding box of this ground truth object in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the bounding box in RGBA format. Defaults to red.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 10.
        """
        bbox = self.bbox
        vehicle_id = self.vehicle_id

        # Draw the bounding box as a polygon in SUMO
        traci_instance.polygon.add(
            polygonID=f"bbox_{vehicle_id}",
            shape=bbox,
            color=color,
            fill=True,
            layer=layer
        )

    def draw_position_vector_in_sumo(self, traci_instance, color=(0, 255, 0, 255), length=5, layer=11):
        """
        Draw the position vector from the centroid and angle of the bounding box in SUMO.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            color (tuple): The color of the position vector in RGBA format. Defaults to green.
            length (float): The length of the position vector. Defaults to 5.
            layer (int): The layer to draw the polygon on. Higher values are drawn on top of lower values. Defaults to 11.
        """
        cx, cy = self.location
        angle_rad = self.rotation - math.pi / 2  # Adjusting the angle by -90 degrees
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

def create_ground_truth_by_id(traci_instance, veh_id):
    """
    Create ground truth data for a single vehicle ID.
    
    Args:
        traci_instance: The TraCI instance to use for retrieving vehicle data.
        veh_id (str): The vehicle ID.
    
    Returns:
        GroundTruthObject: An instance of GroundTruthObject.
    """
    # Get the position of the vehicle (center of the front bumper)
    front_bumper_position = traci_instance.vehicle.getPosition(veh_id)
    
    # Get the speed (velocity magnitude) of the vehicle
    speed = traci_instance.vehicle.getSpeed(veh_id)
    
    # Get the angle of the vehicle in degrees
    angle_deg = traci_instance.vehicle.getAngle(veh_id)
    
    # Compute the velocity vector from speed and angle
    angle_rad = math.radians(-angle_deg)
    velocity_vector = (speed * math.cos(angle_rad), speed * math.sin(angle_rad))
    
    # Get the length and width of the vehicle to form the bounding box
    length = traci_instance.vehicle.getLength(veh_id)
    width = traci_instance.vehicle.getWidth(veh_id)
    
    # Get the type of the vehicle
    vehicle_type = traci_instance.vehicle.getTypeID(veh_id)

    # Adjust the position to the center of the vehicle
    cx = front_bumper_position[0] - (length / 2) * math.cos(angle_rad + math.pi / 2)
    cy = front_bumper_position[1] - (length / 2) * math.sin(angle_rad + math.pi / 2)
    
    # Compute the global coordinates of the bounding box corners
    bbox_x_min = cx - width / 2
    bbox_x_max = cx + width / 2
    bbox_y_min = cy - length / 2
    bbox_y_max = cy + length / 2
    
    # Rotate the bounding box based on the vehicle's angle
    bbox_rotated = [
        utils.rotate_point(bbox_x_min, bbox_y_min, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_max, bbox_y_min, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_max, bbox_y_max, cx, cy, angle_rad),
        utils.rotate_point(bbox_x_min, bbox_y_max, cx, cy, angle_rad)
    ]
    
    # Create and return an instance of GroundTruthObject
    return GroundTruthObject(veh_id, vehicle_type, (cx, cy), velocity_vector, bbox_rotated, angle_rad + math.pi, width, length)

def create_ground_truth_from_list(traci_instance, vehicle_ids):
    """
    Create ground truth data for the given vehicle IDs.
    
    Args:
        traci_instance: The TraCI instance to use for retrieving vehicle data.
        vehicle_ids (list): A list of vehicle IDs.
    
    Returns:
        list: A list of GroundTruthObject instances.
    """
    ground_truth = []

    # Iterate over each vehicle ID
    for veh_id in vehicle_ids:
        ground_truth.append(create_ground_truth_by_id(traci_instance, veh_id))

    return ground_truth

def filter_ground_truth_by_range(ground_truth_objects, x, y, range):
    """
    Filter ground truth objects by their distance to a given (x, y) position.
    
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
        distance = math.sqrt((obj.location[0] - x)**2 + (obj.location[1] - y)**2)
        if distance <= range:
            filtered_objects.append(obj)
    return filtered_objects