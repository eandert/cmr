import math

class GroundTruthObject:
    def __init__(self, vehicle_id, vehicle_type, position, velocity_vector, bounding_box, angle, width, length):
        self.vehicle_id = vehicle_id
        self.type = vehicle_type
        self.bbox = bounding_box
        self.dimensions = [width, length]  # width, length
        self.location = [position[0], position[1]]  # 2D location
        self.rotation = angle
        self.velocity_vector = velocity_vector

    def __str__(self):
        return (f"Type: {self.type}, BBox: {self.bbox}, "
                f"Dimensions: {self.dimensions}, Location: {self.location}, "
                f"Rotation: {self.rotation}, Velocity Vector: {self.velocity_vector}")

def create_ground_truth(traci_instance, vehicle_ids):
    ground_truth = []

    # Iterate over each vehicle ID
    for veh_id in vehicle_ids:
        # Get the position of the vehicle
        position = traci_instance.vehicle.getPosition(veh_id)
        
        # Get the speed (velocity magnitude) of the vehicle
        speed = traci_instance.vehicle.getSpeed(veh_id)
        
        # Get the angle of the vehicle
        angle = traci_instance.vehicle.getAngle(veh_id)
        
        # Compute the velocity vector from speed and angle
        velocity_vector = (speed * math.cos(math.radians(angle)), speed * math.sin(math.radians(angle)))
        
        # Get the length and width of the vehicle to form the bounding box
        length = traci_instance.vehicle.getLength(veh_id)
        width = traci_instance.vehicle.getWidth(veh_id)
        
        # Get the type of the vehicle
        vehicle_type = traci_instance.vehicle.getTypeID(veh_id)
        
        # Compute the global coordinates of the bounding box corners
        bbox_x_min = position[0] - width / 2
        bbox_x_max = position[0] + width / 2
        bbox_y_min = position[1] - length / 2
        bbox_y_max = position[1] + length / 2
        bounding_box = [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]
        
        # Create an instance of GroundTruthObject and store it in the ground truth list
        ground_truth.append(GroundTruthObject(veh_id, vehicle_type, position, velocity_vector, bounding_box, angle, width, length))

    return ground_truth