from enum import Enum
import math

class SensorType(Enum):
    CAMERA = (1, 100, math.radians(90), 0)  # (id, max_range, horizontal_fov_degrees, center_angle_degrees)
    LIDAR = (2, 360,  math.radians(90), 0)
    RADAR = (3, 150,  math.radians(90), 0)
    OS1_128 = (4, 100,  math.radians(360), 0) # Changed ID to 4 to be unique

    def __init__(self, id, max_range, horizontal_fov_degrees, center_angle_degrees):
        self.id = id
        self.max_range = max_range
        self.horizontal_fov = horizontal_fov_degrees
        self.center_angle = center_angle_degrees