import math
import re
import ground_truth
import sensor
import local_sensor_fusion

class CAV:
    """
    Class representing a Connected and Autonomous Vehicle (CAV).
    
    Attributes:
        cav_id (str): The ID of the CAV.
        ground_truth_obj (GroundTruthObject): The ground truth object for the CAV.
        sensors (list): A list of sensor objects for the CAV.
        sensors_extrinsics (list): A list of extrinsic parameters for each sensor.
        sensor_detection_sets (list): A list of detection sets for each sensor.
        detection_set_timestamp (float): The timestamp of the detection set.
    """
    
    def __init__(self, cav_id, sensors, sensors_extrinsics):
        """
        Initialize the CAV with its ID, sensors, and sensors' extrinsics.
        
        Args:
            cav_id (str): The ID of the CAV.
            sensors (list): A list of sensor objects for the CAV.
            sensors_extrinsics (list): A list of extrinsic parameters for each sensor.
        """
        self.cav_id = cav_id
        self.ground_truth_obj = None
        self.sensors = sensors
        self.sensors_extrinsics = sensors_extrinsics
        self.sensor_detection_sets = []
        self.detection_set_timestamp = None
        self.sensor_fusion = local_sensor_fusion.FUSION(self.extract_id(cav_id))

    def extract_id(self, cav_id):
        """
        Extracts the integer value from the cav_id string.
        
        Args:
            cav_id (str): The ID of the CAV.
        
        Returns:
            int: The extracted integer value.
        """
        match = re.search(r'\d+', cav_id)
        return int(match.group()) if match else None

    def set_sensor_poses(self):
        """
        Set the sensor poses for the CAV based on the CAV's pose and the sensors' extrinsics.
        
        Returns:
            list: A list of sensor poses as (x, y, yaw).
        """
        cav_x, cav_y, cav_yaw = self.ground_truth_obj.centroid[0], self.ground_truth_obj.centroid[1], self.ground_truth_obj.rotation
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
        Create detection sets for each sensor in the CAV using the sensor poses.
        
        Args:
            ego_ground_truth (GroundTruthObject): The ground truth object for the ego vehicle.
            ground_truth_objects (list): A list of GroundTruthObject instances.
            time (float): The timestamp of the detection set.
        """
        self.ground_truth_obj = ego_ground_truth

        sensor_poses = self.set_sensor_poses()

        self.sensor_detection_sets = []

        for sensor_obj, sensor_pose in zip(self.sensors, sensor_poses):
            # Filter ground truth by range so that we don't do unnecessary calculations
            ground_truth_objects_filtered = ground_truth.filter_ground_truth_by_range(ground_truth_objects, sensor_pose[0], sensor_pose[1], sensor_obj.max_range)

            # Create detected objects
            detected_objects = sensor.create_detected_bounding_boxes(sensor_obj, sensor_pose, ground_truth_objects_filtered)

            self.sensor_detection_sets.append(detected_objects)

            # Process and match the detections from each sensor
            self.sensor_fusion.processDetectionFrame(time, detected_objects, 2, True)

        self.detection_set_timestamp = time

        # Kick off the actual fusion process
        result, vizualization = self.sensor_fusion.fuseDetectionFrame()
        return result, vizualization

    def draw_detected_objects(self, traci_instance, polygon_ids):
        """
        Draw all detected objects for each sensor in the CAV using the TraCI instance.
        
        Args:
            traci_instance: The TraCI instance to use for drawing.
            polygon_ids (list): A list to store the IDs of the drawn polygons.
        """
        for detected_objects in self.sensor_detection_sets:
            for detected_obj in detected_objects:
                if detected_obj.draw_bounding_box_in_sumo(traci_instance):
                    polygon_ids.append(f"bbox_{detected_obj.vehicle_id}")

# Example usage
# if __name__ == "__main__":
#     import traci

#     # Start SUMO in server mode
#     sumoBinary = "sumo-gui"  # or "sumo" if you don't need the GUI
#     sumoCmd = [sumoBinary, "-c", "path/to/your/sumo/config/file.sumocfg"]
#     traci.start(sumoCmd)

#     # Example ground truth object
#     ground_truth_object = GroundTruthObject("veh1", "veh_passenger", (100, 100), (0, 0), [], 0, 1.8, 5.0)

#     # Example sensors and their extrinsics
#     sensors = [
#         sensor.Sensor(sensor_type=sensor_type.SensorType.OS1_128, detector_type=detector_type.DetectorType.POINT_PILLARS),
#         sensor.Sensor(sensor_type=sensor_type.SensorType.OS1_128, detector_type=detector_type.DetectorType.POINT_PILLARS)
#     ]
#     sensors_extrinsics = [
#         (1.0, 0.5, 0.1),  # (x, y, yaw) relative to the CAV
#         (-1.0, -0.5, -0.1)
#     ]

#     # Initialize the CAV
#     cav = CAV("veh1", ground_truth_object, sensors, sensors_extrinsics)

#     # Get the IDs of all vehicles in the simulation
#     vehicle_ids = traci.vehicle.getIDList()

#     # Create ground truth data for all vehicles
#     ground_truth_objects = [ground_truth.create_ground_truth_by_id(traci, veh_id) for veh_id in vehicle_ids]

#     # Track the IDs of all polygons
#     polygon_ids = []

#     # Simulation loop
#     step = 0
#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()

#         # Remove all polygons from the last frame
#         for polygon_id in polygon_ids:
#             traci.polygon.remove(polygon_id)
#         polygon_ids = []

#         # Update CAVs if step >= 1
#         if step >= 1:
#             cav.create_detection_sets(ground_truth_object, ground_truth_objects, step)
#             cav.draw_detected_objects(traci, polygon_ids)

#         step += 1

#     traci.close()