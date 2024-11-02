import random

class VehicleProbabilityManager:
    def __init__(self, probability, type):
        self.probability = probability
        self.vehicle_list = []
        self.total_vehicles = 0
        self.checked_vehicle_id_list = []
        self.type = type

    def update_vehicles(self, vehicle_id_list):
        # Get the IDs of all vehicles that have appeared in the simulation since the last step
        new_vehicle_id_list = [x for x in vehicle_id_list if x not in self.checked_vehicle_id_list]

        for vehicle_id in new_vehicle_id_list:
            randomnum = random.random()
            if self.probability != 0 and randomnum <= self.probability:
                try:
                    print(f"{self.type} added: ", vehicle_id)
                    self.total_vehicles += 1
                    self.vehicle_list.append(vehicle_id)
                except Exception as e:
                    print(f"ERROR: Couldn't add {self.type}: ", e)

        self.checked_vehicle_id_list += new_vehicle_id_list

    def get_active_vehicles(self, vehicle_ids):
        return list(set(self.vehicle_list) & set(vehicle_ids))