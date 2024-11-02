import traci
import ground_truth as gt
import utils as ut

# Start SUMO in server mode
sumoBinary = "sumo-gui"  # or "sumo" if you don't need the GUI
sumoCmd = [sumoBinary, "-c", "C:/Users/edwar/Documents/GitHub/cmr/maps/tempe_2x3/osm.sumocfg"]
traci.start(sumoCmd)

# Initialize VehicleProbabilityManager for CAVs
cav_manager = ut.VehicleProbabilityManager(probability=0.1, type="CAV")

# Simulation loop
step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    # Get the IDs of all vehicles in the simulation
    vehicle_ids = traci.vehicle.getIDList()

    # Update CAVs if step >= 1
    if step >= 1:
        cav_manager.update_vehicles(vehicle_ids)

    # Get active CAVs
    cav_id_list = cav_manager.get_active_vehicles(vehicle_ids)

    # Print the vehicle IDs
    print(cav_id_list)

    # Create ground truth data for all vehicles
    ground_truth = gt.create_ground_truth(traci, vehicle_ids)

    # # Print the ground truth data
    # for data in ground_truth:
    #     print(data)
    
    step += 1

traci.close()