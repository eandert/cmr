import os
from simulation.simulations import simulate_without_ego, simulate_with_solution, simulate_with_planner

path = os.path.abspath("")
# specify required arguments
name_scenario = 'DEU_Frankfurt-34_11_I-1'
folder_scenarios = os.path.join(path, 'interactive_scenarios')
path_scenario = os.path.join(folder_scenarios, name_scenario)
path_video = os.path.join(path, '../outputs/videos')
# run simulation. alternatively, use functions simulate_with_solution() and simulate_with_planner()
scenario_without_ego, pps = simulate_without_ego(interactive_scenario_path=path_scenario,
			output_folder_path=path_video,
			create_video=True)