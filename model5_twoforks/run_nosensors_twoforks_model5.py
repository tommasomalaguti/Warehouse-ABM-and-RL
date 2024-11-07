import argparse
import pandas as pd
from model5_nosensors_twoforks import WarehouseModel2

def main(total_time_hours=None, max_cycles=None):
    # Set the parameters for the simulation
    width = 30
    height = 30

    # Run the simulation
    model = WarehouseModel2(width, height, total_time_hours=total_time_hours, max_cycles=max_cycles)
    while model.running:
        model.step()

    # Collect the data
    data = model.datacollector.get_agent_vars_dataframe()

    # Save the overall data to a CSV file
    data.to_csv("simulation_results_nosensors_montecarlos_model5_twoforks.csv")

    # Iterate over each forklift to save their respective speeds and other data
    for i, forklift_agent in enumerate(model.forklifts, start=1):
        # Filter the data for the specific forklift agent
        forklift_data = data.xs(forklift_agent.unique_id, level="AgentID")

        print(f"Summary statistics for Forklift {i}:")
        print(forklift_data.describe())

        print(f"Last collected data points for Forklift {i}:")
        print(forklift_data.tail())

        # Save the forklift-specific data to a CSV file
        forklift_data.to_csv(f"simulation_results_forklift_{i}_model5_twoforks.csv")

        # Check lengths of each list
        empty_speeds_len = len(forklift_agent.empty_speeds_used)
        loaded_speeds_len = len(forklift_agent.loaded_speeds_used)
        searching_speeds_len = len(forklift_agent.searching_speeds_used)

        print(f"Lengths for Forklift {i} - Empty: {empty_speeds_len}, Loaded: {loaded_speeds_len}, Searching: {searching_speeds_len}")

        # Ensure all lists are of the same length by trimming to the shortest length
        min_length = min(empty_speeds_len, loaded_speeds_len, searching_speeds_len)
        empty_speeds_used = forklift_agent.empty_speeds_used[:min_length]
        loaded_speeds_used = forklift_agent.loaded_speeds_used[:min_length]
        searching_speeds_used = forklift_agent.searching_speeds_used[:min_length]

        speeds_data = pd.DataFrame({
            'Empty Speeds Used': empty_speeds_used,
            'Loaded Speeds Used': loaded_speeds_used,
            'Searching Speeds Used': searching_speeds_used
        })

        # Save the speeds data to a CSV file
        speeds_data.to_csv(f"speeds_used_forklift_{i}_model5_twoforks.csv", index=False)

        print(f"Speeds used after every unloading for Forklift {i}:")
        print(speeds_data)

        # Save the searching speeds data to a separate CSV file
        searching_speeds_data = pd.DataFrame({
            'Searching Speeds Used': forklift_agent.searching_speeds_used
        })
        
        searching_speeds_data.to_csv(f"searching_speeds_used_forklift_{i}_model5_twoforks.csv", index=False)

        print(f"Searching speeds used for Forklift {i}:")
        print(searching_speeds_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WarehouseModel2 simulation.")
    parser.add_argument("--total_time_hours", type=float, help="Total time to run the simulation in hours.")
    parser.add_argument("--max_cycles", type=int, help="Maximum number of unloading cycles to run the simulation.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug information during simulation.")

    args = parser.parse_args()

    if args.total_time_hours is None and args.max_cycles is None:
        parser.error("At least one of --total_time_hours or --max_cycles must be specified.")

    main(total_time_hours=args.total_time_hours, max_cycles=args.max_cycles)
