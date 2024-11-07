import argparse
import pandas as pd
from model5_sensors_saturation import WarehouseModel

def main(total_time_hours=None, max_cycles=None):
    # Set the parameters for the simulation
    width = 30
    height = 30

    # Run the simulation
    model = WarehouseModel(width, height, total_time_hours=total_time_hours, max_cycles=max_cycles)
    while model.running:
        model.step()

    # Collect the data
    data = model.datacollector.get_agent_vars_dataframe()

    # Filter the data for the forklift agent only
    forklift_data = data.xs(1, level="AgentID")

    # Save the data to a CSV file
    forklift_data.to_csv("simulation_results_sensors_saturation_10steps_model5.csv")

    # Print the summary statistics
    print(forklift_data.describe())

    # Print the last collected data point to ensure accuracy
    print(forklift_data.tail())

    # Print the speeds used after every unloading
    forklift_agent = model.forklift
    speeds_data = pd.DataFrame({
        'Empty Speeds Used': forklift_agent.empty_speeds_used,
        'Loaded Speeds Used': forklift_agent.loaded_speeds_used
    })
    print("Speeds used after every unloading:")
    print(speeds_data)

    # Save the speeds data to a CSV file
    speeds_data.to_csv("speeds_used_sensors_saturation_model5.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WarehouseModel simulation.")
    parser.add_argument("--total_time_hours", type=float, help="Total time to run the simulation in hours.")
    parser.add_argument("--max_cycles", type=int, help="Maximum number of unloading cycles to run the simulation.")

    args = parser.parse_args()

    if args.total_time_hours is None and args.max_cycles is None:
        parser.error("At least one of --total_time_hours or --max_cycles must be specified.")

    main(total_time_hours=args.total_time_hours, max_cycles=args.max_cycles)
