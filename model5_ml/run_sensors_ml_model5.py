import argparse
import pandas as pd
from model5_sensors_ml import WarehouseModel

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
    forklift_data.to_csv("simulation_results_sensors_model5.csv")

    # Print the summary statistics
    print(forklift_data.describe())

    # Print the last collected data point to ensure accuracy
    print(forklift_data.tail())

    # Print the speeds used after every unloading along with loading and unloading times
    forklift_agent = model.forklift
    speeds_data = pd.DataFrame({
        'Empty Speeds Used': forklift_agent.empty_speeds_used,
        'Loaded Speeds Used': forklift_agent.loaded_speeds_used,
        'Loading Time (s)': forklift_agent.loading_times_used,
        'Unloading Time (s)': forklift_agent.unloading_times_used,
        'Total Cycle Time (s)': forklift_agent.cycle_times_used
    })
    print("Speeds and Times used after every unloading:")
    print(speeds_data)

    # Save the speeds and times data to a CSV file
    speeds_data.to_csv("speeds_used_sensors_model5.csv", index=False)

    # Print loading and unloading times
    print("Loading and Unloading Times:")
    print(f"Loading Time: {forklift_agent.loading_time} seconds")
    print(f"Unloading Time: {forklift_agent.unloading_time} seconds")

    # Print the number of accidents and the details for each one
    print(f"Number of Accidents: {forklift_agent.accidents_count}")
    print("Accident Details:")
    for i, (prob, weight, distance, speeds, times, rand_num) in enumerate(zip(
            forklift_agent.accident_probabilities,
            forklift_agent.weights_lost,
            forklift_agent.distances_at_accidents,
            forklift_agent.speeds_at_accidents,
            forklift_agent.times_at_accidents,
            forklift_agent.random_numbers_at_accidents)):
        print(f"Accident {i + 1}:")
        print(f"  Probability = {prob:.4f}")
        print(f"  Random Number = {rand_num:.4f}")
        print(f"  Weight Lost = {weight:.2f} kg")
        print(f"  Distance Since Loading = {distance} steps")
        print(f"  Empty Speed = {speeds[0]:.2f} m/s")
        print(f"  Loaded Speed = {speeds[1]:.2f} m/s")
        print(f"  Loading Time = {times[0]:.2f} seconds")
        print(f"  Unloading Time = {times[1]:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WarehouseModel simulation.")
    parser.add_argument("--total_time_hours", type=float, help="Total time to run the simulation in hours.")
    parser.add_argument("--max_cycles", type=int, help="Maximum number of unloading cycles to run the simulation.")

    args = parser.parse_args()

    if args.total_time_hours is None and args.max_cycles is None:
        parser.error("At least one of --total_time_hours or --max_cycles must be specified.")

    main(total_time_hours=args.total_time_hours, max_cycles=args.max_cycles)
