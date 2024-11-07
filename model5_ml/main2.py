import argparse
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from model5_sensors_ml import WarehouseModel
from gym_model5 import ForkliftEnv

def main(total_time_hours=None, max_cycles=None, total_timesteps=100000, load_path=None):
    # Set the parameters for the simulation
    width = 30
    height = 30

    # Initialize the WarehouseModel
    model = WarehouseModel(width, height, total_time_hours=total_time_hours, max_cycles=max_cycles)

    # Wrap the WarehouseModel with the custom Gym environment
    env = ForkliftEnv(model=model)
    
    # Check if there is a pre-trained model to load
    if load_path and os.path.exists(load_path):
        print(f"Loading pre-trained model from {load_path}")
        ppo_model = PPO.load(load_path, env=env)
    else:
        print("No pre-trained model found, starting training from scratch")
        ppo_model = PPO("MlpPolicy", env, verbose=1)

    # Continue training the model
    ppo_model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    ppo_model.save("ppo_forklift_continuous")

    # Initialize reward structure list
    reward_structure = []

    # After training, run the simulation without resetting
    while model.running:
        model.step()
        reward_structure.append(model.forklift.current_reward)  # Replace with appropriate reward logic

    # Collect the data
    data = model.datacollector.get_agent_vars_dataframe()

    # Filter the data for the forklift agent only
    forklift_data = data.xs(1, level="AgentID")

    # Save the data to a CSV file
    forklift_data.to_csv("simulation_results_continuous_training.csv")

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
    speeds_data.to_csv("speeds_used_continuous_training.csv", index=False)

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

    # Saving the reward structure to a CSV file
    reward_df = pd.DataFrame({"Cycle": range(1, len(reward_structure) + 1), "Reward": reward_structure})
    reward_df.to_csv("reward_structure.csv", index=False)
    print("Reward structure saved to reward_structure.csv")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WarehouseModel simulation with continuous training.")
    parser.add_argument("--total_time_hours", type=float, help="Total time to run the simulation in hours.")
    parser.add_argument("--max_cycles", type=int, help="Maximum number of unloading cycles to run the simulation.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps for PPO training.")
    parser.add_argument("--load_path", type=str, default=None, help="Path to the pre-trained model to load.")

    args = parser.parse_args()

    if args.total_time_hours is None and args.max_cycles is None:
        parser.error("At least one of --total_time_hours or --max_cycles must be specified.")

    main(total_time_hours=args.total_time_hours, max_cycles=args.max_cycles, total_timesteps=args.total_timesteps, load_path=args.load_path)
