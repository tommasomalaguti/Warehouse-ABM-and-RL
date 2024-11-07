import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from test import ForkliftEnv  # Import your environment class

# Load the pre-trained PPO model
model = PPO.load("ppo_forklift.zip")

# Create the environment
env = ForkliftEnv()

# Initialize a list to store the results
results = []

# Run the simulation using the PPO model
for episode in range(999):
    obs = env.reset()
    done = False
    episode_results = []
    while not done:
        # Use the PPO model to predict the action
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Collect relevant information for each step
        step_data = {
            'episode': episode,
            'step': env.step_count,
            'task': env.current_task,
            'speed': info['speed'],
            'loading_time': info['loading_time'],
            'unloading_time': info['unloading_time'],
            'accident_prob': info['accident_prob'],
            'load_weight': env.load_weight,
            'distance_to_target': env.distance_to_target,
            'done': done,
            'accident_occurred': info['accident_occurred'],
            'reward': reward,
            'total_time': env.total_time,
            'energy_consumption': env.total_energy_consumption,
            'cumulative_energy_per_episode': env.cumulative_energy_per_episode,
            'global_cumulative_energy': env.global_energy_consumption
        }
        episode_results.append(step_data)

    # Append episode results to the main results list
    results.extend(episode_results)

# Convert the results list to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('forklift_ppo_sim.csv', index=False)

# Ensure the CSV file in the environment is closed properly
env.close()
