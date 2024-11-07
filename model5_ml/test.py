import csv
import gym
import numpy as np
import math
from stable_baselines3 import PPO

class ForkliftEnv(gym.Env):
    def __init__(self, alpha=3):
        super(ForkliftEnv, self).__init__()

        self.speed_values = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
        self.loading_time_values = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250, 300]
        self.unloading_time_values = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250, 300]
        self.energy_rate = 4.5  # Energy consumption rate in kWh/h

        self.n_actions = len(self.speed_values) * len(self.loading_time_values) * len(self.unloading_time_values)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(4,), dtype=np.float32)

        self.alpha = alpha  # Add alpha here to balance reward

        import os

        # Specify the folder where you want to save the CSV file
        folder_path = 'results_data_3'

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Initialize the CSV file with the full path
        self.csv_file = open(os.path.join(folder_path, 'forklift_training_sim_results_3_4_qalpha.csv'), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'Episode', 'Step', 'Task', 'Speed', 'Loading Time', 'Unloading Time', 
            'Accident Probability', 'Load Weight', 'Distance to Target', 'Done', 
            'Random Number', 'Accident Occurred', 'Total Time', 'Energy Consumption (kWh)',
            'Cumulative Energy Per Episode (kWh)', 'Global Cumulative Energy (kWh)', 'Reward'
        ])

        self.episode = 0
        self.global_energy_consumption = 0  # Initialize global cumulative energy
        self.cumulative_energy_per_episode = 0  # Initialize cumulative energy per episode
        self.reset()

    def reset(self):
        self.current_task = 0  # Start with loading
        self.load_weight = np.random.uniform(500, 3000)
        self.distance_to_target = np.random.uniform(10, 100)  # Adjusted distance for quicker transitions
        self.accident_prob = 0.0
        self.prev_accident_prob = self.accident_prob
        self.travelling_loaded_steps = 0  # Counter for steps while traveling loaded
        self.step_count = 0
        self.total_time = 0  # Initialize total time for the episode
        self.total_energy_consumption = 0  # Initialize total energy consumption for the episode

        # Reset cumulative energy per episode at the start of a new episode
        self.cumulative_energy_per_episode = 0

        # Increment episode counter and return the initial state
        self.episode += 1
        return np.array([self.current_task, self.load_weight, self.distance_to_target, self.accident_prob])

    def step(self, action):
        loaded_speed, loading_time, unloading_time = self.decode_action(action)
        done = False
        accident_occurred = "No"  # Default to "No"

        # Parameters for the normal distribution
        mean = 0.5
        std_dev = 0.1  # You can adjust the standard deviation to control the spread
        random_value = np.random.normal(mean, std_dev)

        if self.current_task == 0:  # Loading
            accident_prob = self.accident_probability_loading(self.load_weight, loading_time)
            if random_value < accident_prob:
                accident_occurred = "Yes"
            self.total_time += loading_time
            energy_consumption = (loading_time / 3600) * self.energy_rate
            self.total_energy_consumption += energy_consumption
            self.cumulative_energy_per_episode += energy_consumption
            self.global_energy_consumption += energy_consumption
            reward = -accident_prob**2 - self.alpha * energy_consumption

            self.log_state("Loading", action, accident_prob, self.load_weight, done, random_value, accident_occurred, self.total_time, self.total_energy_consumption, self.cumulative_energy_per_episode, self.global_energy_consumption, reward)
            
            if not done:
                self.current_task = 1  # Move to traveling loaded
        
        elif self.current_task == 1:  # Traveling loaded
            accident_prob = self.accident_probability_traveling_loaded(self.load_weight, loaded_speed)
            if random_value < accident_prob:
                accident_occurred = "Yes"
            self.travelling_loaded_steps += 1
            self.distance_to_target -= loaded_speed
            time_increment = 1 / loaded_speed
            self.total_time += time_increment
            energy_consumption = (time_increment / 3600) * self.energy_rate
            self.total_energy_consumption += energy_consumption
            self.cumulative_energy_per_episode += energy_consumption
            self.global_energy_consumption += energy_consumption
            reward = -accident_prob**2 - self.alpha * energy_consumption  # New reward function

            self.log_state("Traveling Loaded", action, accident_prob, self.load_weight, done, random_value, accident_occurred, self.total_time, self.total_energy_consumption, self.cumulative_energy_per_episode, self.global_energy_consumption, reward)

            if not done and self.distance_to_target <= 0:
                self.distance_to_target = 0
                self.current_task = 2  # Move to unloading

        elif self.current_task == 2:  # Unloading
            accident_prob = self.accident_probability_unloading(self.load_weight, unloading_time)
            if random_value < accident_prob:
                accident_occurred = "Yes"
            self.total_time += unloading_time
            energy_consumption = (unloading_time / 3600) * self.energy_rate
            self.total_energy_consumption += energy_consumption
            self.cumulative_energy_per_episode += energy_consumption
            self.global_energy_consumption += energy_consumption
            reward = -accident_prob**2 - self.alpha * energy_consumption # New reward function

            done = True  # End the episode after unloading
            self.log_state("Unloading", action, accident_prob, self.load_weight, done, random_value, accident_occurred, self.total_time, self.total_energy_consumption, self.cumulative_energy_per_episode, self.global_energy_consumption, reward)

        state = np.array([self.current_task, self.load_weight, self.distance_to_target, accident_prob])
        info = {
            'speed': loaded_speed,
            'loading_time': loading_time,
            'unloading_time': unloading_time,
            'accident_prob': accident_prob,
            'current_task': self.current_task,
            'accident_occurred': accident_occurred  # Include accident occurrence in info
        }

        self.step_count += 1
        return state, reward, done, info

    def decode_action(self, action):
        # If action is an array, extract the first element
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]  # Flatten to 1D and take the first element

        action = int(action)  # Ensure the action is an integer

        speed_index = action % len(self.speed_values)
        action //= len(self.speed_values)
        loading_time_index = action % len(self.loading_time_values)
        action //= len(self.loading_time_values)
        unloading_time_index = action % len(self.unloading_time_values)

        loaded_speed = self.speed_values[speed_index]
        loading_time = self.loading_time_values[loading_time_index]
        unloading_time = self.unloading_time_values[unloading_time_index]

        return loaded_speed, loading_time, unloading_time

    def log_state(self, task, action, accident_prob, load_weight, done, random_value, accident_occurred, total_time, total_energy_consumption, cumulative_energy_per_episode, global_energy_consumption, reward):
        speed, load_time, unload_time = self.decode_action(action)
        self.csv_writer.writerow([
            self.episode, self.step_count, task, speed, load_time, unload_time, 
            accident_prob, load_weight, self.distance_to_target, done, 
            random_value, accident_occurred, total_time, total_energy_consumption, cumulative_energy_per_episode, global_energy_consumption, reward
        ])

    def accident_probability_loading(self, weight, load_time):
        normalized_weight = (weight - 500) / 2500
        normalized_loading_time = (load_time - 20) / (300 - 1)
        beta_weight = 3
        beta_loading_time = -5
        gamma = 0.007
        linear_combination = beta_weight * normalized_weight + beta_loading_time * normalized_loading_time - gamma
        accident_probability = 1 / (1 + math.exp(-linear_combination))
        return accident_probability

    def accident_probability_traveling_loaded(self, weight, speed):
        normalized_weight = (weight - 500) / 2500
        normalized_speed = (speed - 1) / (3 - 1)
        beta_weight = 1.5
        beta_speed = 2
        gamma = 3
        step_multiplier = 0.005
        linear_combination = beta_weight * normalized_weight + beta_speed * normalized_speed - gamma + self.travelling_loaded_steps * step_multiplier
        accident_probability = 1 / (1 + math.exp(-linear_combination))
        return accident_probability

    def accident_probability_unloading(self, weight, unload_time):
        normalized_weight = (weight - 500) / 2500
        normalized_unloading_time = (unload_time - 20) / (300 - 1)
        beta_weight = 3
        beta_unloading_time = -5
        gamma = 0.007
        linear_combination = beta_weight * normalized_weight + beta_unloading_time * normalized_unloading_time - gamma
        accident_probability = 1 / (1 + math.exp(-linear_combination))
        return accident_probability

    def close(self):
        print("Closing the CSV file...")
        self.csv_file.close()
