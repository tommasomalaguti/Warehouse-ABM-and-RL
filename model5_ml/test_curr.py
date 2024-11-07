import csv
import gym
import numpy as np
import math
from stable_baselines3 import PPO

class ForkliftEnv(gym.Env):
    def __init__(self, alpha=0.1, curriculum_phase=1):
        super(ForkliftEnv, self).__init__()

        # Initialize variables
        self.alpha = alpha
        self.curriculum_phase = curriculum_phase  # Start with phase 1
        self.total_timesteps = 0  # Track total timesteps

        # Curriculum parameters for different phases
        self.phases = {
            1: {'weight_range': (500, 1500), 'speed_range': (2, 3), 'time_range': (20, 60)},  # Easy
            2: {'weight_range': (1000, 2500), 'speed_range': (1.5, 2.5), 'time_range': (40, 120)},  # Intermediate
            3: {'weight_range': (2000, 3000), 'speed_range': (1, 2), 'time_range': (60, 180)}  # Difficult
        }

        # Select parameters based on the current phase
        self.speed_values = np.linspace(*self.phases[self.curriculum_phase]['speed_range'], num=11)
        self.loading_time_values = np.linspace(*self.phases[self.curriculum_phase]['time_range'], num=18).astype(int)
        self.unloading_time_values = np.linspace(*self.phases[self.curriculum_phase]['time_range'], num=18).astype(int)
        self.energy_rate = 4.5  # Energy consumption rate in kWh/h

        self.n_actions = len(self.speed_values) * len(self.loading_time_values) * len(self.unloading_time_values)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(4,), dtype=np.float32)

        # Initialize CSV and other variables
        self.csv_file = open('curr_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Episode', 'Step', 'Task', 'Speed', 'Loading Time', 'Unloading Time', 
            'Accident Probability', 'Load Weight', 'Distance to Target', 'Done', 
            'Random Number', 'Accident Occurred', 'Total Time', 'Energy Consumption (kWh)',
            'Cumulative Energy Per Episode (kWh)', 'Global Cumulative Energy (kWh)', 'Reward'
        ])
        self.episode = 0
        self.global_energy_consumption = 0
        self.cumulative_energy_per_episode = 0
        self.reset()

    def reset(self):
        self.current_task = 0
        self.load_weight = np.random.uniform(*self.phases[self.curriculum_phase]['weight_range'])
        self.distance_to_target = np.random.uniform(10, 50)
        self.accident_prob = 0.0
        self.prev_accident_prob = self.accident_prob
        self.travelling_loaded_steps = 0
        self.step_count = 0
        self.total_time = 0
        self.total_energy_consumption = 0
        self.cumulative_energy_per_episode = 0
        self.episode += 1

        # Check if it's time to move to the next phase based on timesteps
        self.update_curriculum_phase()

        return np.array([self.current_task, self.load_weight, self.distance_to_target, self.accident_prob])

    def update_curriculum_phase(self):
        # Example: move to next phase every 100,000 timesteps
        timestep_thresholds = [100000, 200000]  # Transition thresholds
        if self.total_timesteps >= timestep_thresholds[0] and self.curriculum_phase == 1:
            self.curriculum_phase = 2
        elif self.total_timesteps >= timestep_thresholds[1] and self.curriculum_phase == 2:
            self.curriculum_phase = 3

        # Update parameters for the new phase
        self.speed_values = np.linspace(*self.phases[self.curriculum_phase]['speed_range'], num=11)
        self.loading_time_values = np.linspace(*self.phases[self.curriculum_phase]['time_range'], num=18).astype(int)
        self.unloading_time_values = np.linspace(*self.phases[self.curriculum_phase]['time_range'], num=18).astype(int)

    def step(self, action):
        loaded_speed, loading_time, unloading_time = self.decode_action(action)
        done = False
        accident_occurred = "No"
        mean = 0.7
        std_dev = 0.1
        random_value = np.random.normal(mean, std_dev)

        if self.current_task == 0:
            accident_prob = self.accident_probability_loading(self.load_weight, loading_time)
            self.total_time += loading_time
            energy_consumption = (loading_time / 3600) * self.energy_rate
            self.total_energy_consumption += energy_consumption
            self.cumulative_energy_per_episode += energy_consumption
            self.global_energy_consumption += energy_consumption
            reward = -accident_prob - self.alpha * energy_consumption
            
            if random_value < accident_prob:
                done = True
                accident_occurred = "Yes"
            
            self.log_state("Loading", action, accident_prob, self.load_weight, done, random_value, accident_occurred, self.total_time, self.total_energy_consumption, self.cumulative_energy_per_episode, self.global_energy_consumption, reward)
            
            if not done:
                self.current_task = 1

        elif self.current_task == 1:
            accident_prob = self.accident_probability_traveling_loaded(self.load_weight, loaded_speed)
            self.travelling_loaded_steps += 1
            self.distance_to_target -= loaded_speed
            time_increment = 1 / loaded_speed
            self.total_time += time_increment
            energy_consumption = (time_increment / 3600) * self.energy_rate
            self.total_energy_consumption += energy_consumption
            self.cumulative_energy_per_episode += energy_consumption
            self.global_energy_consumption += energy_consumption
            reward = -accident_prob - self.alpha * energy_consumption

            if random_value < accident_prob:
                done = True
                accident_occurred = "Yes"
            self.log_state("Traveling Loaded", action, accident_prob, self.load_weight, done, random_value, accident_occurred, self.total_time, self.total_energy_consumption, self.cumulative_energy_per_episode, self.global_energy_consumption, reward)

            if not done and self.distance_to_target <= 0:
                self.distance_to_target = 0
                self.current_task = 2

        elif self.current_task == 2:
            accident_prob = self.accident_probability_unloading(self.load_weight, unloading_time)
            self.total_time += unloading_time
            energy_consumption = (unloading_time / 3600) * self.energy_rate
            self.total_energy_consumption += energy_consumption
            self.cumulative_energy_per_episode += energy_consumption
            self.global_energy_consumption += energy_consumption
            reward = -accident_prob - self.alpha * energy_consumption

            if random_value < accident_prob:
                done = True
                accident_occurred = "Yes"
            else:
                done = True
            self.log_state("Unloading", action, accident_prob, self.load_weight, done, random_value, accident_occurred, self.total_time, self.total_energy_consumption, self.cumulative_energy_per_episode, self.global_energy_consumption, reward)

        self.total_timesteps += 1  # Increment total timesteps

        state = np.array([self.current_task, self.load_weight, self.distance_to_target, accident_prob])
        info = {
            'speed': loaded_speed,
            'loading_time': loading_time,
            'unloading_time': unloading_time,
            'accident_prob': accident_prob,
            'current_task': self.current_task,
            'accident_occurred': accident_occurred
        }

        self.step_count += 1
        return state, reward, done, info

    def decode_action(self, action):
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]

        action = int(action)

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
