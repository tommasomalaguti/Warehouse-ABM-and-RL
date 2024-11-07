import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class ForkliftEnv(gym.Env):
    def __init__(self, model):
        super(ForkliftEnv, self).__init__()
        self.model = model
        self.action_space = spaces.Box(low=np.array([0.1, 1, 1]), high=np.array([6.0, 300, 300]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
    
    def reset(self):
        self.model.reset()  
        state = self._get_state()
        print("Environment reset. Initial state:", state)
        return state

    def step(self, action):
        speed, load_time, unload_time = action
        self.model.forklift.speed_empty = speed
        self.model.forklift.speed_loaded = speed 
        self.model.forklift.loading_time = load_time
        self.model.forklift.unloading_time = unload_time
        
        self.model.step()

        # More detailed debugging output
        print(f"\n--- Step Debugging Info ---")
        print(f"Action applied - Speed: {speed}, Load Time: {load_time}, Unload Time: {unload_time}")
        print(f"Forklift parameters after applying action - Speed Loaded: {self.model.forklift.speed_loaded}, "
              f"Loading Time: {self.model.forklift.loading_time}, "
              f"Unloading Time: {self.model.forklift.unloading_time}")
        
        state = self._get_state()
        reward, done = self._calculate_reward(state)
        self.model.forklift.current_reward = reward  # Store the current reward for tracking
        
        print(f"New State: {state}")
        print(f"Reward calculated: {reward}, Episode done: {done}")
        print(f"--- End of Step Debugging Info ---\n")
        
        return state, reward, done, {}
    
    def _get_state(self):
        forklift = self.model.forklift
        speed = forklift.speed_loaded if forklift.currently_loaded else forklift.speed_empty
        return np.array([
            speed,  # Use loaded speed if loaded, else use empty speed
            forklift.load_weight,
            forklift.distance_since_loading,
            forklift.loading_time,
            forklift.unloading_time,
            forklift.current_accident_probability,
            forklift.energy_consumption
        ])

    
    def _calculate_reward(self, state):
        accident_probability = state[5]
        energy_consumption = state[6]
        speed_empty = state[0]
        currently_loaded = self.model.forklift.currently_loaded

        print(f"Calculating reward based on state - Accident Probability: {accident_probability}, "
              f"Energy Consumption: {energy_consumption}, Speed Empty: {speed_empty}, Currently Loaded: {currently_loaded}")

        reward = 0
        done = False

        # Strong, non-linear penalty for high accident probability
        mean_accident_probability = 0.3
        if accident_probability > mean_accident_probability:
            penalty = 500 * (accident_probability - mean_accident_probability) ** 2
            reward -= penalty  # Quadratic penalty
            done = accident_probability > 0.8  # End episode on very high risk
            print(f"High accident probability penalty applied: {penalty}, New Reward: {reward}, Done: {done}")

        # Reward for safe operation with a decent speed
        if currently_loaded and accident_probability < mean_accident_probability:
            reward += 50 * speed_empty  # Reward for higher speed when safely loaded
            print(f"Safe operation reward applied: {50 * speed_empty}, New Reward: {reward}")

        # Penalize low speed when empty
        if not currently_loaded and speed_empty < 3.0:
            penalty = 20 * (3.0 - speed_empty)  # Encourage speed to be at least 3.0 m/s
            reward -= penalty
            print(f"Low speed penalty applied: {penalty}, New Reward: {reward}")

        # Penalty for energy consumption
        energy_penalty = 0.5 * energy_consumption
        reward -= energy_penalty  # Keep this penalty as is
        print(f"Energy consumption penalty applied: {energy_penalty}, New Reward: {reward}")

        # Penalty for long loading and unloading times, but not too strong
        load_time = state[3]
        unload_time = state[4]
        time_penalty = 0.1 * (load_time + unload_time)  # Mild penalty for taking too long
        reward -= time_penalty
        print(f"Time penalty applied: {time_penalty}, New Reward: {reward}")

        # Optional: Small reward for maintaining safe operation
        if accident_probability < mean_accident_probability:
            reward += 2  # Reward for being in a safe state
            print(f"Safe state reward applied: 2, New Reward: {reward}")

        return reward, done

    def render(self, mode='human'):
        pass  # Visualization can be added here if needed

    def close(self):
        pass
