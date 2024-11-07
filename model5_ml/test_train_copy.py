import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import csv

# Replace 'test' with the actual filename where ForkliftEnv is defined
from test import ForkliftEnv  

# Create environment function to ensure new environment for each run
def create_env():
    return ForkliftEnv()

# Callback to track rewards during training
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(np.sum(self.locals['rewards']))
        return True

    def _on_training_end(self) -> None:
        # Save the episode rewards within the callback instance
        self.locals['episode_rewards'] = self.episode_rewards

# Function to run the PPO model without any specific seed
def run_experiment_without_seed(total_timesteps=100000):
    env = create_env()
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        learning_rate=9e-4,  # Fixed learning rate for all runs
        n_steps=2048, 
        gamma=0.99, 
        clip_range=0.2, 
        ent_coef=0.01
    )
    callback = RewardTrackingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Return the rewards recorded by the callback
    return callback.episode_rewards

# Perform multiple runs (e.g., 5 runs) without setting a seed
num_runs = 5
all_rewards = []  # Store rewards for each run

for run in range(num_runs):
    print(f"Running experiment {run+1} without seed")
    rewards = run_experiment_without_seed()
    all_rewards.append(rewards)

# Convert the list of lists into a 2D numpy array (episodes x runs)
max_len = max([len(r) for r in all_rewards])  # Find the maximum episode length
all_rewards_padded = np.array([np.pad(r, (0, max_len - len(r)), 'constant', constant_values=np.nan) for r in all_rewards])

# Calculate the mean and standard deviation across runs
mean_rewards = np.nanmean(all_rewards_padded, axis=0)
std_rewards = np.nanstd(all_rewards_padded, axis=0)

# Plotting the mean reward trend across episodes with standard deviation
plt.figure(figsize=(12, 6))
plt.plot(mean_rewards, label='Mean Reward', color='blue')
plt.fill_between(range(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 color='blue', alpha=0.3, label='Standard Deviation')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Mean Rewards Across Runs (Without Seeds) with Standard Deviation')
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, you can save the results to a CSV file for future analysis
with open('episode_rewards_multiple_runs_no_seed.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode'] + [f'Run_{i+1}' for i in range(num_runs)])  # Header row
    for i in range(max_len):
        row = [i] + [all_rewards_padded[j, i] if not np.isnan(all_rewards_padded[j, i]) else '' for j in range(num_runs)]
        writer.writerow(row)
