import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from test import ForkliftEnv  # Replace 'test' with the actual filename where ForkliftEnv is defined

# Create environment
env = ForkliftEnv()

# Define a learning rate schedule
def learning_rate_schedule(progress_remaining):
    return 9e-4 * progress_remaining

# Define a PPO model with a custom learning rate schedule
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=learning_rate_schedule,  # Use a learning rate schedule
    n_steps=2048,        # Number of steps to run for each environment per update
    gamma=0.99,          # Discount factor
    clip_range=0.2,      # Clipping for the PPO objective
    ent_coef=0.01,       # Coefficient for the entropy term, promoting exploration
)

# Lists to store episode rewards and energy consumption
episode_rewards = []
global_energy_consumption = []

# Define a custom callback to track rewards and energy consumption
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.global_energy = []

    def _on_step(self) -> bool:
        # Check if the episode is done
        done = self.locals['dones'][0]
        if done:
            env = self.locals['env'].envs[0]
            self.episode_rewards.append(np.sum(self.locals['rewards']))
            self.global_energy.append(env.global_energy_consumption)
        return True

    def _on_training_end(self) -> None:
        global episode_rewards, global_energy_consumption
        episode_rewards = self.episode_rewards  # Save the episode rewards to the global list for plotting
        global_energy_consumption = self.global_energy

# Train the model with the reward tracking callback
callback = RewardTrackingCallback()
model.learn(total_timesteps=1000000, callback=callback)

# Save the trained model
model.save("forklift_ppo_model")

# Test the trained model
obs = env.reset()
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
    # Print the current state, action, reward, and accident probability
    current_task, load_weight, distance_to_target, accident_prob = obs    
    if dones:
        obs = env.reset()

# Calculate the moving average (mean reward trend)
window_size = 50  # Adjust the window size as needed
mean_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')

# Plotting the episode rewards and the mean reward trend
plt.figure(figsize=(15, 10))
# Scatter plot of episode rewards
plt.scatter(range(len(episode_rewards)), episode_rewards, label='Episode Reward', alpha=0.5)
# Line plot of mean reward trend
plt.plot(range(window_size - 1, len(episode_rewards)), mean_rewards, label='Mean Reward Trend', color='orange', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode with Mean Reward Trend')
plt.legend()

# Save the figure before showing it
plt.savefig('training_progress.png')

# Show the plot
plt.show()

# Plotting the episode rewards in one window
plt.figure(figsize=(15, 10))
plt.scatter(range(len(episode_rewards)), episode_rewards, label='Episode Reward', alpha=0.5)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.legend()
# Save and show the first plot
plt.savefig('episode_rewards.png')
plt.show()

# Plotting the mean reward trend in another window
plt.figure(figsize=(15, 10))
plt.plot(range(window_size - 1, len(episode_rewards)), mean_rewards, label='Mean Reward Trend', color='orange', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward Trend per Episode')
plt.legend()
# Save and show the second plot
plt.savefig('mean_reward_trend.png')
plt.show()
