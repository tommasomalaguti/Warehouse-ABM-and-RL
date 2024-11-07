# Load the pre-trained PPO model
from stable_baselines3 import PPO
import csv
from test2 import ForkliftEnv


model = PPO.load("ppo_forklift.zip")

# Create the environment
env = ForkliftEnv()

# Run the simulation using the PPO model
for episode in range(999):
    obs = env.reset()
    done = False
    while not done:
        # Use the PPO model to predict the action
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

# Ensure the CSV file is closed properly
env.close()