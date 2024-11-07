from stable_baselines3 import PPO
from model5_sensors_ml import WarehouseModel  # Import the WarehouseModel class
from gym_model5 import ForkliftEnv  # Import the ForkliftEnv class
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def main():
    # Define the dimensions of the warehouse
    width, height = 30, 30
    max_cycles = 100  # Set the desired number of unloading cycles

    # Initialize the WarehouseModel
    warehouse = WarehouseModel(width, height, max_cycles=max_cycles)

    # Wrap the WarehouseModel with the custom Gym environment
    env = ForkliftEnv(model=warehouse)
    
    # Initialize the PPO model with the environment
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000, log_interval=10)

    # Save the trained model to a file
    model.save("ppo_forklift")

    # Test the trained model in a continuous simulation
    obs = env.reset()
    completed_cycles = 0

    while completed_cycles < max_cycles:  # Run until the desired number of unloading cycles is reached
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        # Check if a crash occurred or if the task is done
        if dones:
            completed_cycles += 1  # Increment the unloading cycle counter
            print(f"Unloading cycle {completed_cycles} completed.")
            if completed_cycles < max_cycles:
                # Reset the environment and choose a new waitpoint
                warehouse.reset_loading_slot()  # Choose a new red waitpoint
                obs = env.reset()
        else:
            # Continue the simulation
            env.render()  # Add rendering to visualize the environment

    print(f"Simulation completed: {completed_cycles} unloading cycles performed.")

if __name__ == "__main__":
    main()
