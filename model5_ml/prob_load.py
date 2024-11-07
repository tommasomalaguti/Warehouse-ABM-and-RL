import argparse
import math

from matplotlib import pyplot as plt
import numpy as np

def accident_probability(weight, unload_time):
    # Normalize the weight and loading_time
    normalized_weight = (weight - 500) / 2500  # Scales to range [0, 1]
    
    # Normalize loading_time, assuming it ranges from 1 to 300 seconds
    normalized_unloading_time = (unload_time - 20) / (300 - 1)  # Scales to range [0, 1]
    
    # Adjust the betas
    beta_weight = 3
    beta_unloading_time = -4  # Negative to decrease probability with more time
    gamma = 0.007

    # Linear combination of inputs
    linear_combination = beta_weight * normalized_weight + beta_unloading_time * normalized_unloading_time - gamma

    # Calculate probability using the sigmoid function
    accident_probability = 1 / (1 + math.exp(-linear_combination))
    
    return accident_probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accident probability based on weight and loading time.')
    
    parser.add_argument('--weight', type=float, required=True, help='Weight of the vehicle/load (e.g., 1500)')
    parser.add_argument('--time', type=float, required=True, help='Time to load (e.g., 150)')
    
    args = parser.parse_args()
    
    # Access the time argument correctly
    probability = accident_probability(args.weight, args.time)
    
    print(f"The accident probability is: {probability:.4f}")
