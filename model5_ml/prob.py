import argparse
from matplotlib import pyplot as plt
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accident_probability(weight, speed, step=0, alpha=1, beta=1, gamma=3, step_multiplier=0.03):
    # Normalize weight and speed
    W_norm = (weight - 500) / (3000 - 500)
    S_norm = (speed - 1) / (3 - 1)
    
    # Linear combination of inputs with a multiplier for the step
    linear_combination = alpha * W_norm + beta * S_norm - gamma + step * step_multiplier
    
    # Calculate probability using the sigmoid function
    probability = sigmoid(linear_combination)
    
    return probability

def simulate_accident_probability(weight, speed, steps, alpha=1.5, beta=1.5, gamma=3, step_multiplier=0.005):
    for step in range(steps):
        probability = accident_probability(weight, speed, step, alpha, beta, gamma, step_multiplier)
        print(f"Step {step + 1}: The accident probability is: {probability:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate accident probability over a number of steps.')
    
    parser.add_argument('--weight', type=float, required=True, help='Weight of the vehicle/load (e.g., 1500)')
    parser.add_argument('--speed', type=float, required=True, help='Speed of the vehicle (e.g., 2)')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps to simulate (e.g., 20)')
    
    args = parser.parse_args()
    
    simulate_accident_probability(args.weight, args.speed, args.steps)