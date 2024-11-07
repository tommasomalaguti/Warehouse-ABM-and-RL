import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
import plotly.graph_objs as go
import plotly.io as pio
from scipy.ndimage import uniform_filter1d

# Load the data from the CSV file
csv = "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_4_qalpha.csv"
data = pd.read_csv(csv)

# Extract the relevant columns based on the provided CSV headers
episode = data['Episode']
cumulative_energy = data['Cumulative Energy Per Episode (kWh)']
global_energy_consumption = data['Global Cumulative Energy (kWh)']
accident_probability = data['Accident Probability']
energy_consumption = data['Energy Consumption (kWh)']
random_value = data['Random Number']

# Calculate the maximum energy per episode
max_energy_per_episode = data.groupby('Episode')['Cumulative Energy Per Episode (kWh)'].max()

# Calculate the mean energy per episode
mean_energy_per_episode = data.groupby('Episode')['Global Cumulative Energy (kWh)'].mean()

data_sampled = data.sample(frac=0.1, random_state=42)  # Sample 10% of the data for performance

# Assuming there is a 'Reward' column in your CSV file
episode_rewards = data['Reward']  # Replace 'Reward' with the actual column name in your dataset

# Calculate the mean reward trend using a moving average or smoothing function
window_size = 50  # Define the window size for smoothing
mean_rewards = uniform_filter1d(episode_rewards, size=window_size)

# Adjust the x range to match the length of mean_rewards
x_values = range(len(mean_rewards))

# File paths for the CSV results
file_paths = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_4_qalpha.csv"
]

# Load all the CSV files
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Extract the 'Reward' column from each CSV
rewards_list = [df['Reward'].to_numpy() for df in dataframes]

# Ensure all reward arrays have the same length by trimming to the minimum length (in case of slight differences)
min_length = min(map(len, rewards_list))
rewards_list = [rewards[:min_length] for rewards in rewards_list]

# Convert the list of rewards into a numpy array (shape: num_runs x num_steps)
rewards_array = np.array(rewards_list)

# Calculate the mean and standard deviation across runs
mean_rewards = np.mean(rewards_array, axis=0)
std_rewards = np.std(rewards_array, axis=0)

# Increase smoothing window size for a cleaner line (e.g., 1000)
window_size = 1000

# Recalculate smoothed mean rewards
mean_rewards_smoothed = np.convolve(mean_rewards, np.ones(window_size) / window_size, mode='valid')

# Downsample the indices for standard deviation shading to reduce plotting load
# For instance, sample every 10th point to reduce the amount of drawing
downsample_factor = 10
downsampled_indices = np.arange(0, len(mean_rewards_smoothed), downsample_factor)

# Cumulative Reward Across Runs
cumulative_rewards_list = [df['Reward'].cumsum() for df in dataframes]
avg_cumulative_rewards = np.mean(cumulative_rewards_list, axis=0)

plt.figure(figsize=(12, 6))
for i, cum_reward in enumerate(cumulative_rewards_list):
    plt.plot(cum_reward, label=f'Run {i+1}', alpha=0.6)

plt.plot(avg_cumulative_rewards, label='Average Cumulative Reward', linewidth=2, color='black')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Across Different Runs when α=1.5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_reward_across_runs_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('cumulative_reward_across_runs_1.5q.png', format='png', dpi=600)
plt.close()

# Create the plot
plt.figure(figsize=(15, 8))

# Plot the smoothed mean rewards using a line plot
plt.plot(range(len(mean_rewards_smoothed)), mean_rewards_smoothed, label='Mean Reward (smoothed)', color='blue', linewidth=2)

# Scale the SD to a fraction (e.g., 0.5) to reduce the dominance of the shaded area
scale_factor = 0.2

# Adjust the opacity of the shaded standard deviation area using downsampled points
plt.fill_between(downsampled_indices,
                 mean_rewards_smoothed[downsampled_indices] - scale_factor * std_rewards[:len(mean_rewards_smoothed)][downsampled_indices],
                 mean_rewards_smoothed[downsampled_indices] + scale_factor * std_rewards[:len(mean_rewards_smoothed)][downsampled_indices],
                 color='blue', alpha=0.1)  # Reduced alpha for better visibility of the trend line

# Set y-limits to focus on a more relevant range
plt.ylim(-0.3, 0.1)

plt.xlabel('Episode', fontsize=14)  # Increase font size for the x-axis label
plt.ylabel('Reward', fontsize=14)  # Increase font size for the y-axis label
plt.title('Learning Curve Over 5 Runs when α=1.5', fontsize=16)  # Increase font size for the title
plt.legend(fontsize=12)  # Increase font size for the legend
plt.tick_params(axis='both', which='major', labelsize=12)  # Increase font size for tick labels
plt.grid(True)

# Save the plot
plt.savefig('mean_reward_trend_with_scaled_std_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('mean_reward_trend_with_scaled_std_1.5q.png', format='png', dpi=600)
plt.close()

# Plot Reward Distribution
plt.figure(figsize=(12, 6))
plt.hist(episode_rewards, bins=50, color='green', alpha=0.7)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Over All Episodes when α=1.5')
plt.tight_layout()
plt.savefig('reward_distribution_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('reward_distribution_1.5q.png', format='png', dpi=600)
plt.close()

# Extract the 'Energy Consumption' column from each CSV
energy_consumption_list = [df['Energy Consumption (kWh)'].to_numpy() for df in dataframes]

# Ensure all energy arrays have the same length by trimming to the minimum length (in case of slight differences)
min_length = min(map(len, energy_consumption_list))
energy_consumption_list = [energy[:min_length] for energy in energy_consumption_list]

# Convert the list of energy consumptions into a numpy array (shape: num_runs x num_steps)
energy_consumption_array = np.array(energy_consumption_list)

# Calculate the mean energy consumption across runs
mean_energy_consumption = np.mean(energy_consumption_array, axis=0)

# Apply smoothing to the mean energy consumption
window_size = 500  # Adjust window size for more or less smoothing
smoothed_mean_energy_consumption = uniform_filter1d(mean_energy_consumption, size=window_size)

# Plot smoothed mean energy consumption over episodes
plt.figure(figsize=(12, 6))
plt.plot(range(len(smoothed_mean_energy_consumption)), smoothed_mean_energy_consumption, label='Smoothed Mean Energy Consumption (kWh)', color='blue', linewidth=2)

# Add labels, title, and legend
plt.xlabel('Episode')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Smoothed Mean Energy Consumption Over Time (Averaged Across 5 Runs) when α=1.5')
plt.legend(fontsize=12)
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig('smoothed_mean_energy_consumption_trend_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('smoothed_mean_energy_consumption_trend_1.5q.png', format='png', dpi=600)
plt.close()

# Extract the 'Accident Probability' column from each CSV
accident_probability_list = [df['Accident Probability'].to_numpy() for df in dataframes]

# Ensure all accident probability arrays have the same length by trimming to the minimum length (in case of slight differences)
min_length = min(map(len, accident_probability_list))
accident_probability_list = [acc_prob[:min_length] for acc_prob in accident_probability_list]

# Convert the list of accident probabilities into a numpy array (shape: num_runs x num_steps)
accident_probability_array = np.array(accident_probability_list)

# Calculate the mean accident probability across runs
mean_accident_probability = np.mean(accident_probability_array, axis=0)

# Apply smoothing to the mean accident probability
window_size = 500  # Adjust window size for more or less smoothing
smoothed_mean_accident_probability = uniform_filter1d(mean_accident_probability, size=window_size)

# Plot smoothed mean accident probability over episodes
plt.figure(figsize=(12, 6))
plt.plot(range(len(smoothed_mean_accident_probability)), smoothed_mean_accident_probability, label='Smoothed Mean Accident Probability', color='orange', linewidth=2)

# Add labels, title, and legend
plt.xlabel('Episode')
plt.ylabel('Accident Probability')
plt.title('Smoothed Mean Accident Probability Over Time (Averaged Across 5 Runs) when α=1.5')
plt.legend(fontsize=12)
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig('smoothed_mean_accident_probability_trend_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('smoothed_mean_accident_probability_trend_1.5q.png', format='png', dpi=600)
plt.close()

# Plot 7: Comparison of Smoothed Accident Probability and Random Value
window_size_small = 10  # Original window size for less smoothing
window_size_large = 2000  # Larger window size for more smoothing

# Smoothing with a smaller window size
smoothed_accident_prob_small = uniform_filter1d(accident_probability, size=window_size_small)
smoothed_random_value_small = uniform_filter1d(random_value, size=window_size_small)

# Smoothing with a larger window size
smoothed_accident_prob_large = uniform_filter1d(accident_probability, size=window_size_large)
smoothed_random_value_large = uniform_filter1d(random_value, size=window_size_large)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_accident_prob_small, label='Smoothed Accident Probability (window=10)', color='blue', alpha=0.7)
plt.plot(episode, smoothed_random_value_small, label='Smoothed Random Value (window=10)', color='red', alpha=0.7)
plt.plot(episode, smoothed_accident_prob_large, label='Smoothed Accident Probability (window=2000)', color='blue', linestyle='--', alpha=0.9)
plt.plot(episode, smoothed_random_value_large, label='Smoothed Random Value (window=2000)', color='red', linestyle='--', alpha=0.9)
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Comparison of Smoothed Accident Probability and Random Value with Different Window Sizes when α=1.5')
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.savefig('smoothed_comparison_accident_probability_random_value_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('smoothed_comparison_accident_probability_random_value_1.5q.png', format='png', dpi=600)
plt.close()

# Calculate the mean energy consumption across runs
energy_consumption_list = [df['Energy Consumption (kWh)'].to_numpy() for df in dataframes]
min_length = min(map(len, energy_consumption_list))
energy_consumption_list = [energy[:min_length] for energy in energy_consumption_list]
energy_consumption_array = np.array(energy_consumption_list)
mean_energy_consumption = np.mean(energy_consumption_array, axis=0)

window_size = 500  # You can adjust this window size for more or less smoothing

# Apply smoothing to the mean energy consumption
smoothed_mean_energy_consumption = uniform_filter1d(mean_energy_consumption, size=window_size)

# Accident probability already calculated above
accident_probability_list = [df['Accident Probability'].to_numpy() for df in dataframes]
accident_probability_list = [acc_prob[:min_length] for acc_prob in accident_probability_list]
accident_probability_array = np.array(accident_probability_list)
mean_accident_probability = np.mean(accident_probability_array, axis=0)

# Apply smoothing to the mean accident probability
smoothed_mean_accident_probability = uniform_filter1d(mean_accident_probability, size=window_size)

# Plot comparison of smoothed mean accident probability and energy consumption
plt.figure(figsize=(12, 6))
plt.plot(range(len(smoothed_mean_accident_probability)), smoothed_mean_accident_probability, label='Smoothed Mean Accident Probability', color='blue', alpha=0.7)
plt.plot(range(len(smoothed_mean_energy_consumption)), smoothed_mean_energy_consumption, label='Smoothed Mean Energy Consumption', color='red', alpha=0.7)

# Add labels, title, and legend
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Comparison of Smoothed Mean Accident Probability and Energy Consumption Over Time when α=1.5')
plt.legend(fontsize=12)
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig('comparison_mean_accident_probability_energy_consumption_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('comparison_mean_accident_probability_energy_consumption_1.5q.png', format='png', dpi=600)
plt.close()

# Create a 3D scatter plot with Matplotlib
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data with swapped axes (accident_probability on z-axis and episode on y-axis)
scatter = ax.scatter(data_sampled['Energy Consumption (kWh)'], data_sampled['Episode'], data_sampled['Accident Probability'], 
                     c=data_sampled['Episode'], cmap='viridis', alpha=0.3)

# Labels and title
ax.set_xlabel('Energy Consumption (kWh)')
ax.set_ylabel('Episode')
ax.set_zlabel('Accident Probability')
ax.set_title('3D Scatter Plot of Energy Consumption vs. Episode vs. Accident Probability when α=1.5')

# Add a color bar to indicate the episode number
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Episode')

# Set the viewing angle
ax.view_init(elev=20, azim=45)

# Save the plot as a PDF file with 300 DPI
plt.savefig('3d_scatter_plot_1.5q.pdf', format='pdf', dpi=300)
plt.savefig('3d_scatter_plot_1.5q.png', format='png', dpi=600)
plt.close()

# Create a 3D scatter plot with Matplotlib
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
