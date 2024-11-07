import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
import plotly.graph_objs as go
import plotly.io as pio
from scipy.ndimage import uniform_filter1d

# Load the data from the CSV file
csv = "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/forklift_training_sim_results_1.5_qalpha_maxed.csv"
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

# Plot 1: Combined plot of episode rewards and mean reward trend
plt.figure(figsize=(15, 10))
plt.scatter(x_values, episode_rewards[:len(mean_rewards)], label='Episode Reward', alpha=0.5)
plt.plot(x_values, mean_rewards, label='Mean Reward Trend', color='orange', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode with Mean Reward Trend')
plt.legend(loc='upper left')  # Set a specific location for the legend
plt.savefig('combined_rewards_mean_trend_1.5f.png')  # Save plot as a file
plt.show()

# Plot 2: Episode rewards
plt.figure(figsize=(15, 10))
plt.scatter(range(len(episode_rewards)), episode_rewards, label='Episode Reward', alpha=0.5)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.legend(loc='upper left')  # Set a specific location for the legend
plt.savefig('episode_rewards_1.5f.png')  # Save plot as a file
plt.show()

# Plot 3: Mean reward trend
plt.figure(figsize=(15, 10))
plt.plot(range(len(mean_rewards)), mean_rewards, label='Mean Reward Trend', color='orange', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward Trend per Episode')
plt.legend(loc='upper left')  # Set a specific location for the legend
plt.savefig('mean_reward_trend_1.5f.png')  # Save plot as a file
plt.show()

# Plot 4: Max energy per episode as a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(max_energy_per_episode.index, max_energy_per_episode.values, color='red', label='Max Energy Per Episode')
plt.xlabel('Episode')
plt.ylabel('Max Energy (kWh)')
plt.title('Max Energy Consumption Per Episode')
plt.legend()
plt.tight_layout()
plt.savefig('max_energy_per_episode_1.5f.png')  # Save plot as a file
plt.show()

# Plot 5: Scatter plot between energy consumption and accident probability
plt.figure(figsize=(12, 6))
plt.scatter(energy_consumption, accident_probability, color='purple', alpha=0.6)
plt.xlabel('Energy Consumption (kWh)')
plt.ylabel('Accident Probability')
plt.title('Scatter Plot of Energy Consumption vs. Accident Probability')
plt.tight_layout()
plt.savefig('energy_vs_accident_probability_1.5f.png')  # Save plot as a file
plt.show()

# Plot 6: Smoothed trend of accident probability
window_size = 50  # You can adjust this window size for more or less smoothing
smoothed_accident_prob = uniform_filter1d(accident_probability, size=window_size)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_accident_prob, label='Smoothed Accident Probability', color='orange')
plt.xlabel('Episode')
plt.ylabel('Accident Probability')
plt.title('Smoothed Trend of Accident Probability Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_accident_probability_1.5f.png')  # Save plot as a file
plt.show()

# Plot 7: Comparison of Smoothed Accident Probability and Random Value
window_size_small = 10  # Original window size for less smoothing
window_size_large = 500  # Larger window size for more smoothing

# Smoothing with a smaller window size
smoothed_accident_prob_small = uniform_filter1d(accident_probability, size=window_size_small)
smoothed_random_value_small = uniform_filter1d(random_value, size=window_size_small)

# Smoothing with a larger window size
smoothed_accident_prob_large = uniform_filter1d(accident_probability, size=window_size_large)
smoothed_random_value_large = uniform_filter1d(random_value, size=window_size_large)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_accident_prob_small, label='Smoothed Accident Probability (window=10)', color='blue', alpha=0.7)
plt.plot(episode, smoothed_random_value_small, label='Smoothed Random Value (window=10)', color='red', alpha=0.7)
plt.plot(episode, smoothed_accident_prob_large, label='Smoothed Accident Probability (window=500)', color='blue', linestyle='--', alpha=0.9)
plt.plot(episode, smoothed_random_value_large, label='Smoothed Random Value (window=500)', color='red', linestyle='--', alpha=0.9)
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Comparison of Smoothed Accident Probability and Random Value with Different Window Sizes')
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_comparison_accident_probability_random_value_1.5f.png')  # Save plot as a file
plt.show()

# Plot 8: Comparison of Accident Probability and Energy Consumption
window_size = 500  # Increase the window size for more smoothing
smoothed_accident_prob = uniform_filter1d(accident_probability, size=window_size)
smoothed_energy_consumption = uniform_filter1d(energy_consumption, size=window_size)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_accident_prob, label='Smoothed Accident Probability', color='blue', alpha=0.7)
plt.plot(episode, smoothed_energy_consumption, label='Smoothed Energy Consumption', color='red', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Comparison of Smoothed Accident Probability and Energy Consumption')
plt.legend()
plt.tight_layout()
plt.savefig('comparison_accident_probability_energy_consumption_1.5f.png')  # Save plot as a file
plt.show()

# Plot 9: Smoothed Energy Consumption Over Training Time
smoothed_energy_consumption = uniform_filter1d(energy_consumption, size=window_size)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_energy_consumption, label='Smoothed Energy Consumption', color='blue', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Over Training Time')
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_energy_consumption_over_time_1.5f.png')  # Save plot as a file
plt.show()

# Create a 3D scatter plot using Plotly
fig = go.Figure(data_sampled=[go.Scatter3d(
    x=energy_consumption,
    y=accident_probability,
    z=episode,
    mode='markers',
    marker=dict(
        size=2,
        color=episode,  # Color by episode
        colorscale='Viridis',  # Colorscale
        opacity=0.7
    )
)])

fig.update_layout(scene = dict(
                    xaxis_title='Energy Consumption (kWh)',
                    yaxis_title='Accident Probability',
                    zaxis_title='Episode'),
                  title='3D Scatter Plot with Plotly')

# Show the plot in a web browser
pio.show(fig)


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
ax.set_title('3D Scatter Plot of Energy Consumption vs. Episode vs. Accident Probability')

# Add a color bar to indicate the episode number
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Episode')

# Show the plot
plt.show()
