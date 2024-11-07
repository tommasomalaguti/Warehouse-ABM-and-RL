import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
import plotly.graph_objs as go
import plotly.io as pio
from scipy.ndimage import uniform_filter1d

# Load the data from the CSV file
csv = "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/forklift_training_sim_results_2alpha_more.csv"
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

# Plot 1: Global energy consumption over time
plt.figure(figsize=(12, 6))
plt.plot(global_energy_consumption.index, global_energy_consumption, label='Global Cumulative Energy Consumption', color='green')
plt.xlabel('Episode')
plt.ylabel('Global Energy Consumption (kWh)')
plt.title('Global Cumulative Energy Consumption over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Max energy per episode as a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(max_energy_per_episode.index, max_energy_per_episode.values, color='red', label='Max Energy Per Episode')
plt.xlabel('Episode')
plt.ylabel('Max Energy (kWh)')
plt.title('Max Energy Consumption Per Episode')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Mean energy per episode
plt.figure(figsize=(12, 6))
plt.plot(mean_energy_per_episode.index, mean_energy_per_episode.values, label='Mean Energy Per Episode', color='blue')
plt.xlabel('Episode')
plt.ylabel('Mean Energy (kWh)')
plt.title('Mean Energy Consumption Per Episode')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Scatter plot between energy consumption and accident probability
plt.figure(figsize=(12, 6))
plt.scatter(energy_consumption, accident_probability, color='purple', alpha=0.6)
plt.xlabel('Energy Consumption (kWh)')
plt.ylabel('Accident Probability')
plt.title('Scatter Plot of Energy Consumption vs. Accident Probability')
plt.tight_layout()
plt.show()

# Plot 5: Smoothed trend of accident probability
window_size = 50  # You can adjust this window size for more or less smoothing
smoothed_accident_prob = uniform_filter1d(accident_probability, size=window_size)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_accident_prob, label='Smoothed Accident Probability', color='orange')
plt.xlabel('Episode')
plt.ylabel('Accident Probability')
plt.title('Smoothed Trend of Accident Probability Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 6: Comparison of Smoothed Accident Probability and Random Value

window_size_small = 10  # Original window size for less smoothing
window_size_large = 500  # Larger window size for more smoothing

# Smoothing with a smaller window size
smoothed_accident_prob_small = uniform_filter1d(accident_probability, size=window_size_small)
smoothed_random_value_small = uniform_filter1d(random_value, size=window_size_small)

# Smoothing with a larger window size
smoothed_accident_prob_large = uniform_filter1d(accident_probability, size=window_size_large)
smoothed_random_value_large = uniform_filter1d(random_value, size=window_size_large)

plt.figure(figsize=(12, 6))

# Plotting the less smoothed data
plt.plot(episode, smoothed_accident_prob_small, label='Smoothed Accident Probability (window=10)', color='blue', alpha=0.7)
plt.plot(episode, smoothed_random_value_small, label='Smoothed Random Value (window=10)', color='red', alpha=0.7)

# Plotting the more smoothed data
plt.plot(episode, smoothed_accident_prob_large, label='Smoothed Accident Probability (window=500)', color='blue', linestyle='--', alpha=0.9)
plt.plot(episode, smoothed_random_value_large, label='Smoothed Random Value (window=500)', color='red', linestyle='--', alpha=0.9)

plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Comparison of Smoothed Accident Probability and Random Value with Different Window Sizes')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 7: Comparison of Accident Probability and Energy Consumption
window_size = 500  # Increase the window size for more smoothing
smoothed_accident_prob = uniform_filter1d(accident_probability, size=window_size)
smoothed_energy_consumption = uniform_filter1d(energy_consumption, size=window_size)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_accident_prob, label='Smoothed Accident Probability', color='blue', alpha=0.7)
plt.plot(episode, smoothed_energy_consumption, label='Smoothed Energy Consumptin', color='red', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Comparison of Smoothed Accident Probability and Energy Consumption')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 7: Smoothed Energy Consumption Over Training Time
smoothed_energy_consumption = uniform_filter1d(energy_consumption, size=window_size)

plt.figure(figsize=(12, 6))
plt.plot(episode, smoothed_energy_consumption, label='Smoothed Energy Consumption', color='blue', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Over Training Time')
plt.legend()
plt.tight_layout()
plt.show()

# Create a 3D scatter plot with Matplotlib
data_sampled = data.sample(frac=0.1, random_state=42)  # Sample 10% of the data for performance
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

# Create a 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
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
