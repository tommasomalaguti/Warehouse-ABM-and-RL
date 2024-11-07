import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = '/Users/tommasomalaguti/Desktop/Tesi Magistrale/Electricity CO2 Intensity 15.csv'
data = pd.read_csv(file_path)

# Filter the data for Italy and the most recent year available
italy_data = data[data['Member State:text'] == 'Italy'].dropna(subset=['Greenhouse gas (GHG) emission intensity:number'])

# Get the most recent year's data for Italy
most_recent_year_data = italy_data.sort_values(by='Year:year', ascending=False).iloc[0]

# Extract the emission intensity value
emission_intensity = most_recent_year_data['Greenhouse gas (GHG) emission intensity:number']

# Define the energy consumption in kWh
energy_consumption_kwh = 100

# Calculate total emissions in grams of CO2 equivalent
total_emissions_g_co2eq = energy_consumption_kwh * emission_intensity

# Convert grams to kilograms
total_emissions_kg_co2eq = total_emissions_g_co2eq / 1000

print(f"Total emissions for {energy_consumption_kwh} kWh of energy consumption: {total_emissions_kg_co2eq} kg CO2 equivalent")

# Filter the data for Italy
italy_data = data[data['Member State:text'] == 'Italy'].dropna(subset=['Greenhouse gas (GHG) emission intensity:number'])

# Extract year and emission intensity columns
years = italy_data['Year:year']
emission_intensities = italy_data['Greenhouse gas (GHG) emission intensity:number']

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(years, emission_intensities, marker='o', linestyle='-', color='b')
plt.title('GHG Emission Intensity for Electricity in Italy (g CO2eq/kWh)')
plt.xlabel('Year')
plt.ylabel('Emission Intensity (g CO2eq/kWh)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ghg_emissions_italy.pdf', format='pdf', dpi=300)
plt.savefig('ghg_emissions_italy.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
# Display the plot
plt.show()

# Provided data for energy prices in Italy
time_periods = ['2021-S2', '2022-S1', '2022-S2', '2023-S1', '2023-S2']
prices = [0.2170, 0.3011, 0.3907, 0.3031, 0.2740]

# Plotting the graph to recreate the desired visualization
plt.figure(figsize=(8,5))
plt.plot(time_periods, prices, marker='o', color='blue', linestyle='-')
plt.title('Energy Prices for Non-Households in Italy (2021-2023)')
plt.xlabel('Time')
plt.ylabel('Price (EUR/kWh)')
plt.grid(True)
plt.savefig('energy_prices_italy.pdf', format='pdf', dpi=300)
plt.savefig('energy_prices_italy.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
plt.show()

# Generate data for the logistic function
x = np.linspace(-10, 10, 400)
logistic_function = 1 / (1 + np.exp(-x))

# Create the plot
plt.figure(figsize=(8,6))

# Plot the logistic function
plt.plot(x, logistic_function, color='orange', label='Logistic Function', linewidth=2)

# Plot the threshold line at p = 0.5
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (p = 0.5)')

# Plot the vertical line at log-odds = 0
plt.axvline(x=0, color='green', linestyle='--', label='Log-Odds = 0')

# Add titles and labels
plt.title('Logistic Function Curve')
plt.xlabel('Linear Combination of Inputs (Log-Odds)')
plt.ylabel('Probability')

# Display the legend
plt.legend()

# Add grid
plt.grid(True)

plt.savefig('logistic_function.pdf', format='pdf', dpi=300)
plt.savefig('logistic_function.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
# Show the plot
plt.show()

# Generate data for the logistic function
x = np.linspace(-10, 10, 400)
logistic_function = 1 / (1 + np.exp(-x))

# Create the plot with increased figure size for publication
plt.figure(figsize=(8,6))

# Plot the logistic function with a thicker line for better visibility
plt.plot(x, logistic_function, color='blue', label='Logistic Function', linewidth=2.5)

# Plot the threshold line at p = 0.5 with a thinner dashed line
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold (p = 0.5)')

# Plot the vertical line at log-odds = 0 with a thinner dashed line
plt.axvline(x=0, color='green', linestyle='--', linewidth=1.5, label='Log-Odds = 0')

# Add titles and labels with larger font size and LaTeX formatting
plt.title(r'Logistic Function Curve', fontsize=16)
plt.xlabel(r'Linear Combination of Inputs (Log-Odds)', fontsize=14)
plt.ylabel(r'Probability', fontsize=14)

# Display the legend with increased font size, placed outside the plot area for clarity
plt.legend(loc='upper left', fontsize=12)

# Set tick parameters for better visibility
plt.tick_params(axis='both', which='major', labelsize=12)

# Lighten the grid for a cleaner look
plt.grid(True, linestyle=':', linewidth=0.5)

# Save the figure in high resolution and as a PDF
plt.savefig('logistic_function_professional.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('logistic_function_professional.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
# Show the plot
plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt

def accident_probability(weight, unload_time):
    # Normalize the weight and loading_time
    normalized_weight = (weight - 500) / 2500  # Scales to range [0, 1]
    
    # Normalize loading_time, assuming it ranges from 1 to 300 seconds
    normalized_unloading_time = (unload_time - 20) / (300 - 1)  # Scales to range [0, 1]
    
    # Adjust the betas
    beta_weight = 3
    beta_unloading_time = -5  # Negative to decrease probability with more time
    gamma = 0.007

    # Linear combination of inputs
    linear_combination = beta_weight * normalized_weight + beta_unloading_time * normalized_unloading_time - gamma

    # Calculate probability using the sigmoid function
    accident_probability = 1 / (1 + math.exp(-linear_combination))
    
    return accident_probability

# Parameters
weight = 1500  # fixed weight in kg
times = np.linspace(1, 300, 300)  # time range from 1 to 300 seconds

# Calculate accident probability for each time value
probabilities = [accident_probability(weight, t) for t in times]

# Create the plot
plt.figure(figsize=(8,6))
plt.plot(times, probabilities, color='blue', label=f'Loading/Unloading (Weight = {weight} kg)', linewidth=2)

# Add title and labels
plt.title('Accident Probability vs Time for Loading/Unloading')
plt.xlabel('Time (sec)')
plt.ylabel('Accident Probability')

# Display the legend
plt.legend()

# Add grid for better visibility
plt.grid(True)

plt.savefig('accident_probability_load.pdf', format='pdf', dpi=300)
plt.savefig('accident_probability_load.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
# Show the plot
plt.show()

# Accident probability function based on weight and speed for traveling loaded
def accident_probability_traveling_loaded(weight, speed, traveling_loaded_steps=0):
    # Normalize weight and speed
    normalized_weight = (weight - 500) / 2500
    normalized_speed = (speed - 1) / (3 - 1)
    
    # Parameters from the function you provided
    beta_weight = 1.5
    beta_speed = 2
    gamma = 3
    step_multiplier = 0.005
    
    # Linear combination of inputs with step multiplier
    linear_combination = beta_weight * normalized_weight + beta_speed * normalized_speed - gamma + traveling_loaded_steps * step_multiplier
    
    # Calculate probability using the sigmoid function
    accident_probability = 1 / (1 + math.exp(-linear_combination))
    
    return accident_probability

# Parameters
weight = 1500  # fixed weight in kg
speeds = np.linspace(1, 3, 100)  # speed range from 1 to 3 m/s
traveling_loaded_steps = 0  # default value for traveling loaded steps

# Calculate accident probability for each speed value
probabilities = [accident_probability_traveling_loaded(weight, s, traveling_loaded_steps) for s in speeds]

# Create the plot
plt.figure(figsize=(8,6))
plt.plot(speeds, probabilities, color='green', label=f'Traveling Loaded (Weight = {weight} kg)', linewidth=2)

# Add title and labels
plt.title('Accident Probability vs Speed for Traveling Loaded')
plt.xlabel('Speed (m/s)')
plt.ylabel('Accident Probability')

# Display the legend
plt.legend()

# Add grid for better visibility
plt.grid(True)

plt.savefig('accident_probability_speed.pdf', format='pdf', dpi=300)
plt.savefig('accident_probability_speed.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
# Show the plot
plt.show()

# Data for the bar chart
scenarios = ['Best-case', 'Likely-case\nScenarios', 'Worst-case']
energy_consumption = [0.7875, 0.9937, 1.2375]  # kWh values for the three scenarios

# Create the bar chart
plt.figure(figsize=(6,4))
bars = plt.bar(scenarios, energy_consumption, color=['blue', 'orange', 'red'])

# Add labels and title
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Comparison')

# Add both horizontal and vertical grid with increased transparency
plt.grid(axis='both', linestyle='--', alpha=0.3)

plt.savefig('energy_consumption_comparison_real_world.pdf', format='pdf', dpi=300)
plt.savefig('energy_consumption_comparison_real_world.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
# Show the plot
plt.show()

# Data for the bar chart representing time and energy saved
scenarios = ['Best-case', 'Likely-case\nScenarios', 'Worst-case']

# Time saved in minutes (converted from the provided times)
time_saved = [0.25, 3, 6.25]  # in minutes

# Energy saved in kWh
energy_saved = [0.01875, 0.225, 0.46875]  # kWh values for the three scenarios

# Create a subplot with 2 bar charts: one for time saved and one for energy saved
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart for Time Saved
axes[0].bar(scenarios, time_saved, color=['blue', 'orange', 'red'])
axes[0].set_title('Time Saved (in minutes)')
axes[0].set_ylabel('Time Saved (minutes)')
axes[0].grid(axis='both', linestyle='--', alpha=0.3)

# Bar chart for Energy Saved
axes[1].bar(scenarios, energy_saved, color=['blue', 'orange', 'red'])
axes[1].set_title('Energy Saved (in kWh)')
axes[1].set_ylabel('Energy Saved (kWh)')
axes[1].grid(axis='both', linestyle='--', alpha=0.3)

# Display the plot
plt.tight_layout()

plt.savefig('time_and_energy_saved_comparison.pdf', format='pdf', dpi=300)
plt.savefig('time_and_energy_saved_comparison.png', format='png', dpi=600)  # Changed dpi to 600 for higher resolution
plt.show()