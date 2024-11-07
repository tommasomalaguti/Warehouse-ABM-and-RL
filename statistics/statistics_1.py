import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for the CSV files
forklift1_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5_twoforks/simulation_results_forklift_1_model5_twoforks.csv'
forklift2_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5_twoforks/simulation_results_forklift_2_model5_twoforks.csv'
merged_csv = 'simulation_results_twoforks_merged.csv'

# Read the CSV files
forklift1_data = pd.read_csv(forklift1_csv)
forklift2_data = pd.read_csv(forklift2_csv)

# Ensure the data is aligned by step
merged_data = pd.merge(forklift1_data, forklift2_data, on='Step', suffixes=('_1', '_2'))

# Sum the relevant columns and take the max for Total Time
summed_data = pd.DataFrame({
    'Step': merged_data['Step'],
    'Energy Consumption': merged_data['Energy Consumption_1'] + merged_data['Energy Consumption_2'],
    'Total Distance': merged_data['Total Distance_1'] + merged_data['Total Distance_2'],
    'Unloading Cycles': merged_data['Unloading Cycles_1'] + merged_data['Unloading Cycles_2'],
    'Total Time': merged_data[['Total Time_1', 'Total Time_2']].max(axis=1)  # Use max instead of sum
})

# Save the merged DataFrame to a new CSV file
summed_data.to_csv(merged_csv, index=False)

# Print a message indicating successful merging
print(f"Merged data saved to {merged_csv}")

####### three forklifts ########
# File paths for the CSV files
forklift1_three_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5_threeforks/simulation_results_forklift_1_model5_threeforks.csv'
forklift2_three_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5_threeforks/simulation_results_forklift_2_model5_threeforks.csv'
forklift3_three_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5_threeforks/simulation_results_forklift_3_model5_threeforks.csv'
merged_three_csv = 'simulation_results_threeforks_merged.csv'

# Read the CSV files
forklift1_three_data = pd.read_csv(forklift1_three_csv)
forklift2_three_data = pd.read_csv(forklift2_three_csv)
forklift3_three_data = pd.read_csv(forklift3_three_csv)

# Ensure the data is aligned by step
merged_three_data = pd.merge(pd.merge(forklift1_three_data, forklift2_three_data, on='Step', suffixes=('_1', '_2')), forklift3_three_data, on='Step')
merged_three_data.rename(columns={
    'Energy Consumption': 'Energy Consumption_3',
    'Total Distance': 'Total Distance_3',
    'Unloading Cycles': 'Unloading Cycles_3',
    'Total Time': 'Total Time_3'
}, inplace=True)

# Sum the relevant columns and take the max for Total Time
summed_three_data = pd.DataFrame({
    'Step': merged_three_data['Step'],
    'Energy Consumption': merged_three_data['Energy Consumption_1'] + merged_three_data['Energy Consumption_2'] + merged_three_data['Energy Consumption_3'],
    'Total Distance': merged_three_data['Total Distance_1'] + merged_three_data['Total Distance_2'] + merged_three_data['Total Distance_3'],
    'Unloading Cycles': merged_three_data['Unloading Cycles_1'] + merged_three_data['Unloading Cycles_2'] + merged_three_data['Unloading Cycles_3'],
    'Total Time': merged_three_data[['Total Time_1', 'Total Time_2', 'Total Time_3']].max(axis=1)  # Use max instead of sum
})

# Save the merged DataFrame to a new CSV file
summed_three_data.to_csv(merged_three_csv, index=False)

# Print a message indicating successful merging
print(f"Merged data saved to {merged_three_csv}")

# File paths
sensors_model1 = '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_sensors_1000_model1.csv'
nosensors_model1 = '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_nosensors_1000_model1.csv'
sensors_model2 = '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_sensors_1000_model2.csv'
nosensors_model2 = '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_nosensors_1000_model2.csv'
sensors_model3 = '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_sensors_1000_model3.csv'
nosensors_model3 = '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_nosensors_1000_model3.csv'
sensors_model4 = '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_sensors_1000_model4.csv'
nosensors_model4 = '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_nosensors_1000_model4.csv'
sensors_model5 = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_model5.csv'
nosensors_model5 = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_nosensors_1000_model5.csv'
nosensors_org_lev2_model4 = '/Users/tommasomalaguti/Documents/Python/Tesi/model4_org_lev2/simulation_results_nosensors_org_lev2_model4.csv'
nosensors_org_lev2_model5 = '/Users/tommasomalaguti/Documents/Python/Tesi/model5_org_lev2/simulation_results_nosensors_org_lev2_model5.csv'
nosensors_twoforks_model5 = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_twoforks_merged.csv'
nosensors_threeforks_model5 = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_threeforks_merged.csv'
sensors_model1_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_sensors_160_model1.csv'
nosensors_model1_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_nosensors_160_model1.csv'
sensors_model2_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_sensors_160_model2.csv'
nosensors_model2_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_nosensors_160_model2.csv'
sensors_model3_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_sensors_160_model3.csv'
nosensors_model3_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_nosensors_160_model3.csv'
sensors_model4_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_sensors_160_model4.csv'
nosensors_model4_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_nosensors_160_model4.csv'
sensors_model5_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_160_model5.csv'
nosensors_model5_160 = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_nosensors_160_model5.csv'


# Check if files exist
file_paths = [
    sensors_model1, nosensors_model1, sensors_model2, nosensors_model2, sensors_model3, nosensors_model3,
    sensors_model4, nosensors_model4, sensors_model5, nosensors_model5, nosensors_org_lev2_model4, nosensors_org_lev2_model5,
    nosensors_twoforks_model5, nosensors_threeforks_model5, sensors_model1_160, nosensors_model1_160, sensors_model2_160, nosensors_model2_160,
    sensors_model3_160, nosensors_model3_160, sensors_model4_160, nosensors_model4_160, sensors_model5_160, nosensors_model5_160
]

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()

# Load the data
sensors_data_model1 = pd.read_csv(sensors_model1)
nosensors_data_model1 = pd.read_csv(nosensors_model1)
sensors_data_model2 = pd.read_csv(sensors_model2)
nosensors_data_model2 = pd.read_csv(nosensors_model2)
sensors_data_model3 = pd.read_csv(sensors_model3)
nosensors_data_model3 = pd.read_csv(nosensors_model3)
sensors_data_model4 = pd.read_csv(sensors_model4)
nosensors_data_model4 = pd.read_csv(nosensors_model4)
sensors_data_model5 = pd.read_csv(sensors_model5)
nosensors_data_model5 = pd.read_csv(nosensors_model5)
nosensors_data_org_lev2_model4 = pd.read_csv(nosensors_org_lev2_model4)
nosensors_data_org_lev2_model5 = pd.read_csv(nosensors_org_lev2_model5)
nosensors_twoforks_model5 = pd.read_csv(nosensors_twoforks_model5)
nosensors_threeforks_model5 = pd.read_csv(nosensors_threeforks_model5)
sensors_model1_data_160 = pd.read_csv(sensors_model1_160)
nosensors_model1_data_160 = pd.read_csv(nosensors_model1_160)
sensors_model2_data_160 = pd.read_csv(sensors_model2_160)
nosensors_model2_data_160 = pd.read_csv(nosensors_model2_160)
sensors_model3_data_160 = pd.read_csv(sensors_model3_160)
nosensors_model3_data_160 = pd.read_csv(nosensors_model3_160)
sensors_model4_data_160 = pd.read_csv(sensors_model4_160)
nosensors_model4_data_160 = pd.read_csv(nosensors_model4_160)
sensors_model5_data_160 = pd.read_csv(sensors_model5_160)
nosensors_model5_data_160 = pd.read_csv(nosensors_model5_160)


# Add a column to identify each model
sensors_data_model1['Model'] = 'Optimized Model with Base Wait Points'
nosensors_data_model1['Model'] = 'Non-Optimized Model with Base Wait Points'
sensors_data_model2['Model'] = 'Optimized Model with Low Wait Points'
nosensors_data_model2['Model'] = 'Non-Optimized Model with Low Wait Points'
sensors_data_model3['Model'] = 'Optimized Model with Moderate Wait Points'
nosensors_data_model3['Model'] = 'Non-Optimized Model with Moderate Wait Points'
sensors_data_model4['Model'] = 'Optimized Model with High Wait Points'
nosensors_data_model4['Model'] = 'Non-Optimized Model with High Wait Points'
sensors_data_model5['Model'] = 'Optimized Model with Maximum Wait Points'
nosensors_data_model5['Model'] = 'Non-Optimized Model with Maximum Wait Points'
sensors_model1_data_160['Model'] = 'Optimized Model with Base Wait Points'
nosensors_model1_data_160['Model'] = 'Non-Optimized Model with Base Wait Points'
sensors_model2_data_160['Model'] = 'Optimized Model with Low Wait Points'
nosensors_model2_data_160['Model'] = 'Non-Optimized Model with Low Wait Points'
sensors_model3_data_160['Model'] = 'Optimized Model with Moderate Wait Points'
nosensors_model3_data_160['Model'] = 'Non-Optimized Model with Moderate Wait Points'
sensors_model4_data_160['Model'] = 'Optimized Model with High Wait Points'
nosensors_model4_data_160['Model'] = 'Non-Optimized Model with High Wait Points'
sensors_model5_data_160['Model'] = 'Optimized Model with Maximum Wait Points'
nosensors_model5_data_160['Model'] = 'Non-Optimized Model with Maximum Wait Points'
nosensors_data_org_lev2_model4['Model'] = 'Non-Optimized Level Enhanced Organization Model with High Wait Points'
nosensors_data_org_lev2_model5['Model'] = 'Non-Optimized Level Enhanced Organization Model with Maximum Wait Points'
nosensors_twoforks_model5['Model'] = 'Two Forklifts Model with Maximum Wait Points'
nosensors_threeforks_model5['Model'] = 'Three Forklifts Model with Maximum Wait Points'

# Combine only the sensor and no sensor models
combined_data = pd.concat([
    sensors_data_model1, nosensors_data_model1, sensors_data_model2, nosensors_data_model2,
    sensors_data_model3, nosensors_data_model3, sensors_data_model4, nosensors_data_model4,
    sensors_data_model5, nosensors_data_model5, nosensors_data_org_lev2_model4, 
    nosensors_data_org_lev2_model5, nosensors_twoforks_model5, nosensors_threeforks_model5
])

# Combine only the sensor and no sensor models
combined_standard_data = pd.concat([
    sensors_data_model1, nosensors_data_model1, sensors_data_model2, nosensors_data_model2,
    sensors_data_model3, nosensors_data_model3, sensors_data_model4, nosensors_data_model4,
    sensors_data_model5, nosensors_data_model5
])

# Combine only the sensor and no sensor models
combined_160_data = pd.concat([
    sensors_model1_data_160, sensors_model2_data_160, sensors_model3_data_160, sensors_model4_data_160, 
    sensors_model5_data_160, nosensors_model1_data_160, nosensors_model2_data_160, nosensors_model3_data_160, 
    nosensors_model4_data_160, nosensors_model5_data_160, 
])

# Sample every 50th cycle
combined_data = combined_data[combined_data['Unloading Cycles'] % 50 == 0]
combined_standard_data = combined_standard_data[combined_standard_data['Unloading Cycles'] % 50 == 0]

# Concatenate the data for sensors and no sensors separately
sensors_combined_data = pd.concat([
    sensors_data_model1, sensors_data_model2, sensors_data_model3, sensors_data_model4, sensors_data_model5
])
nosensors_combined_data = pd.concat([
    nosensors_data_model1, nosensors_data_model2, nosensors_data_model3, nosensors_data_model4,
    nosensors_data_model5, nosensors_data_org_lev2_model4, nosensors_data_org_lev2_model5, 
    nosensors_twoforks_model5, nosensors_threeforks_model5
])

nosensors_standard_combined_data = pd.concat([
    nosensors_data_model1, nosensors_data_model2, nosensors_data_model3, nosensors_data_model4,
    nosensors_data_model5
])
standard_org_combined_data = pd.concat([
    sensors_data_model1, sensors_data_model2, sensors_data_model3, sensors_data_model4, sensors_data_model5, 
    nosensors_data_model1, nosensors_data_model2, nosensors_data_model3, nosensors_data_model4,
    nosensors_data_model5, nosensors_data_org_lev2_model4, nosensors_data_org_lev2_model5
])

# Sample every 50th cycle
sensors_combined_data = sensors_combined_data[sensors_combined_data['Unloading Cycles'] % 50 == 0]
nosensors_combined_data = nosensors_combined_data[nosensors_combined_data['Unloading Cycles'] % 50 == 0]
nosensors_standard_combined_data = nosensors_standard_combined_data[nosensors_standard_combined_data['Unloading Cycles'] % 50 == 0]
standard_org_combined_data = standard_org_combined_data[standard_org_combined_data['Unloading Cycles'] % 50 == 0]

# Extract relevant columns
cycles = combined_data['Unloading Cycles']
energy = combined_data['Energy Consumption']
models = combined_data['Model']

colors = {
    'Optimized Model with Base Wait Points': 'blue',
    'Non-Optimized Model with Base Wait Points': 'orange',
    'Optimized Model with Low Wait Points': 'green',
    'Non-Optimized Model with Low Wait Points': 'red',
    'Optimized Model with Moderate Wait Points': 'purple',
    'Non-Optimized Model with Moderate Wait Points': 'brown',
    'Optimized Model with High Wait Points': 'pink',
    'Non-Optimized Model with High Wait Points': 'teal',
    'Optimized Model with Maximum Wait Points': 'khaki',
    'Non-Optimized Model with Maximum Wait Points': 'magenta',
    'Non-Optimized Level Enhanced Organization Model with High Wait Points': 'cyan',
    'Non-Optimized Level Enhanced Organization Model with Maximum Wait Points': 'black',
    'Two Forklifts Model with Maximum Wait Points': 'slategray',
    'Three Forklifts Model with Maximum Wait Points': 'olive',
    'Optimized Model with Extreme Wait Points': 'navy',
    'Non-Optimized Model with Extreme Wait Points': 'darkorange',
    'Enhanced Optimized Model with Maximum Wait Points': 'turquoise',
    'Enhanced Non-Optimized Model with Maximum Wait Points': 'maroon',
    'Optimized Model with Minimum Wait Points': 'lime',
    'Non-Optimized Model with Minimum Wait Points': 'darkred'
}

# Plot the data: Line plot
plt.figure(figsize=(14, 10))
for model in combined_data['Model'].unique():
    model_data = combined_data[combined_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Optimized and Non-Optimized Models')
plt.legend()
plt.grid(True)
plt.savefig('combined_energy_consumption_vs_cycles_sensors_vs_nosensors.png', dpi=600)
plt.savefig('combined_energy_consumption_vs_cycles_sensors_vs_nosensors.pdf', dpi=300)
plt.close()

# Plot the data: Line plot
plt.figure(figsize=(14, 10))
for model in standard_org_combined_data['Model'].unique():
    model_data = standard_org_combined_data[standard_org_combined_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Standard Models and Enhanced Organization Models')
plt.legend()
plt.grid(True)
plt.savefig('combined_energy_consumption_vs_cycles_standard_sensors_vs_nosensors_org.png', dpi=600)
plt.savefig('combined_energy_consumption_vs_cycles_standard_sensors_vs_nosensors_org.pdf', dpi=300)
plt.close()

# Plot the data: Line plot
plt.figure(figsize=(10, 6))
for model in combined_standard_data['Model'].unique():
    model_data = combined_standard_data[combined_standard_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Optimized and Non-Optimized Standard Models')
plt.legend()
plt.grid(True)
plt.savefig('combined_energy_consumption_vs_cycles_sensors_vs_nosensors_standard.png', dpi=600)
plt.savefig('combined_energy_consumption_vs_cycles_sensors_vs_nosensors_standard.pdf', dpi=300)
plt.close()

# Convert Operation Time from seconds to hours
combined_160_data['Total Time'] = combined_160_data['Total Time'] / 3600

# Round the Total Time to the nearest integer
combined_160_data['Total Time Rounded'] = combined_160_data['Total Time'].round()

# Filter the data to include every 10-hour interval
filtered_combined_160_data = combined_160_data[combined_160_data['Total Time Rounded'] % 10 == 0]

# Plot the data: Line plot
plt.figure(figsize=(10, 6))
for model in filtered_combined_160_data['Model'].unique():
    model_data = filtered_combined_160_data[filtered_combined_160_data['Model'] == model]
    plt.plot(model_data['Total Time'], model_data['Unloading Cycles'], label=model, marker='o', color=colors[model])

plt.xlabel('Operation Time (hours)')
plt.ylabel('Unloading Cycles')
plt.title('Unloading Cycles during a 160 Hours Work Cycle Simulation for Optimized and Non-Optimized Standard Models')
plt.legend()
plt.grid(True)
plt.savefig('cycles_sensors_vs_nosensors_standard_160_hours.png', dpi=600)
plt.savefig('cycles_sensors_vs_nosensors_standard_160_hours.pdf', dpi=300)
plt.close()


# Calculate total energy consumption for each model
total_energy_consumption = combined_data.groupby('Model')['Energy Consumption'].sum()

# Plot the bar chart: Total Energy Consumption
plt.figure(figsize=(10, 6))
total_energy_consumption.plot(kind='bar', color=[colors[model] for model in total_energy_consumption.index])
plt.ylabel('Total Energy Consumption (kWh)')
plt.title('Total Energy Consumption for Sensor and No Sensor Models')
plt.savefig('total_energy_consumption_sensors_vs_nosensors.png', dpi=600)
plt.savefig('total_energy_consumption_sensors_vs_nosensors.pdf', dpi=300)
plt.close()

# Plot the box plot: Energy Consumption Distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x='Model', y='Energy Consumption', data=combined_data, palette=colors)
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Distribution for Sensor and No Sensor Models')
plt.xticks(rotation=45)
plt.savefig('energy_consumption_distribution_sensors_vs_nosensors.png', dpi=600)
plt.savefig('energy_consumption_distribution_sensors_vs_nosensors.pdf', dpi=300)
plt.close()

# Plot the line chart: Energy Consumption vs. Unloading Cycles for No Sensors
plt.figure(figsize=(10, 6))
for model in nosensors_combined_data['Model'].unique():
    model_data = nosensors_combined_data[nosensors_combined_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Nom-Optimized Models')
plt.legend()
plt.grid(True)
plt.savefig('nosensors_energy_consumption_vs_cycles.png', dpi=600)
plt.savefig('nosensors_energy_consumption_vs_cycles.pdf', dpi=300)
plt.close()

# Plot the line chart: Energy Consumption vs. Unloading Cycles for No Sensors
plt.figure(figsize=(10, 6))
for model in nosensors_standard_combined_data['Model'].unique():
    model_data = nosensors_standard_combined_data[nosensors_standard_combined_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Non-Optimized Standard Models')
plt.legend()
plt.grid(True)
plt.savefig('standard_nosensors_energy_consumption_vs_cycles.png', dpi=600)
plt.savefig('standard_nosensors_energy_consumption_vs_cycles.pdf', dpi=300)
plt.close()

# Plot the line chart: Energy Consumption vs. Unloading Cycles for No Sensors
plt.figure(figsize=(10, 6))
for model in sensors_combined_data['Model'].unique():
    model_data = sensors_combined_data[sensors_combined_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Optimized Models')
plt.legend()
plt.grid(True)
plt.savefig('sensors_energy_consumption_vs_cycles.png', dpi=600)
plt.savefig('sensors_energy_consumption_vs_cycles.pdf', dpi=300)
plt.close()

# Filter the data for unloading cycles >= 600
filtered_data = sensors_combined_data[sensors_combined_data['Unloading Cycles'] >= 600]

# Plot the line chart: Energy Consumption vs. Unloading Cycles for No Sensors
plt.figure(figsize=(10, 6))
for model in filtered_data['Model'].unique():
    model_data = filtered_data[filtered_data['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Optimized Models (Cycles >= 600)')
plt.legend()
plt.grid(True)
plt.savefig('sensors_energy_consumption_vs_cycles_filtered.png', dpi=600)
plt.savefig('sensors_energy_consumption_vs_cycles_filtered.pdf', dpi=300)
plt.close()

# Calculate total energy consumption for each model
sensors_total_energy_consumption = sensors_combined_data.groupby('Model')['Energy Consumption'].sum()
nosensors_total_energy_consumption = nosensors_combined_data.groupby('Model')['Energy Consumption'].sum()

# Plot the bar chart: Total Energy Consumption for Sensors
plt.figure(figsize=(10, 6))
sensors_total_energy_consumption.plot(kind='bar', color=[colors[model] for model in sensors_total_energy_consumption.index])
plt.ylabel('Total Energy Consumption (kWh)')
plt.title('Total Energy Consumption for Sensor Models')
plt.savefig('sensors_total_energy_consumption.png', dpi=600)
plt.savefig('sensors_total_energy_consumption.pdf', dpi=300)
plt.close()

# Plot the bar chart: Total Energy Consumption for No Sensors
plt.figure(figsize=(10, 6))
nosensors_total_energy_consumption.plot(kind='bar', color=[colors[model] for model in nosensors_total_energy_consumption.index])
plt.ylabel('Total Energy Consumption (kWh)')
plt.title('Total Energy Consumption for Non-Optimized Models')
plt.savefig('nosensors_total_energy_consumption.png', dpi=600)
plt.savefig('nosensors_total_energy_consumption.pdf', dpi=300)
plt.close()
########## two forklifts vs sensors ##########
# File paths for the CSV files
sensors_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_model5.csv'
nosensors_twoforks_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_twoforks_merged.csv'

# Check if files exist
file_paths = [sensors_model5_csv, nosensors_twoforks_model5_csv]

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()

# Load the data
sensors_data_model5 = pd.read_csv(sensors_model5_csv)
nosensors_twoforks_model5 = pd.read_csv(nosensors_twoforks_model5_csv)

# Add a column to identify each model
sensors_data_model5['Model'] = 'Sensors Model 5'
nosensors_twoforks_model5['Model'] = 'Two Forklifts Model 5'

# Combine the data for comparison
combined_data_twoforks = pd.concat([sensors_data_model5, nosensors_twoforks_model5])

# Sample every 50th cycle
combined_data_twoforks = combined_data_twoforks[combined_data_twoforks['Unloading Cycles'] % 50 == 0]

# Define colors for each model
colors_twoforks = {
    'Sensors Model 5': 'black',
    'Two Forklifts Model 5': 'red'
}

# Plot the data: Line plot for two forklifts vs sensors
plt.figure(figsize=(10, 6))
for model in combined_data_twoforks['Model'].unique():
    model_data = combined_data_twoforks[combined_data_twoforks['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors_twoforks[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Sensors Model 5 and Two Forklifts Model 5')
plt.legend()
plt.grid(True)
plt.savefig('energy_consumption_vs_cycles_sensors_vs_twoforks_model5.pdf', dpi=300, format='pdf')
plt.savefig('energy_consumption_vs_cycles_sensors_vs_twoforks_model5.png', dpi=600)
plt.close()

########## multiple forklifts vs sensors ##############
# File paths for the CSV files
sensors_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_model5.csv'
nosensors_twoforks_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_twoforks_merged.csv'
nosensors_threeforks_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_threeforks_merged.csv'

# Check if files exist
file_paths = [sensors_model5_csv, nosensors_twoforks_model5_csv, nosensors_threeforks_model5_csv]

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()

# Load the data
sensors_data_model5 = pd.read_csv(sensors_model5_csv)
nosensors_twoforks_model5 = pd.read_csv(nosensors_twoforks_model5_csv)
nosensors_threeforks_model5 = pd.read_csv(nosensors_threeforks_model5_csv)

# Add a column to identify each model
sensors_data_model5['Model'] = 'Sensors Model 5'
nosensors_twoforks_model5['Model'] = 'Two Forklifts Model 5'
nosensors_threeforks_model5['Model'] = 'Three Forklifts Model 5'

# Combine the data for comparison
combined_data_threeforks = pd.concat([sensors_data_model5, nosensors_twoforks_model5, nosensors_threeforks_model5])

# Sample every 50th cycle
combined_data_threeforks = combined_data_threeforks[combined_data_threeforks['Unloading Cycles'] % 50 == 0]

# Define colors for each model
colors_threeforks = {
    'Sensors Model 5': 'black',
    'Two Forklifts Model 5': 'red',
    'Three Forklifts Model 5': 'green'
}

# Plot the data: Line plot for three forklifts vs sensors
plt.figure(figsize=(10, 6))
for model in combined_data_threeforks['Model'].unique():
    model_data = combined_data_threeforks[combined_data_threeforks['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=model, marker='o', color=colors_threeforks[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption vs. Unloading Cycles for Sensors Model 5, Two Forklifts Model 5, and Three Forklifts Model 5')
plt.legend()
plt.grid(True)
plt.savefig('energy_consumption_vs_cycles_sensors_vs_twoforks_vs_threeforks_model5.pdf', dpi=300, format='pdf')
plt.savefig('energy_consumption_vs_cycles_sensors_vs_twoforks_vs_threeforks_model5.png', dpi=600)
plt.close()

# Plot the data: Line plot for three forklifts vs sensors (Total Time)
plt.figure(figsize=(10, 6))
for model in combined_data_threeforks['Model'].unique():
    model_data = combined_data_threeforks[combined_data_threeforks['Model'] == model]
    plt.plot(model_data['Unloading Cycles'], model_data['Total Time'], label=model, marker='o', color=colors_threeforks[model])

plt.xlabel('Unloading Cycles')
plt.ylabel('Total Time (h)')
plt.title('Total Time vs. Unloading Cycles for Sensors Model 5, Two Forklifts Model 5, and Three Forklifts Model 5')
plt.legend()
plt.grid(True)
plt.savefig('total_time_vs_cycles_sensors_vs_twoforks_vs_threeforks_model5.pdf', dpi=300, format='pdf')
plt.savefig('total_time_vs_cycles_sensors_vs_twoforks_vs_threeforks_model5.png', dpi=600)
plt.close()


######## three axes ############
# File paths for the CSV files
sensors_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_model5.csv'
nosensors_twoforks_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_twoforks_merged.csv'
nosensors_threeforks_model5_csv = '/Users/tommasomalaguti/Documents/Python/Tesi/simulation_results_threeforks_merged.csv'

# Check if files exist
file_paths = [sensors_model5_csv, nosensors_twoforks_model5_csv, nosensors_threeforks_model5_csv]

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()

# Load the data
sensors_data_model5 = pd.read_csv(sensors_model5_csv)
nosensors_twoforks_model5 = pd.read_csv(nosensors_twoforks_model5_csv)
nosensors_threeforks_model5 = pd.read_csv(nosensors_threeforks_model5_csv)

# Add a column to identify each model
sensors_data_model5['Model'] = 'Sensors Model 5'
nosensors_twoforks_model5['Model'] = 'Two Forklifts Model 5'
nosensors_threeforks_model5['Model'] = 'Three Forklifts Model 5'

# Combine the data for comparison
combined_data_threeforks = pd.concat([sensors_data_model5, nosensors_twoforks_model5, nosensors_threeforks_model5])

# Sample every 50th cycle
combined_data_threeforks = combined_data_threeforks[combined_data_threeforks['Unloading Cycles'] % 50 == 0]

# Define colors for each model
colors_threeforks = {
    'Sensors Model 5': 'black',
    'Two Forklifts Model 5': 'red',
    'Three Forklifts Model 5': 'green'
}

# Plot the data: Line plot for three forklifts vs sensors with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot energy consumption on the left y-axis
ax1.set_xlabel('Unloading Cycles')
ax1.set_ylabel('Energy Consumption (kWh)', color='tab:blue')
for model in combined_data_threeforks['Model'].unique():
    model_data = combined_data_threeforks[combined_data_threeforks['Model'] == model]
    ax1.plot(model_data['Unloading Cycles'], model_data['Energy Consumption'], label=f"{model} Energy", marker='o', color=colors_threeforks[model])

ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second y-axis for total time
ax2 = ax1.twinx()
ax2.set_ylabel('Total Time (h)', color='tab:red')
for model in combined_data_threeforks['Model'].unique():
    model_data = combined_data_threeforks[combined_data_threeforks['Model'] == model]
    ax2.plot(model_data['Unloading Cycles'], model_data['Total Time'], label=f"{model} Time", marker='x', linestyle='--', color=colors_threeforks[model])

ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('Energy Consumption and Total Time vs. Unloading Cycles for Sensors Model 5, Two Forklifts Model 5, and Three Forklifts Model 5')
fig.tight_layout()
plt.savefig('energy_and_time_vs_cycles_sensors_vs_twoforks_vs_threeforks_model5.pdf', dpi=300, format='pdf')
plt.savefig('energy_and_time_vs_cycles_sensors_vs_twoforks_vs_threeforks_model5.png', dpi=600)
plt.close()