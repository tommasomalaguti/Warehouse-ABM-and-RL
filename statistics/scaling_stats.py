import pandas as pd

# Define the scaling factor
current_cycles = 1000  # Base number of cycles for the current data
target_cycles = 12000  # Target number of cycles to scale to
scaling_factor = target_cycles / current_cycles

# Sample data (replace with your actual data)
data = {
    'Configuration': ['Base Wait Points', 'Base Wait Points', 'Low Wait Points', 'Low Wait Points', 
                      'Moderate Wait Points', 'Moderate Wait Points', 'High Wait Points', 'High Wait Points', 
                      'Maximum Wait Points', 'Maximum Wait Points'],
    'Model Type': ['Non-Optimized', 'Optimized', 'Non-Optimized', 'Optimized', 
                   'Non-Optimized', 'Optimized', 'Non-Optimized', 'Optimized', 
                   'Non-Optimized', 'Optimized'],
    'Operation Time (hours)': [99.8, 87.65, 109.1, 86, 128.5, 88.45, 147.95, 88.82, 162.03, 87.62],
    'Energy Consumption (kWh)': [449.16, 394.43, 491, 387.07, 578.17, 398.07, 665.79, 399.68, 729.13, 394.34],
    'Energy Cost (€)': [123.52, 108.47, 135.03, 106.44, 159, 109.47, 183.09, 109.91, 200.51, 108.44],
    'Driver Salary Cost (€)': [2994.39, 2629.51, 3273.34, 2580.44, 3854.45, 2653.82, 4438.61, 2664.52, 4860.84, 2628.9],
    'Total Variable Costs (€)': [3117.91, 2737.98, 3408.36, 2686.88, 4013.44, 2763.29, 4621.71, 2774.43, 5061.35, 2737.35],
    'GHG Emissions (kg CO2eq)': [152.27, 133.71, 166.45, 131.22, 196, 134.95, 225.7, 135.49, 247.18, 133.68],
    'Total Distance (m)': [52264, 32561, 85402, 37509, 129331, 39654, 181489, 40518, 215484, 35892]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Scale the results
df_scaled = df.copy()
df_scaled['Operation Time (hours)'] *= scaling_factor
df_scaled['Energy Consumption (kWh)'] *= scaling_factor
df_scaled['Energy Cost (€)'] *= scaling_factor
df_scaled['Driver Salary Cost (€)'] *= scaling_factor
df_scaled['Total Variable Costs (€)'] *= scaling_factor
df_scaled['GHG Emissions (kg CO2eq)'] *= scaling_factor
df_scaled['Total Distance (m)'] *= scaling_factor

# Function to convert decimal hours to hours and minutes
def hours_to_hm(x):
    hours = int(x)
    minutes = int((x - hours) * 60)
    return f"{hours}h {minutes}m"

# Function to format numbers without trailing zeros
def format_number(x, unit):
    if x.is_integer():
        return f"{int(x)} {unit}"
    else:
        return f"{x:.2f} {unit}"

# Convert meters to kilometers and format all values
df_scaled['Operation Time (hours)'] = df_scaled['Operation Time (hours)'].apply(hours_to_hm)
df_scaled['Energy Consumption (kWh)'] = df_scaled['Energy Consumption (kWh)'].apply(lambda x: format_number(x, 'kWh'))
df_scaled['Energy Cost (€)'] = df_scaled['Energy Cost (€)'].apply(lambda x: format_number(x, '€'))
df_scaled['Driver Salary Cost (€)'] = df_scaled['Driver Salary Cost (€)'].apply(lambda x: format_number(x, '€'))
df_scaled['Total Variable Costs (€)'] = df_scaled['Total Variable Costs (€)'].apply(lambda x: format_number(x, '€'))
df_scaled['GHG Emissions (kg CO2eq)'] = df_scaled['GHG Emissions (kg CO2eq)'].apply(lambda x: format_number(x, 'kg CO2eq'))
df_scaled['Total Distance (m)'] = (df_scaled['Total Distance (m)'] / 1000).apply(lambda x: format_number(x, 'km'))

# Display the scaled table
print(df_scaled)

# Save the scaled table to a CSV file
df_scaled.to_csv('scaled_results.csv', index=False)

# Save the scaled table to an Excel file
df_scaled.to_excel('scaled_results.xlsx', index=False)

##############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define model directories
model_files = {
    'Model with Base Wait Points': {
        'sensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_sensors_1000_2_model1.csv',
        'nosensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_nosensors_1000_2_model1.csv'
    },
    'Model with Low Wait Points': {
        'sensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_sensors_1000_2_model2.csv',
        'nosensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_nosensors_1000_2_model2.csv'
    },
    'Model with Moderate Wait Points': {
        'sensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_sensors_1000_model3.csv',
        'nosensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_nosensors_1000_model3.csv'
    },
    'Model with High Wait Points': {
        'sensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_sensors_1000_model4.csv',
        'nosensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_nosensors_1000_model4.csv'
    },
    'Model with Maximum Wait Points': {
        'sensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_2_model5.csv',
        'nosensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_nosensors_1000_2_model5.csv'
    }
}

# Create a list to store results
results = []

# Loop through each model
for model_name, files in model_files.items():
    # Load the data
    sensors_df = pd.read_csv(files['sensors'])
    nosensors_df = pd.read_csv(files['nosensors'])

    # Polynomial Regression for Energy Consumption - Sensors
    poly = PolynomialFeatures(degree=1)
    X_sensors = poly.fit_transform(sensors_df[['Unloading Cycles']])  
    y_sensors = sensors_df['Energy Consumption']

    model_sensors = LinearRegression()
    model_sensors.fit(X_sensors, y_sensors)

    # Polynomial Regression for Energy Consumption - No Sensors
    X_nosensors = poly.fit_transform(nosensors_df[['Unloading Cycles']])  
    y_nosensors = nosensors_df['Energy Consumption']

    model_nosensors = LinearRegression()
    model_nosensors.fit(X_nosensors, y_nosensors)

    # Predict for 12,000 cycles
    X_new = poly.transform([[12000]])
    predicted_energy_sensors = model_sensors.predict(X_new)
    predicted_energy_nosensors = model_nosensors.predict(X_new)

    print(f"{model_name} with sensors, 12,000 cycles: {predicted_energy_sensors[0]:.2f} kWh")
    print(f"{model_name} without sensors, 12,000 cycles: {predicted_energy_nosensors[0]:.2f} kWh")

    # Store results in the list
    results.append({
        'Model': model_name,
        'Energy Consumption with Sensors (kWh)': predicted_energy_sensors[0],
        'Energy Consumption without Sensors (kWh)': predicted_energy_nosensors[0]
    })

    # Visualization
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(sensors_df['Unloading Cycles'], y_sensors, color='blue', label='Data with Sensors')
    plt.scatter(nosensors_df['Unloading Cycles'], y_nosensors, color='red', label='Data without Sensors')

    # Generate a range of values for prediction and plotting
    X_range = np.linspace(0, 12000, 300).reshape(-1, 1)  # from 0 to 12,000 cycles

    # Predict using the model for plotting
    y_range_sensors = model_sensors.predict(poly.transform(X_range))
    y_range_nosensors = model_nosensors.predict(poly.transform(X_range))

    # Plot the regression curves
    plt.plot(X_range, y_range_sensors, color='blue', label='Regression with Sensors')
    plt.plot(X_range, y_range_nosensors, color='red', label='Regression without Sensors')

    # Add labels and title
    plt.xlabel('Unloading Cycles')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(f'Energy Consumption vs Unloading Cycles for {model_name}')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('regression_results.csv', index=False)

# Save results to Excel
results_df.to_excel('regression_results.xlsx', index=False)