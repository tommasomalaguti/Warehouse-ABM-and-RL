import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the non-linear model function, e.g., a quadratic function
def polynomial_model(x, a, b, c):
    return a * x**2 + b * x + c

# Define file paths for both sensors and nosensors data
file_paths = {
    'sensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_2_model5.csv',
    'nosensors': '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_nosensors_1000_2_model5.csv'
}

# Dictionary to store fitted parameters for each model
fitted_params = {}

# Loop through both datasets
for key in file_paths:
    # Load the dataset
    df_original = pd.read_csv(file_paths[key])

    # Apply a degradation function to simulate increasing energy consumption
    # due to battery degradation. Here, we'll use a quadratic function for demonstration.
    degradation_factor = 1 + 10e-9 * (df_original['Unloading Cycles'] ** 2)

    # Apply the degradation to the energy consumption
    df_degraded = df_original.copy()
    df_degraded['Energy Consumption'] = df_degraded['Energy Consumption'] * degradation_factor

    # Save the new dataset with the degradation effect
    new_file_path = f'/Users/tommasomalaguti/Documents/Python/Tesi/statistics/new_dataset_with_degradation_{key}.csv'
    df_degraded.to_csv(new_file_path, index=False)

    # Fit the non-linear model to the degraded data
    x_data = df_degraded['Unloading Cycles']
    y_data = df_degraded['Energy Consumption']

    # Use curve_fit to find the best fit parameters for the polynomial model
    params, _ = curve_fit(polynomial_model, x_data, y_data)

    # Store the parameters
    fitted_params[key] = params

    # Use the fitted model to predict energy consumption up to 12,000 cycles
    x_pred = np.arange(1, 120001)  # From 1 to 12,000 cycles
    y_pred = polynomial_model(x_pred, *params)

    # Create a new DataFrame with the predicted values
    df_predicted = pd.DataFrame({
        'Unloading Cycles': x_pred,
        'Predicted Energy Consumption': y_pred
    })

    # Save the predicted data to a new CSV file
    new_file_path_predicted = f'/Users/tommasomalaguti/Documents/Python/Tesi/statistics/predicted_energy_consumption_12000cycles_{key}.csv'
    df_predicted.to_csv(new_file_path_predicted, index=False)

    # Plot the degraded data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(df_degraded['Unloading Cycles'], df_degraded['Energy Consumption'], 'bo', label=f'{key.capitalize()} Degraded Data')
    plt.plot(x_pred, y_pred, 'r-', label=f'{key.capitalize()} Non-Linear Fit (Quadratic)')
    plt.xlabel('Unloading Cycles')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(f'Non-Linear Regression: Energy Consumption with Battery Degradation ({key.capitalize()})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Compare the two models (sensors and nosensors) on the same plot
plt.figure(figsize=(10, 6))

for key in file_paths:
    # Load the predicted data
    df_predicted = pd.read_csv(f'/Users/tommasomalaguti/Documents/Python/Tesi/statistics/predicted_energy_consumption_12000cycles_{key}.csv')

    # Plot the predicted energy consumption
    plt.plot(df_predicted['Unloading Cycles'], df_predicted['Predicted Energy Consumption'], label=f'{key.capitalize()} Model')

plt.xlabel('Unloading Cycles')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Comparison of Energy Consumption with Battery Degradation (Sensors vs. NoSensors)')
plt.legend()
plt.grid(True)
plt.show()
