import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Define file paths
file_paths = {
    "Model1_Sensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_sensors_1000_model1.csv',
    "Model1_NoSensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_nosensors_1000_model1.csv',
    "Model2_Sensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_sensors_1000_model2.csv',
    "Model2_NoSensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model2/simulation_results_nosensors_1000_model2.csv',
    "Model3_Sensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_sensors_1000_model3.csv',
    "Model3_NoSensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model3/simulation_results_nosensors_1000_model3.csv',
    "Model4_Sensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_sensors_1000_model4.csv',
    "Model4_NoSensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model4/simulation_results_nosensors_1000_model4.csv',
    "Model5_Sensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_sensors_1000_model5.csv',
    "Model5_NoSensors": '/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_nosensors_1000_model5.csv'
}

# Constants for cost and emissions calculations
energy_cost_per_kwh = 0.275  # €/kWh
driver_salary_per_hour = 30  # €/hour
emission_intensity = 390  # g CO2eq/kWh

# Define descriptive labels for models
descriptive_labels = {
    1: "Base Wait Points",
    2: "Low Wait Points",
    3: "Moderate Wait Points",
    4: "High Wait Points",
    5: "Maximum Wait Points"
}

# Load data and add metadata columns
final_results = []

for model_number in range(1, 6):
    plt.figure(figsize=(10, 6))
    for optimization in ['Sensors', 'NoSensors']:
        key = f"Model{model_number}_{optimization}"
        path = file_paths[key]
        df = pd.read_csv(path)
        
        # Multiple Linear Regression: Unloading Cycles, Total Time, and Distance to predict Energy Consumption
        X = df[['Unloading Cycles', 'Total Time', 'Total Distance']].values
        y = df['Energy Consumption'].values
        
        # Add a constant (intercept) to the independent variables
        X_with_constant = sm.add_constant(X)
        
        # Fit the model using statsmodels for detailed statistics
        model = sm.OLS(y, X_with_constant).fit()
        
        # Print model summary to get statistical data
        print(f"Statistical Summary for {descriptive_labels[model_number]} ({'Optimized' if optimization == 'Sensors' else 'Non-Optimized'})")
        print(model.summary())
        
        # Predict values up to 12,000 cycles
        max_cycles = 12000
        last_total_time = df['Total Time'].iloc[-1]
        last_total_distance = df['Total Distance'].iloc[-1]
        
        avg_time_increment = (df['Total Time'].iloc[-1] - df['Total Time'].iloc[0]) / (df['Unloading Cycles'].iloc[-1] - df['Unloading Cycles'].iloc[0])
        avg_distance_increment = (df['Total Distance'].iloc[-1] - df['Total Distance'].iloc[0]) / (df['Unloading Cycles'].iloc[-1] - df['Unloading Cycles'].iloc[0])
        
        X_new = np.array([
            [
                i, 
                last_total_time + (i - df['Unloading Cycles'].iloc[-1]) * avg_time_increment, 
                last_total_distance + (i - df['Unloading Cycles'].iloc[-1]) * avg_distance_increment
            ]
            for i in range(1, max_cycles + 1)
        ])
        X_new_with_constant = sm.add_constant(X_new)
        y_new = model.predict(X_new_with_constant)
        
        # Store the final result at 12,000 cycles
        final_cycles = 12000
        predicted_energy = y_new[-1]
        predicted_total_time_seconds = X_new[-1, 1]
        predicted_total_distance_meters = X_new[-1, 2]
        
        # Convert results to hours and kilometers
        predicted_total_time_hours = predicted_total_time_seconds / 3600
        predicted_total_distance_km = predicted_total_distance_meters / 1000
        
        # Calculate additional costs and emissions
        energy_cost = predicted_energy * energy_cost_per_kwh
        driver_salary_cost = predicted_total_time_hours * driver_salary_per_hour
        total_variable_costs = energy_cost + driver_salary_cost
        ghg_emissions = predicted_energy * emission_intensity / 1000  # Convert g CO2eq to kg CO2eq
        
        final_results.append({
            'Model': f'Model{model_number}',
            'Optimization': optimization,
            'Unloading Cycles': final_cycles,
            'Predicted Energy Consumption (kWh)': round(predicted_energy, 2),
            'Predicted Total Time (Hours)': round(predicted_total_time_hours, 2),
            'Predicted Total Distance (Km)': round(predicted_total_distance_km, 2),
            'Energy Cost (€)': round(energy_cost, 2),
            'Driver Salary Cost (€)': round(driver_salary_cost, 2),
            'Total Variable Costs (€)': round(total_variable_costs, 2),
            'GHG Emissions (kg CO2eq)': round(ghg_emissions, 2)
        })
        
        # Create DataFrame for the predicted data and save as CSV
        predicted_df = pd.DataFrame({
            'Unloading Cycles': X_new[:, 0],
            'Predicted Energy Consumption (kWh)': y_new,
            'Predicted Total Time (Seconds)': X_new[:, 1],
            'Predicted Total Distance (Meters)': X_new[:, 2]
        })
        predicted_df.to_csv(f'/Users/tommasomalaguti/Documents/Python/Tesi/statistics/{key}_predicted_12000cycles.csv', index=False)
        
        # Generate a descriptive label for the plot
        descriptive_label = f"{'Optimized' if optimization == 'Sensors' else 'Non-Optimized'} Model with {descriptive_labels[model_number]}"
        
        # Plot actual and predicted energy consumption
        plt.plot(df['Unloading Cycles'], df['Energy Consumption'], 'o', label=f'{descriptive_label} (Actual)')
        plt.plot(X_new[:, 0], y_new, '-', label=f'{descriptive_label} (Predicted)')
    
    # Set the title using the descriptive label
    plt.title(f'Energy Consumption Comparison for Model with {descriptive_labels[model_number]}')
    plt.xlabel('Unloading Cycles')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/Users/tommasomalaguti/Documents/Python/Tesi/statistics/Model{model_number}_comparison_12000cycles.png')
    plt.show()

# Convert the final results to a DataFrame
final_results_df = pd.DataFrame(final_results)

# Check the headers to ensure units are included
print(final_results_df.columns)

# Save to Excel with appropriate units in the column names
excel_path = '/Users/tommasomalaguti/Documents/Python/Tesi/statistics/final_results_12000cycles.xlsx'
final_results_df.to_excel(excel_path, index=False)

# Display the final results to the user
final_results_df
