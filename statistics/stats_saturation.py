import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to convert time from seconds to "Xh Ym" format
def convert_seconds_to_hm(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{int(hours)}h {int(minutes)}m"

# Define a function to load, plot the data, and compile results into a DataFrame
def plot_and_compile_results(csv_files, title, model_type, configuration):
    step_intervals = []
    max_unloading_cycles = []
    final_times = []

    # Load each CSV file
    for file in csv_files:
        # Extract the step interval from the filename
        step_part = [part for part in file.split('_') if 'steps' in part][0]
        step_interval = int(step_part.replace('steps', '').replace('model5.csv', ''))
        step_intervals.append(step_interval)

        # Load the CSV and extract the maximum unloading cycle value and final time
        df = pd.read_csv(file)
        max_cycle = df['Unloading Cycles'].max()  # Adjust this depending on how unloading cycles are recorded
        final_time_seconds = df['Total Time'].iloc[-1]  # Assuming 'Total Time' is cumulative and last entry is the final time

        # Convert final time to hours and minutes
        final_time_hm = convert_seconds_to_hm(final_time_seconds)
        final_times.append(final_time_hm)

        max_unloading_cycles.append(max_cycle)

    # Compile the results into a DataFrame
    results_df = pd.DataFrame({
        'Model Type': model_type,
        'Configuration': configuration,
        'Step': step_intervals,
        'Max Unloading Cycles': max_unloading_cycles,
        'Final Time': final_times
    })

    # Plot the data
    plt.figure(figsize=(12, 8))

    # Bar chart
    plt.bar(step_intervals, max_unloading_cycles, color='skyblue', alpha=0.6, label='Bar Chart')

    # Line chart
    plt.plot(step_intervals, max_unloading_cycles, marker='o', color='red', label='Line Chart')

    plt.title(title, fontsize=16)
    plt.xlabel('Step Interval (Number of Steps)', fontsize=14)
    plt.ylabel('Maximum Unloading Cycles', fontsize=14)
    plt.gca().invert_xaxis()  # Invert the x-axis to have steps decreasing from left to right
    plt.grid(True)
    plt.xticks(step_intervals, fontsize=12)  # Ensure all step intervals are shown on the x-axis
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    # Save the plot as PDF with 300 DPI and PNG with 72 DPI
    plt.savefig(f"{title.replace(' ', '_').lower()}.pdf", format='pdf', dpi=300)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", format='png', dpi=600)
    
    plt.show()

    return step_intervals, max_unloading_cycles, final_times, results_df

# Function to plot combined graph for both models and compile combined results
def plot_combined_graph_and_compile(nosensors_data, sensors_data, title):
    step_intervals_nosensors, max_unloading_cycles_nosensors, final_times_nosensors, _ = nosensors_data
    step_intervals_sensors, max_unloading_cycles_sensors, final_times_sensors, _ = sensors_data

    plt.figure(figsize=(12, 8))

    # Bar chart for No Sensors
    plt.bar([x - 0.2 for x in step_intervals_nosensors], max_unloading_cycles_nosensors, width=0.4, color='skyblue', alpha=0.6, label='Non-Optimized - Bar Chart')

    # Bar chart for Sensors
    plt.bar([x + 0.2 for x in step_intervals_sensors], max_unloading_cycles_sensors, width=0.4, color='lightgreen', alpha=0.6, label='Optimized - Bar Chart')

    # Line chart for No Sensors
    plt.plot(step_intervals_nosensors, max_unloading_cycles_nosensors, marker='o', color='blue', label='Non-Optimized - Line Chart')

    # Line chart for Sensors
    plt.plot(step_intervals_sensors, max_unloading_cycles_sensors, marker='o', color='green', label='Optimized - Line Chart')

    plt.title(title)
    plt.xlabel('Step Interval (Number of Steps)')
    plt.ylabel('Maximum Unloading Cycles')
    plt.gca().invert_xaxis()  # Invert the x-axis to have steps decreasing from left to right
    plt.grid(True)
    plt.xticks(step_intervals_nosensors + step_intervals_sensors)
    plt.legend()

    # Save the plot as PDF with 300 DPI and PNG with 72 DPI
    plt.savefig(f"{title.replace(' ', '_').lower()}.pdf", format='pdf', dpi=300)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", format='png', dpi=600)

    plt.show()

# Define the CSV files for sensors and no sensors
nosensors_files = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_38steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_37steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_36steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_35steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_34steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_33steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_32steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_31steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_30steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_20steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_nosensors_saturation_10steps_model5.csv"
    # Add the rest of your files in similar fashion
]

sensors_files = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_20steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_19steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_18steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_17steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_16steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_15steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_14steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_13steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_12steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_11steps_model5.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_saturation/simulation_results_sensors_saturation_10steps_model5.csv"
    # Add the rest of your files in similar fashion
]

# Plot for No Sensors Model and compile results
nosensors_data = plot_and_compile_results(nosensors_files, 'Maximum Unloading Cycles for Non-Optimized Models', 'Maximum Wait Points', 'Non-Optimized')

# Plot for Sensors Model and compile results
sensors_data = plot_and_compile_results(sensors_files, 'Maximum Unloading Cycles for Optimized Models', 'Maximum Wait Points', 'Optimized')

# Combined Plot for both Models
plot_combined_graph_and_compile(nosensors_data, sensors_data, 'Comparison of Maximum Unloading Cycles under Stress Conditions for Optimized and Non-Optimized Models')

# Combine the results into a single DataFrame
combined_results = pd.concat([nosensors_data[3], sensors_data[3]], ignore_index=True)

# Save the combined results to CSV and Excel
combined_results.to_csv('combined_results.csv', index=False)
combined_results.to_excel('combined_results.xlsx', index=False)
