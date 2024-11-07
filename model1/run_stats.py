import pandas as pd
import numpy as np
import glob

# Step 1: Read all CSV files into a list of DataFrames
files = glob.glob("/Users/tommasomalaguti/Documents/Python/Tesi/model1/combined_simulation_results_160_nosensors.csv")  # Adjust the path
all_data = [pd.read_csv(f) for f in files]

# Step 2: Concatenate all DataFrames into a single DataFrame
combined_data = pd.concat(all_data, axis=0, ignore_index=True)

# Step 3: Calculate summary statistics
summary_stats = combined_data.groupby(['Step']).agg({
    'Energy Consumption': ['mean', 'std'],
    'Total Distance': ['mean', 'std'],
    'Unloading Cycles': ['mean', 'std'],
    'Total Time': ['mean', 'std'],
    # Add other metrics as needed
}).reset_index()

# Step 4: Create a representative dataset (optional)
# This could be the average run or the median run. Here we'll calculate the average.

# Calculate average values across all runs for each timestep or cycle
representative_run = combined_data.groupby(['Step']).mean().reset_index()

# Step 5: Save the combined data and summary statistics
combined_data.to_csv("combined_simulation_results_nosensors_160.csv", index=False)
summary_stats.to_csv("summary_statistics.csv", index=False)
representative_run.to_csv("representative_run_160_nosensors.csv", index=False)

# Optionally, print some summary statistics for your thesis
print("Summary Statistics:")
print(summary_stats)
print("\nRepresentative Run:")
print(representative_run.head())
