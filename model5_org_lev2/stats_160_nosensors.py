import pandas as pd

# Load your dataset
data = pd.read_csv("/Users/tommasomalaguti/Documents/Python/Tesi/model5_org_lev2/simulation_results_nosensors_160_model5_org_lev2.csv")

# Convert the 'Total Time' to hours (assuming 'Total Time' is in seconds)
data['Total Time (hours)'] = data['Total Time'] / 3600

# Define intervals for filtering (e.g., every 10 hours)
time_intervals = range(10, 161, 10)  # 10, 20, 30, ..., 160 hours

# Initialize an empty DataFrame to collect the filtered data
summary_data = pd.DataFrame()

# Loop through the desired intervals and find the closest matching 'Total Time (hours)'
for interval in time_intervals:
    closest_row = data.iloc[(data['Total Time (hours)'] - interval).abs().argsort()[:1]]
    summary_data = pd.concat([summary_data, closest_row])

# Reset the index of the summary_data DataFrame
summary_data.reset_index(drop=True, inplace=True)

# Function to convert seconds into hours, minutes, and seconds
def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"

# Apply the conversion to the 'Total Time' column
summary_data['Operation Time'] = summary_data['Total Time'].apply(convert_seconds)

# Round the numerical data
summary_data['Energy Consumption'] = summary_data['Energy Consumption'].round(2)
summary_data['Total Distance'] = summary_data['Total Distance']
summary_data['Total Time'] = summary_data['Total Time'].round(2)

# Energy Cost Calculation
energy_price_per_kWh = 0.275  # Euro per kWh
summary_data['Energy Cost (€)'] = summary_data['Energy Consumption'] * energy_price_per_kWh

# GHG Emissions Calculation
emission_intensity = 339  # g CO2eq/kWh, example value for Italy
summary_data['GHG Emissions (g CO2eq)'] = summary_data['Energy Consumption'] * emission_intensity

# Convert GHG emissions to kilograms for easier readability
summary_data['GHG Emissions (kg CO2eq)'] = summary_data['GHG Emissions (g CO2eq)'] / 1000

# Add units to the columns
summary_data['Energy Consumption (kWh)'] = summary_data['Energy Consumption'].apply(lambda x: f"{x} kWh")
summary_data['Total Distance (m)'] = summary_data['Total Distance'].apply(lambda x: f"{x} m")
summary_data['Energy Cost (€)'] = summary_data['Energy Cost (€)'].apply(lambda x: f"{x:.2f} €")
summary_data['GHG Emissions (kg CO2eq)'] = summary_data['GHG Emissions (kg CO2eq)'].apply(lambda x: f"{x:.2f} kg CO2eq")

# Drop the original columns without units and the 'Total Time' and 'Total Time (hours)' columns
summary_data.drop(columns=['Energy Consumption', 'Total Distance', 'GHG Emissions (g CO2eq)', 'Total Time', 'Total Time (hours)'], inplace=True)

# Save the summary table to an Excel file
summary_data.to_excel("summary_data_160h_nosensors.xlsx", index=False, engine='openpyxl')

# Save the summary table to a CSV file
summary_data.to_csv("summary_data_160h_nosensors.csv", index=False)

print("Summary data has been saved to both summary_data_160h.xlsx and summary_data_160h.csv")