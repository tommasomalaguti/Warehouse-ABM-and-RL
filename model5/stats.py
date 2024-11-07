import pandas as pd

# Load your dataset
data = pd.read_csv("/Users/tommasomalaguti/Documents/Python/Tesi/model5/simulation_results_nosensors_montecarlos_model5.csv")

# Filter the dataset to keep only the last entry for each specified Unloading Cycle
summary_data = data[data['Unloading Cycles'].isin([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])]
summary_data = summary_data.groupby('Unloading Cycles').last().reset_index()

# Function to convert seconds into hours, minutes, and seconds
def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"

# Apply the conversion to the 'Total Time' column
summary_data['Readable Time'] = summary_data['Total Time'].apply(convert_seconds)

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

# Drop the original columns without units if needed
summary_data.drop(columns=['Energy Consumption', 'Total Distance', 'GHG Emissions (g CO2eq)'], inplace=True)

# Save the summary table to an Excel file
summary_data.to_excel("summary_data_nosensors_1000.xlsx", index=False, engine='openpyxl')

# Save the summary table to a CSV file
summary_data.to_csv("summary_data_nosensors_1000.csv", index=False)

print("Summary data has been saved to both summary_data.xlsx and summary_data.csv")
