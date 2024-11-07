import pandas as pd

# Load your dataset
data = pd.read_csv("/Users/tommasomalaguti/Documents/Python/Tesi/model1/simulation_results_sensors_1000_model1.csv")

# Filter the dataset to keep only the last entry for each specified Unloading Cycle
summary_data = data[data['Unloading Cycles'].isin([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])]
summary_data = summary_data.groupby('Unloading Cycles').last().reset_index()

# Function to convert seconds into hours, minutes, and seconds
def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"

# Apply the conversion to the 'Total Time' column and store as 'Operation Time'
summary_data['Operation Time'] = summary_data['Total Time'].apply(convert_seconds)

# Round the numerical data
summary_data['Energy Consumption'] = summary_data['Energy Consumption'].round(2)

# Energy Cost Calculation
energy_price_per_kWh = 0.275  # Euro per kWh
summary_data['Energy Cost (€)'] = summary_data['Energy Consumption'] * energy_price_per_kWh

# GHG Emissions Calculation
emission_intensity = 339  # g CO2eq/kWh, example value for Italy
summary_data['GHG Emissions (g CO2eq)'] = summary_data['Energy Consumption'] * emission_intensity

# Convert GHG emissions to kilograms for easier readability
summary_data['GHG Emissions (kg CO2eq)'] = summary_data['GHG Emissions (g CO2eq)'] / 1000

# Driver Salary Cost Calculation
driver_salary_per_hour = 30  # Euro per hour
summary_data['Driver Salary Cost (€)'] = (summary_data['Total Time'] / 3600) * driver_salary_per_hour

# Total Variable Costs Calculation
summary_data['Total Variable Costs (€)'] = summary_data['Energy Cost (€)'] + summary_data['Driver Salary Cost (€)']

# Remove decimals from 'Total Distance' and convert to integer
summary_data['Total Distance (m)'] = summary_data['Total Distance'].astype(int).apply(lambda x: f"{x} m")

# Add units to the columns
summary_data['Energy Consumption (kWh)'] = summary_data['Energy Consumption'].apply(lambda x: f"{x} kWh")
summary_data['Energy Cost (€)'] = summary_data['Energy Cost (€)'].apply(lambda x: f"{x:.2f} €")
summary_data['GHG Emissions (kg CO2eq)'] = summary_data['GHG Emissions (kg CO2eq)'].apply(lambda x: f"{x:.2f} kg CO2eq")
summary_data['Driver Salary Cost (€)'] = summary_data['Driver Salary Cost (€)'].apply(lambda x: f"{x:.2f} €")
summary_data['Total Variable Costs (€)'] = summary_data['Total Variable Costs (€)'].apply(lambda x: f"{x:.2f} €")

# Reorder the columns and exclude 'Step'
summary_data = summary_data[[
    'Unloading Cycles', 
    'Operation Time', 
    'Energy Consumption (kWh)', 
    'Energy Cost (€)', 
    'Driver Salary Cost (€)', 
    'Total Variable Costs (€)', 
    'GHG Emissions (kg CO2eq)', 
    'Total Distance (m)'
]]

# Save the summary table to an Excel file
summary_data.to_excel("summary_data_sensors_1000.xlsx", index=False, engine='openpyxl')

# Save the summary table to a CSV file
summary_data.to_csv("summary_data_sensors_1000.csv", index=False)

print("Summary data has been saved to both summary_data.xlsx and summary_data.csv")
