import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for the CSV results
file_paths_2q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_4_qalpha.csv"
]

file_paths_1q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_4_qalpha.csv"
]

file_paths_15q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_4_qalpha.csv"
]

file_paths_3q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_4_qalpha.csv"
]

# Alpha values corresponding to the models
alpha_values = [1, 1.5, 2, 3]

# List to store the data for the table
results_data = []

# Dictionary to hold the file paths corresponding to each alpha value
file_paths_dict = {
    1: file_paths_1q,
    1.5: file_paths_15q,
    2: file_paths_2q,
    3: file_paths_3q
}

# Loop through each alpha value and its corresponding file paths
for alpha, file_paths in file_paths_dict.items():
    avg_reward_list = []
    avg_energy_consumption_list = []
    avg_accident_prob_list = []
    convergence_episode_list = []

    # Loop through each file for the specific alpha value
    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Calculate the average reward per episode
        avg_reward_list.append(df['Reward'].mean())
        
        # Calculate the average energy consumption (kWh) per episode
        avg_energy_consumption_list.append(df['Energy Consumption (kWh)'].mean())
        
        # Calculate the average accident probability
        avg_accident_prob_list.append(df['Accident Probability'].mean())
        
        # Determine the episode of convergence
        rolling_reward_mean = df['Reward'].rolling(window=100).mean()  # moving average over 100 episodes
        convergence_episode_list.append(rolling_reward_mean.idxmax())  # or some custom logic for determining stabilization
    
    # Average the results across the 5 runs for this specific alpha
    avg_reward = sum(avg_reward_list) / len(avg_reward_list)
    avg_energy_consumption = sum(avg_energy_consumption_list) / len(avg_energy_consumption_list)
    avg_accident_prob = sum(avg_accident_prob_list) / len(avg_accident_prob_list)
    convergence_episode = sum(convergence_episode_list) / len(convergence_episode_list)
    
    # Append the data to the results_data list
    results_data.append([alpha, avg_reward, avg_energy_consumption, avg_accident_prob, convergence_episode])

# Create a DataFrame for the results table
results_df = pd.DataFrame(results_data, columns=['Alpha', 'Avg. Reward', 'Avg. Energy Consumption (kWh)', 'Avg. Accident Probability', 'Convergence Episode'])

# Display the table
print(results_df)

# Plotting the results

# Set the theme for the plots
sns.set_theme(style="whitegrid")

# Line Plot: Avg. Reward vs Alpha
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x='Alpha', y='Avg. Reward', marker='o')
plt.title('Average Reward vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Reward')
plt.savefig('av_reward_vs_alpha.pdf', format='pdf', dpi=300)
plt.savefig('av_reward_vs_alpha.png', format='png', dpi=600)
plt.close()

# Line Plot: Avg. Energy Consumption vs Alpha
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x='Alpha', y='Avg. Energy Consumption (kWh)', marker='o', color='g')
plt.title('Average Energy Consumption vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Avg. Energy Consumption (kWh)')
plt.savefig('av_en_cons_vs_alpha.pdf', format='pdf', dpi=300)
plt.savefig('av_en_cons_vs_alpha.png', format='png', dpi=600)
plt.close()

# Line Plot: Avg. Accident Probability vs Alpha
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x='Alpha', y='Avg. Accident Probability', marker='o', color='r')
plt.title('Average Accident Probability vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Avg. Accident Probability')
plt.savefig('av_acc_prob_vs_alpha.pdf', format='pdf', dpi=300)
plt.savefig('av_acc_prob_vs_alpha.png', format='png', dpi=600)
plt.close()

# Line Plot: Convergence Episode vs Alpha
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x='Alpha', y='Convergence Episode', marker='o', color='purple')
plt.title('Convergence Episode vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Convergence Episode')
plt.savefig('conv_ep_vs_alpha.pdf', format='pdf', dpi=300)
plt.savefig('conv_ep_vs_alpha.png', format='png', dpi=600)
plt.close()

# Scatter Plot: Energy Consumption vs Accident Probability (Trade-off)
plt.figure(figsize=(10,6))
sns.scatterplot(data=results_df, x='Avg. Energy Consumption (kWh)', y='Avg. Accident Probability', hue='Alpha', size='Alpha', sizes=(50, 200))
plt.title('Energy Consumption vs Accident Probability (Trade-off)')
plt.xlabel('Avg. Energy Consumption (kWh)')
plt.ylabel('Avg. Accident Probability')
plt.savefig('en_con_vs_acc_prob.pdf', format='pdf', dpi=300)
plt.savefig('en_con_vs_acc_prob.png', format='png', dpi=600)
plt.close()

# Save the results DataFrame to a CSV file
results_df.to_csv('/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/comparison_data/rl_performance_metrics_table.csv', index=False)

# Save the results DataFrame to an Excel (XLSX) file
results_df.to_excel('/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/comparison_data/rl_performance_metrics_table.xlsx', index=False)

print("CSV and XLSX files saved successfully!")


def compare_cumulative_rewards(alpha_files_dict):
    plt.figure(figsize=(10,6))

    # Loop through the different alphas and their corresponding file paths
    for alpha, file_paths in alpha_files_dict.items():
        cumulative_rewards_list = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            
            # Calculate cumulative reward over episodes
            df['Cumulative Reward'] = df['Reward'].cumsum()
            cumulative_rewards_list.append(df['Cumulative Reward'])
        
        # Calculate average cumulative reward for this alpha
        avg_cumulative_reward = pd.DataFrame(cumulative_rewards_list).mean(axis=0)
        
        # Plot the cumulative reward curve for each alpha
        plt.plot(avg_cumulative_reward, label=f'Alpha = {alpha}')
    
    plt.title('Cumulative Reward vs Episodes for Different Training Runs of each Alpha Model')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward (Negative)')
    plt.grid(True)
    plt.legend()
    plt.savefig('cum_rewards.pdf', format='pdf', dpi=300)
    plt.savefig('cum_rewards.png', format='png', dpi=600)
    plt.close()

# Define your alpha file paths dictionary here and call the function
alpha_files_dict = {
    1: file_paths_1q,
    1.5: file_paths_15q,
    2: file_paths_2q,
    3: file_paths_3q
}

compare_cumulative_rewards(alpha_files_dict)

def compare_average_metrics(alpha_files_dict, metric, ylabel, title):
    plt.figure(figsize=(10,6))

    # Loop through the different alphas and their corresponding file paths
    for alpha, file_paths in alpha_files_dict.items():
        avg_metric_values = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            
            # Compute the average value per episode
            avg_metric_values.append(df[metric].mean())
        
        # Plot the average metric for each alpha
        plt.plot(avg_metric_values, label=f'Alpha = {alpha}', marker='o')
    
    plt.title(f'{title} (Average per Alpha) for Different Alpha Values')
    plt.xlabel('Alpha')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{metric.lower().replace(" ", "_")}_vs_alpha.pdf', format='pdf', dpi=300)
    plt.savefig(f'{metric.lower().replace(" ", "_")}_vs_alpha.png', format='png', dpi=600)
    plt.close()

# Define your alpha file paths dictionary here and call the function for each metric
alpha_files_dict = {
    1: file_paths_1q,
    1.5: file_paths_15q,
    2: file_paths_2q,
    3: file_paths_3q
}

# Compare average Energy Consumption
compare_average_metrics(alpha_files_dict, 'Energy Consumption (kWh)', 'Average Energy Consumption (kWh)', 'Average Energy Consumption')

# Compare average Accident Probability
compare_average_metrics(alpha_files_dict, 'Accident Probability', 'Average Accident Probability', 'Average Accident Probability')
