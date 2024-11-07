from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
import os

# File paths for different alpha values
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

file_paths_2q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_2/forklift_training_sim_results_2_4_qalpha.csv"
]

file_paths_3q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_3/forklift_training_sim_results_3_4_qalpha.csv"
]

# Dictionary of alpha file paths
alpha_files_dict = {
    1: file_paths_1q,
    1.5: file_paths_15q,
    2: file_paths_2q,
    3: file_paths_3q
}

# Output directories
output_directory = "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/comparison_data"
os.makedirs(output_directory, exist_ok=True)

# Ensure the output directory has a trailing slash
if not output_directory.endswith('/'):
    output_directory += '/'

# Function to check for Pareto efficiency
def is_pareto_efficient(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True
    return is_efficient if return_mask else np.where(is_efficient)[0]

# Function to plot Pareto front
def plot_pareto_front(alpha_files_dict, output_directory):
    avg_energy_consumption = []
    avg_accident_prob = []
    alpha_values = []
    
    for alpha, file_paths in alpha_files_dict.items():
        combined_energy_values = []
        combined_accident_values = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            combined_energy_values.extend(df['Energy Consumption (kWh)'])
            combined_accident_values.extend(df['Accident Probability'])
        
        avg_energy_consumption.append(np.mean(combined_energy_values))
        avg_accident_prob.append(np.mean(combined_accident_values))
        alpha_values.append(alpha)
    
    costs = np.array(list(zip(avg_energy_consumption, avg_accident_prob)))
    pareto_efficient_mask = is_pareto_efficient(costs)
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=avg_energy_consumption, y=avg_accident_prob, hue=alpha_values, s=100, palette="deep", legend="full")
    plt.scatter(np.array(avg_energy_consumption)[pareto_efficient_mask], np.array(avg_accident_prob)[pareto_efficient_mask], color='red', s=150, label='Pareto Efficient Points')
    plt.title('Pareto Front: Energy Consumption vs Accident Probability')
    plt.xlabel('Average Energy Consumption (kWh)')
    plt.ylabel('Average Accident Probability')
    plt.grid(True)
    plt.legend(title="Alpha", loc='best')
    plt.savefig(f"{output_directory}pareto_front.pdf", format='pdf', dpi=300)
    plt.savefig(f"{output_directory}pareto_front.png", format='png', dpi=600)
    plt.close()

# Function to plot learning curves
def plot_learning_curves(alpha_files_dict, smoothing_window=500):
    plt.figure(figsize=(10,6))
    
    line_width = 2 

    for idx, (alpha, file_paths) in enumerate(alpha_files_dict.items()):
        rewards_list = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            smoothed_reward = df['Reward'].rolling(window=smoothing_window).mean()
            rewards_list.append(smoothed_reward)
        
        avg_reward = pd.concat(rewards_list, axis=1).mean(axis=1)
        plt.plot(avg_reward, label=f'Alpha = {alpha}', linewidth=line_width)
    
    plt.title('Learning Curves: Reward vs Episodes for Different Alphas')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f"{output_directory}learning_curves.pdf", format='pdf', dpi=300)
    plt.savefig(f"{output_directory}learning_curves.png", format='png', dpi=600)
    plt.close()



# Plot Pareto front
plot_pareto_front(alpha_files_dict, output_directory)

# Plot learning curves
plot_learning_curves(alpha_files_dict)

def plot_tradeoff(alpha_files_dict):
    avg_energy_consumption = []
    avg_accident_prob = []
    alpha_values = []
    
    # Loop through the different alphas and combine the data for trade-off analysis
    for alpha, file_paths in alpha_files_dict.items():
        combined_energy_values = []
        combined_accident_values = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            combined_energy_values.extend(df['Energy Consumption (kWh)'])
            combined_accident_values.extend(df['Accident Probability'])
        
        # Calculate average energy consumption and accident probability for this alpha
        avg_energy_consumption.append(sum(combined_energy_values) / len(combined_energy_values))
        avg_accident_prob.append(sum(combined_accident_values) / len(combined_accident_values))
        alpha_values.append(alpha)
    
    # Plot the trade-off between Energy Consumption and Accident Probability
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=avg_energy_consumption, y=avg_accident_prob, hue=alpha_values, s=100, palette="deep", legend="full")
    plt.title('Trade-off between Energy Consumption and Accident Probability')
    plt.xlabel('Average Energy Consumption (kWh)')
    plt.ylabel('Average Accident Probability')
    plt.grid(True)
    plt.legend(title="Alpha", loc='best')
    plt.savefig(f"{output_directory}tradeoff_plot_1.pdf", format='pdf', dpi=300)
    plt.savefig(f"{output_directory}tradeoff_plot_1.png", format='png', dpi=600)
    plt.close()

# Plot the trade-off
plot_tradeoff(alpha_files_dict)