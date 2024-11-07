from scipy import stats
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_paths_1q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1/forklift_training_sim_results_1_4_qalpha.csv"
]
file_paths_125q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.25/forklift_training_sim_results_1.25_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.25/forklift_training_sim_results_1.25_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.25/forklift_training_sim_results_1.25_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.25/forklift_training_sim_results_1.25_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.25/forklift_training_sim_results_1.25_4_qalpha.csv"
]

file_paths_15q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.5/forklift_training_sim_results_1.5_4_qalpha.csv"
]

file_paths_175q = [
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.75/forklift_training_sim_results_1.75_0_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.75/forklift_training_sim_results_1.75_1_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.75/forklift_training_sim_results_1.75_2_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.75/forklift_training_sim_results_1.75_3_qalpha.csv",
    "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/results_data_1.75/forklift_training_sim_results_1.75_4_qalpha.csv"
]

# File paths for the CSV results
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

# Alpha values corresponding to the models
alpha_values = [1, 1.25, 1.5, 1.75, 2, 3]

# List to store the data for the table
results_data = []

# Dictionary to hold the file paths corresponding to each alpha value
file_paths_dict = {
    1: file_paths_1q,
    1.25: file_paths_125q,
    1.5: file_paths_15q,
    1.75: file_paths_175q,
    2: file_paths_2q,
    3: file_paths_3q
}

# Define your alpha file paths dictionary here and call the function for each metric
alpha_files_dict = {
    1: file_paths_1q,
    1.25: file_paths_125q,
    1.5: file_paths_15q,
    1.75: file_paths_175q,
    2: file_paths_2q,
    3: file_paths_3q
}

# Function to perform ANOVA on a specific metric
def perform_anova(alpha_files_dict, metric):
    # Collect all metric data grouped by alpha
    metric_data = []
    for alpha, file_paths in alpha_files_dict.items():
        alpha_metric_values = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            alpha_metric_values.extend(df[metric])
        metric_data.append(alpha_metric_values)
    
    # Perform ANOVA
    f_statistic, p_value = stats.f_oneway(*metric_data)
    print(f'ANOVA Results for {metric}: F-statistic = {f_statistic}, p-value = {p_value}')
    
    # Interpretation of the result
    if p_value < 0.05:
        print(f'There is a significant difference in {metric} between different alpha values.')
    else:
        print(f'There is no significant difference in {metric} between different alpha values.')

# Perform ANOVA for Energy Consumption
perform_anova(alpha_files_dict, 'Energy Consumption (kWh)')

# Perform ANOVA for Accident Probability
perform_anova(alpha_files_dict, 'Accident Probability')

import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perform_tukey_test(alpha_files_dict, metric, metric_name):
    # Prepare data for Tukey's HSD test
    all_values = []
    alpha_groups = []
    
    # Combine data from all alphas
    for alpha, file_paths in alpha_files_dict.items():
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            all_values.extend(df[metric])
            alpha_groups.extend([alpha] * len(df[metric]))
    
    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=all_values, groups=alpha_groups, alpha=0.05)
    print(f"\nTukey's HSD Test Results for {metric_name}:\n")
    print(tukey_result)

# Perform Tukey's HSD test for Energy Consumption
perform_tukey_test(alpha_files_dict, 'Energy Consumption (kWh)', 'Energy Consumption (kWh)')

# Perform Tukey's HSD test for Accident Probability
perform_tukey_test(alpha_files_dict, 'Accident Probability', 'Accident Probability')

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
    plt.savefig('tradeoff_plot.pdf', format='pdf', dpi=300)
    plt.close()

# Plot the trade-off
plot_tradeoff(alpha_files_dict)

import numpy as np

def is_pareto_efficient(costs, return_mask=True):
    """Find Pareto efficient points. Return indices or a mask."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Mask points dominated by the current point
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True  # Keep current point
    return is_efficient if return_mask else np.where(is_efficient)[0]

def plot_pareto_front(alpha_files_dict):
    avg_energy_consumption = []
    avg_accident_prob = []
    alpha_values = []
    
    # Collect averages
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
    
    # Find Pareto-efficient points
    pareto_efficient_mask = is_pareto_efficient(costs)
    
    # Plot all points
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=avg_energy_consumption, y=avg_accident_prob, hue=alpha_values, s=100, palette="deep", legend="full")
    
    # Highlight Pareto-efficient points
    plt.scatter(np.array(avg_energy_consumption)[pareto_efficient_mask], np.array(avg_accident_prob)[pareto_efficient_mask], color='red', s=150, label='Pareto Efficient Points')
    
    plt.title('Pareto Front: Energy Consumption vs Accident Probability')
    plt.xlabel('Average Energy Consumption (kWh)')
    plt.ylabel('Average Accident Probability')
    plt.grid(True)
    plt.legend(title="Alpha", loc='best')
    plt.savefig('pareto_front.pdf', format='pdf', dpi=300)
    plt.close()

# Run the updated Pareto front analysis
plot_pareto_front(alpha_files_dict)


def plot_learning_curves(alpha_files_dict, smoothing_window=500):
    plt.figure(figsize=(10,6))
    
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']  # Different line styles for each alpha
    line_width = 2  # Increase line width for better visibility

    # Loop through the different alphas and their corresponding file paths
    for idx, (alpha, file_paths) in enumerate(alpha_files_dict.items()):
        rewards_list = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            # Apply rolling mean for smoothing the reward curve
            smoothed_reward = df['Reward'].rolling(window=smoothing_window).mean()
            rewards_list.append(smoothed_reward)
        
        # Calculate average reward across runs for this alpha
        avg_reward = pd.concat(rewards_list, axis=1).mean(axis=1)
        
        # Plot the learning curve with adjusted style and width
        plt.plot(avg_reward, label=f'Alpha = {alpha}', linestyle=line_styles[idx % len(line_styles)], linewidth=line_width)
    
    plt.title('Learning Curves: Reward vs Episodes for Different Alphas')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig('learning_curves.pdf', format='pdf', dpi=300)
    plt.close()

# Adjust the window size for smoothing (larger window -> more smoothing)
plot_learning_curves(alpha_files_dict, smoothing_window=1200)

# Plot learning curves
plot_learning_curves(alpha_files_dict)


def plot_learning_curves_subplots(alpha_files_dict, smoothing_window=500):
    fig, axes = plt.subplots(len(alpha_files_dict), 1, figsize=(10, 6 * len(alpha_files_dict)))
    
    line_width = 2

    # Loop through the different alphas and plot each in its own subplot
    for idx, (alpha, file_paths) in enumerate(alpha_files_dict.items()):
        rewards_list = []
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            smoothed_reward = df['Reward'].rolling(window=smoothing_window).mean()
            rewards_list.append(smoothed_reward)
        
        avg_reward = pd.concat(rewards_list, axis=1).mean(axis=1)
        
        axes[idx].plot(avg_reward, label=f'Alpha = {alpha}', linewidth=line_width)
        axes[idx].set_title(f'Alpha = {alpha}')
        axes[idx].set_xlabel('Episodes')
        axes[idx].set_ylabel('Smoothed Reward')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves_subplots.pdf', format='pdf', dpi=300)
    plt.close()

# Plot with subplots
plot_learning_curves_subplots(alpha_files_dict, smoothing_window=1000)
