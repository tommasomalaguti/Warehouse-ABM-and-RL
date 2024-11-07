import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import randint

# Load your data
print("Loading data...")
csv = "/Users/tommasomalaguti/Documents/Python/Tesi/model5_ml/forklift_training_sim_results_2_qalpha.csv"
data = pd.read_csv(csv)
print("Data loaded.")

# Define features (X) and target (y) for energy consumption
print("Defining features for energy consumption prediction...")
features_for_energy = ['Speed', 'Load Weight', 'Loading Time', 'Unloading Time', 'Distance to Target']
X_energy = data[features_for_energy]
y_energy = data['Energy Consumption (kWh)']
print("Features defined.")

# Split the data
print("Splitting data into training and testing sets for energy consumption...")
X_train, X_test, y_train, y_test = train_test_split(X_energy, y_energy, test_size=0.2, random_state=42)
print("Data split complete.")

# Train a regression model to predict energy consumption
print("Training the energy consumption prediction model...")
energy_model = LinearRegression()
energy_model.fit(X_train, y_train)
print("Energy consumption model trained.")

# Predict energy consumption
print("Predicting energy consumption on the test set...")
predicted_energy = energy_model.predict(X_test)
print("Energy consumption prediction complete.")

# Evaluate the model
print("Evaluating energy consumption model...")
mse_energy = mean_squared_error(y_test, predicted_energy)
print(f"Mean Squared Error for Energy Prediction: {mse_energy}")

# Add the predicted energy consumption to your test set
print("Adding predicted energy consumption to the test set...")
X_test['Predicted Energy Consumption (kWh)'] = predicted_energy
print("Predicted energy added.")

# Define the parameter distribution for Randomized Search
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Perform Randomized Search for the best parameters
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Define features and target for accident probability prediction
print("Defining features for accident probability prediction...")
features_for_accident = ['Speed', 'Load Weight', 'Distance to Target', 'Loading Time', 'Unloading Time']
X_accident = X_test[features_for_accident]
y_accident = data.loc[X_test.index, 'Accident Probability']
print("Features defined.")

# Fit the Randomized Search model
random_search.fit(X_accident, y_accident)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

# Evaluate the best model
best_model = random_search.best_estimator_
predicted_accident_prob = best_model.predict(X_accident)
mse_accident = mean_squared_error(y_accident, predicted_accident_prob)
print(f"Mean Squared Error for Accident Probability Prediction: {mse_accident}")

# Feature Importance Analysis
print("Analyzing feature importances...")
importances = best_model.feature_importances_
feature_names = features_for_accident

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances in Accident Probability Prediction')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()
