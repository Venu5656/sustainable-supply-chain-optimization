import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Define paths
project_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_dir, 'data', 'processed', 'processed_data.csv')
model_path = os.path.join(project_dir, 'models', 'emission_model_rf_tuned.pkl')
plot_path = os.path.join(project_dir, 'plots', 'feature_importance.png')

# Create directories if they don't exist
os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(project_dir, 'plots'), exist_ok=True)

# Load the processed dataset
df = pd.read_csv(data_path)

# Handle rare categories in categorical features
# Group countries that appear less than 1% of the time into 'Other'
min_freq = 0.01 * len(df)  # 1% of the dataset size
for col in ['Origin_Country', 'Destination_Country']:
    value_counts = df[col].value_counts()
    rare_categories = value_counts[value_counts < min_freq].index
    df[col] = df[col].replace(rare_categories, 'Other')

# Verify the number of unique categories after grouping
print(f"Unique Origin_Country categories: {len(df['Origin_Country'].unique())}")
print(f"Categories: {df['Origin_Country'].unique()}")
print(f"Unique Destination_Country categories: {len(df['Destination_Country'].unique())}")
print(f"Categories: {df['Destination_Country'].unique()}")
print(f"Unique Shipping_Mode categories: {len(df['Shipping_Mode'].unique())}")
print(f"Categories: {df['Shipping_Mode'].unique()}")

# Select features and target
features = ['Distance_km', 'Shipping_Mode', 'Cost', 'Quantity', 'Origin_Country', 'Destination_Country']
target = 'Carbon_Emissions_kg'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the size of the train/test sets
print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Define preprocessing for numerical and categorical features
numerical_features = ['Distance_km', 'Cost', 'Quantity']
categorical_features = ['Shipping_Mode', 'Origin_Country', 'Destination_Country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# Create pipelines for both models
# Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# XGBoost for comparison
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

# Define a smaller hyperparameter grid for Random Forest
param_grid_rf = {
    'regressor__n_estimators': [100, 150],
    'regressor__max_depth': [5, 10],  # Further limit max_depth
    'regressor__min_samples_split': [10],  # Increase to reduce overfitting
    'regressor__min_samples_leaf': [5]  # Add min_samples_leaf to reduce overfitting
}

# Define a hyperparameter grid for XGBoost
param_grid_xgb = {
    'regressor__n_estimators': [100, 150],
    'regressor__max_depth': [3, 5],
    'regressor__learning_rate': [0.1, 0.3]
}

# Perform GridSearchCV for Random Forest
print("Performing hyperparameter tuning with GridSearchCV for Random Forest...")
grid_search_rf = GridSearchCV(
    rf_pipeline,
    param_grid_rf,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=1
)
grid_search_rf.fit(X_train, y_train)

# Perform GridSearchCV for XGBoost
print("Performing hyperparameter tuning with GridSearchCV for XGBoost...")
grid_search_xgb = GridSearchCV(
    xgb_pipeline,
    param_grid_xgb,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=1
)
grid_search_xgb.fit(X_train, y_train)

# Get the best models
best_rf_model = grid_search_rf.best_estimator_
best_xgb_model = grid_search_xgb.best_estimator_
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)

# Make predictions on the test set
y_pred_rf = best_rf_model.predict(X_test)
y_pred_xgb = best_xgb_model.predict(X_test)

# Ensure predictions are non-negative
y_pred_rf = np.maximum(y_pred_rf, 0)
y_pred_xgb = np.maximum(y_pred_xgb, 0)

# Evaluate both models
print("\nRandom Forest Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R-squared (R2): {r2_score(y_test, y_pred_rf):.2f}")

print("\nXGBoost Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_xgb):.2f}")
print(f"R-squared (R2): {r2_score(y_test, y_pred_xgb):.2f}")

# Save the Random Forest model
joblib.dump(best_rf_model, model_path)
print(f"Model saved to {model_path}")

# Feature Importance Analysis for Random Forest
feature_names = (
    numerical_features +
    list(best_rf_model.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(categorical_features))
)
importances = best_rf_model.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances for Predicting Carbon Emissions (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Feature importance plot saved to {plot_path}")

# Example: Predict emissions for a new shipment and suggest optimization
example = pd.DataFrame({
    'Distance_km': [5000],
    'Shipping_Mode': ['First Class'],
    'Cost': [200],
    'Quantity': [2],
    'Origin_Country': ['usa'],
    'Destination_Country': ['china']
})

# Ensure example data aligns with training data categories
for col in ['Origin_Country', 'Destination_Country']:
    unique_categories = df[col].unique()
    if example[col].iloc[0] not in unique_categories:
        print(f"Replacing {col}={example[col].iloc[0]} with 'Other' as it is not in training data: {unique_categories}")
        example[col] = 'Other'

# Predict with both models
predicted_emissions_rf = best_rf_model.predict(example)
predicted_emissions_xgb = best_xgb_model.predict(example)

# Ensure predictions are non-negative
predicted_emissions_rf = np.maximum(predicted_emissions_rf, 0)
predicted_emissions_xgb = np.maximum(predicted_emissions_xgb, 0)

print("\nExample Prediction with Random Forest:")
print(f"Predicted Carbon Emissions for a 5000 km First Class shipment (USA to China): {predicted_emissions_rf[0]:.2f} kg CO2e")

print("\nExample Prediction with XGBoost:")
print(f"Predicted Carbon Emissions for a 5000 km First Class shipment (USA to China): {predicted_emissions_xgb[0]:.2f} kg CO2e")

# Suggest optimization: What if we use Second Class instead?
example_alt = example.copy()
example_alt['Shipping_Mode'] = ['Second Class']
predicted_emissions_rf_alt = best_rf_model.predict(example_alt)
predicted_emissions_xgb_alt = best_xgb_model.predict(example_alt)

# Ensure predictions are non-negative
predicted_emissions_rf_alt = np.maximum(predicted_emissions_rf_alt, 0)
predicted_emissions_xgb_alt = np.maximum(predicted_emissions_xgb_alt, 0)

print(f"\nRandom Forest: Predicted Carbon Emissions if switched to Second Class: {predicted_emissions_rf_alt[0]:.2f} kg CO2e")
print(f"Random Forest: Emissions reduction: {(predicted_emissions_rf[0] - predicted_emissions_rf_alt[0]):.2f} kg CO2e")

print(f"\nXGBoost: Predicted Carbon Emissions if switched to Second Class: {predicted_emissions_xgb_alt[0]:.2f} kg CO2e")
print(f"XGBoost: Emissions reduction: {(predicted_emissions_xgb[0] - predicted_emissions_xgb_alt[0]):.2f} kg CO2e")