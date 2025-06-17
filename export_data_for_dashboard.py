import pandas as pd
import numpy as np
import joblib
import os

# Define paths
project_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_dir, 'data', 'processed', 'processed_data.csv')
model_path = os.path.join(project_dir, 'models', 'emission_model_rf_tuned.pkl')
output_dir = os.path.join(project_dir, 'data', 'dashboard')
os.makedirs(output_dir, exist_ok=True)

# Load the processed dataset
df = pd.read_csv(data_path)

# Handle rare categories in categorical features
min_freq = 0.01 * len(df)
for col in ['Origin_Country', 'Destination_Country']:
    value_counts = df[col].value_counts()
    rare_categories = value_counts[value_counts < min_freq].index
    df[col] = df[col].replace(rare_categories, 'Other')

# Compute average delivery days and cost factors
delivery_days = df.groupby('Shipping_Mode')['Shipping_Days'].mean().to_dict()
average_costs = df.groupby('Shipping_Mode')['Cost'].mean().to_dict()
base_cost = average_costs['Standard Class']
cost_factors = {mode: cost / base_cost for mode, cost in average_costs.items()}

# Load the trained model
model = joblib.load(model_path)

# Define possible shipping modes
shipping_modes = ['Standard Class', 'First Class', 'Second Class', 'Same Day']

# Function to predict emissions for a given shipment and shipping mode
def predict_emissions(shipment, shipping_mode, model):
    shipment_copy = shipment.copy()
    shipment_copy['Shipping_Mode'] = shipping_mode
    features = ['Distance_km', 'Shipping_Mode', 'Cost', 'Quantity', 'Origin_Country', 'Destination_Country']
    # Extract feature values into a dictionary
    feature_dict = {feature: shipment_copy[feature] for feature in features}
    # Create a DataFrame with a single row
    X = pd.DataFrame([feature_dict])
    prediction = model.predict(X)[0]
    return max(prediction, 0)

# Step 1: Export the processed dataset with predictions
df['Predicted_Emissions_kg'] = df.apply(
    lambda row: predict_emissions(row, row['Shipping_Mode'], model), axis=1
)
processed_data_path = os.path.join(output_dir, 'processed_data_with_predictions.csv')
df.to_csv(processed_data_path, index=False)
print(f"Exported processed dataset with predictions to {processed_data_path}")

# Step 2: Run optimization on the entire dataset and export results
optimization_results = []
for idx, row in df.iterrows():
    shipment_dict = row.to_dict()

    # Handle rare or unseen categories
    for col in ['Origin_Country', 'Destination_Country']:
        unique_categories = df[col].unique()
        if shipment_dict[col] not in unique_categories:
            shipment_dict[col] = 'Other'

    current_mode = shipment_dict['Shipping_Mode']
    current_cost = shipment_dict['Cost']
    current_days = shipment_dict['Shipping_Days']
    current_emissions = predict_emissions(shipment_dict, current_mode, model)

    # Skip if delivery days are missing
    if pd.isna(current_days):
        optimization_results.append({
            'original_shipping_mode': current_mode,
            'recommended_shipping_mode': current_mode,
            'original_emissions_kg': None,
            'recommended_emissions_kg': None,
            'emissions_reduction_kg': 0.0,
            'cost_increase_percent': 0.0,
            'delivery_days_original': current_days,
            'delivery_days_recommended': current_days
        })
        continue

    # Predict emissions for all shipping modes
    emissions = {}
    for mode in shipping_modes:
        emissions[mode] = predict_emissions(shipment_dict, mode, model)

    # Find the best shipping mode
    best_mode = current_mode
    best_emissions = current_emissions
    cost_increase = 0.0
    new_days = current_days

    for mode in shipping_modes:
        if mode == current_mode:
            continue
        predicted_emissions = emissions[mode]
        # Apply constraints
        new_cost = current_cost * cost_factors.get(mode, 1.0) / cost_factors.get(current_mode, 1.0)
        cost_increase_percent = (new_cost - current_cost) / current_cost * 100
        if cost_increase_percent > 30:
            continue
        max_allowed_days = current_days * 1.1
        if delivery_days.get(mode, float('inf')) > max_allowed_days:
            continue
        if predicted_emissions < best_emissions:
            best_mode = mode
            best_emissions = predicted_emissions
            cost_increase = cost_increase_percent
            new_days = delivery_days.get(mode, current_days)

    emissions_reduction = current_emissions - best_emissions

    optimization_results.append({
        'original_shipping_mode': current_mode,
        'recommended_shipping_mode': best_mode,
        'original_emissions_kg': current_emissions,
        'recommended_emissions_kg': best_emissions,
        'emissions_reduction_kg': emissions_reduction,
        'cost_increase_percent': cost_increase,
        'delivery_days_original': current_days,
        'delivery_days_recommended': new_days
    })

# Convert optimization results to DataFrame
optimization_df = pd.DataFrame(optimization_results)
# Combine with original dataset for context
optimization_df = pd.concat([df.reset_index(drop=True), optimization_df], axis=1)
optimization_path = os.path.join(output_dir, 'optimization_results.csv')
optimization_df.to_csv(optimization_path, index=False)
print(f"Exported optimization results to {optimization_path}")

# Step 3: Calculate and export aggregated metrics
total_emissions_reduced = optimization_df['emissions_reduction_kg'].sum()
total_cost_change = (optimization_df['cost_increase_percent'] * optimization_df['Cost'] / 100).sum()
avg_delivery_days_change = (optimization_df['delivery_days_recommended'] - optimization_df['delivery_days_original']).mean()
co2_reduction_goal_met = (optimization_df['emissions_reduction_kg'] / optimization_df['original_emissions_kg'] * 100 >= 20).mean() * 100

metrics = {
    'total_emissions_reduced_kg': total_emissions_reduced,
    'total_cost_change_usd': total_cost_change,
    'avg_delivery_days_change': avg_delivery_days_change,
    'percent_shipments_meeting_co2_goal': co2_reduction_goal_met
}
metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join(output_dir, 'business_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Exported business metrics to {metrics_path}")

if __name__ == "__main__":
    print("Data export completed successfully.")