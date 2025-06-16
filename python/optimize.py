import pandas as pd
import numpy as np
import os
import joblib

# Define paths
project_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_dir, 'data', 'processed', 'processed_data.csv')
model_path = os.path.join(project_dir, 'models', 'emission_model_rf_tuned.pkl')
output_path = os.path.join(project_dir, 'data', 'optimized', 'optimized_shipments.csv')

# Create directories if they don't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the processed dataset
df = pd.read_csv(data_path)

# Handle rare categories in categorical features (same as in model.py)
min_freq = 0.01 * len(df)  # 1% of the dataset size
for col in ['Origin_Country', 'Destination_Country']:
    value_counts = df[col].value_counts()
    rare_categories = value_counts[value_counts < min_freq].index
    df[col] = df[col].replace(rare_categories, 'Other')

# Load the trained model
model = joblib.load(model_path)

# Define possible shipping modes
shipping_modes = ['Standard Class', 'First Class', 'Second Class', 'Same Day']

# Compute average delivery days for each shipping mode from the dataset
delivery_days = df.groupby('Shipping_Mode')['Shipping_Days'].mean().to_dict()
print("Average delivery days per shipping mode:", delivery_days)

# Cost adjustment factors for each shipping mode (relative to Standard Class)
# These are assumptions and can be adjusted
cost_factors = {
    'Standard Class': 1.0,  # Baseline
    'First Class': 1.5,     # More expensive
    'Second Class': 0.8,    # Cheaper
    'Same Day': 2.0         # Most expensive
}

# Function to predict emissions for a given row and shipping mode
def predict_emissions(row, shipping_mode, model):
    row_copy = row.copy()
    row_copy['Shipping_Mode'] = shipping_mode
    features = ['Distance_km', 'Shipping_Mode', 'Cost', 'Quantity', 'Origin_Country', 'Destination_Country']
    X = pd.DataFrame([row_copy[features]])
    prediction = model.predict(X)[0]
    return max(prediction, 0)  # Ensure non-negative predictions

# Optimize shipping mode for each shipment
recommendations = []
total_emissions_before = 0
total_emissions_after = 0

for idx, row in df.iterrows():
    current_mode = row['Shipping_Mode']
    current_emissions = row['Carbon_Emissions_kg']
    current_cost = row['Cost']
    current_days = row['Shipping_Days']
    
    # Skip if current delivery days are missing
    if pd.isna(current_days):
        recommendations.append({
            'Order_Id': row['Order_Id'],
            'Original_Shipping_Mode': current_mode,
            'Recommended_Shipping_Mode': current_mode,
            'Original_Emissions_kg': current_emissions,
            'Recommended_Emissions_kg': current_emissions,
            'Emissions_Reduction_kg': 0.0,
            'Cost_Increase_Percent': 0.0,
            'Delivery_Days_Original': current_days,
            'Delivery_Days_Recommended': current_days
        })
        total_emissions_before += current_emissions
        total_emissions_after += current_emissions
        continue
    
    # Predict emissions for all shipping modes
    emissions = {}
    for mode in shipping_modes:
        emissions[mode] = predict_emissions(row, mode, model)
    
    # Find the best shipping mode that minimizes emissions while satisfying constraints
    best_mode = current_mode
    best_emissions = current_emissions
    cost_increase = 0.0
    new_days = current_days
    
    for mode in shipping_modes:
        if mode == current_mode:
            continue
        predicted_emissions = emissions[mode]
        # Apply constraints
        # 1. Cost increase should not exceed 20%
        new_cost = current_cost * cost_factors[mode] / cost_factors[current_mode]
        cost_increase_percent = (new_cost - current_cost) / current_cost * 100
        if cost_increase_percent > 20:
            continue
        # 2. Delivery time should not exceed current delivery time
        if delivery_days.get(mode, float('inf')) > current_days:
            continue
        # Update best mode if emissions are lower
        if predicted_emissions < best_emissions:
            best_mode = mode
            best_emissions = predicted_emissions
            cost_increase = cost_increase_percent
            new_days = delivery_days.get(mode, current_days)
    
    # Calculate emissions reduction
    emissions_reduction = current_emissions - best_emissions
    
    # Store the recommendation
    recommendations.append({
        'Order_Id': row['Order_Id'],
        'Original_Shipping_Mode': current_mode,
        'Recommended_Shipping_Mode': best_mode,
        'Original_Emissions_kg': current_emissions,
        'Recommended_Emissions_kg': best_emissions,
        'Emissions_Reduction_kg': emissions_reduction,
        'Cost_Increase_Percent': cost_increase,
        'Delivery_Days_Original': current_days,
        'Delivery_Days_Recommended': new_days
    })
    
    total_emissions_before += current_emissions
    total_emissions_after += best_emissions

# Create a DataFrame with recommendations
recommendations_df = pd.DataFrame(recommendations)

# Save the recommendations
recommendations_df.to_csv(output_path, index=False)
print(f"Optimization recommendations saved to {output_path}")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total Emissions Before Optimization: {total_emissions_before:.2f} kg CO2e")
print(f"Total Emissions After Optimization: {total_emissions_after:.2f} kg CO2e")
print(f"Total Emissions Reduction: {(total_emissions_before - total_emissions_after):.2f} kg CO2e")
print(f"Percentage Reduction: {((total_emissions_before - total_emissions_after) / total_emissions_before * 100):.2f}%")

# Print the number of shipments where the mode was changed
changed_shipments = len(recommendations_df[recommendations_df['Original_Shipping_Mode'] != recommendations_df['Recommended_Shipping_Mode']])
print(f"Number of shipments with changed shipping mode: {changed_shipments}")
print(f"Percentage of shipments changed: {(changed_shipments / len(df) * 100):.2f}%")