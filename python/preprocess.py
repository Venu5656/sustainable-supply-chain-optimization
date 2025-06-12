import pandas as pd
import sqlite3
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
import swifter

# Define paths
project_dir = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(project_dir, 'data', 'supply_chain.db')
cities_path = os.path.join(project_dir, 'data', 'external', 'worldcities.csv')
output_csv = os.path.join(project_dir, 'data', 'processed', 'processed_data.csv')

# Emission factors (kg CO2e per km)
emission_factors = {
    'Standard Class': 0.1,  # Truck
    'First Class': 0.5,     # Air
    'Second Class': 0.03,   # Rail
    'Same Day': 0.5         # Air
}

# Haversine formula to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Extract data with correct column names
query = '''
SELECT
    "Order Id" AS Order_Id,
    "Customer City" AS Origin_City,
    "Customer Country" AS Origin_Country,
    "Latitude" AS Origin_Latitude,
    "Longitude" AS Origin_Longitude,
    "Order City" AS Destination_City,
    "Order Country" AS Destination_Country,
    "Shipping Mode" AS Shipping_Mode,
    "Days for shipping (real)" AS Shipping_Days,
    "Order Item Total" AS Cost,
    "Order Item Quantity" AS Quantity,
    "Product Name" AS Product_Name,
    "order date (DateOrders)" AS order_date,
    "Shipping date (DateOrders)" AS Shipping_date
FROM supply_chain
WHERE "Delivery Status" != 'Shipping canceled'
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Print initial row count
print(f"Initial number of rows: {len(df)}")

# Normalize city and country names
df['Destination_City'] = df['Destination_City'].str.lower().str.strip()
df['Destination_Country'] = df['Destination_Country'].str.lower().str.strip()
df['Origin_Country'] = df['Origin_Country'].str.lower().str.strip()
cities_df = pd.read_csv(cities_path)
cities_df['city_ascii'] = cities_df['city_ascii'].str.lower().str.strip()
cities_df['country'] = cities_df['country'].str.lower().str.strip()

# Map country name variations (e.g., "ee. uu." to "usa", "méxico" to "mexico")
country_mapping = {
    'ee. uu.': 'usa',
    'méxico': 'mexico',
    'alemania': 'germany',
    'república dominicana': 'dominican republic',
    'perú': 'peru',
    'brasil': 'brazil',
    'puerto rico': 'puerto rico',
    'el salvador': 'el salvador'
}
df['Destination_Country'] = df['Destination_Country'].replace(country_mapping)
df['Origin_Country'] = df['Origin_Country'].replace(country_mapping)
cities_df['country'] = cities_df['country'].replace(country_mapping)

# Create a city-country key for merging
print("Merging city-country data...")
df['City_Country'] = df['Destination_City'] + ', ' + df['Destination_Country']
cities_df['City_Country'] = cities_df['city_ascii'] + ', ' + cities_df['country']

# Merge to get destination coordinates
df = df.merge(
    cities_df[['City_Country', 'lat', 'lng']],
    on='City_Country',
    how='left'
)

# Rename columns
df = df.rename(columns={'lat': 'Destination_Latitude', 'lng': 'Destination_Longitude'})

# Log the number of missing coordinates
missing_coords = df['Destination_Latitude'].isna().sum()
print(f"Number of rows with missing coordinates: {missing_coords}")
print(f"Percentage of rows with missing coordinates: {missing_coords / len(df) * 100:.2f}%")

# Drop rows with missing coordinates
df = df.dropna(subset=['Destination_Latitude', 'Destination_Longitude'])

# Calculate distances in parallel
print("Calculating distances...")
df['Distance_km'] = df.swifter.apply(
    lambda row: haversine(
        row['Origin_Latitude'], row['Origin_Longitude'],
        row['Destination_Latitude'], row['Destination_Longitude']
    ), axis=1
)

# Handle outliers in Distance_km (cap at 20,000 km, roughly the max possible distance on Earth)
distance_cap = 20000
outliers_distance = df[df['Distance_km'] > distance_cap]
print(f"Number of distance outliers (> {distance_cap} km): {len(outliers_distance)}")
df.loc[df['Distance_km'] > distance_cap, 'Distance_km'] = distance_cap

# Calculate emissions in parallel with added noise
print("Calculating carbon emissions with noise...")
# Base emissions
df['Carbon_Emissions_kg'] = df.swifter.apply(
    lambda row: row['Distance_km'] * emission_factors.get(row['Shipping_Mode'], 0.1), axis=1
)
# Add random noise (±10% of the base emissions)
np.random.seed(42)  # For reproducibility
noise = np.random.uniform(low=-0.3, high=0.3, size=len(df))
df['Carbon_Emissions_kg'] = df['Carbon_Emissions_kg'] * (1 + noise)

# Handle outliers in Carbon_Emissions_kg (cap at 99th percentile)
emissions_cap = df['Carbon_Emissions_kg'].quantile(0.99)
outliers_emissions = df[df['Carbon_Emissions_kg'] > emissions_cap]
print(f"Number of emissions outliers (> {emissions_cap:.2f} kg CO2e): {len(outliers_emissions)}")
df.loc[df['Carbon_Emissions_kg'] > emissions_cap, 'Carbon_Emissions_kg'] = emissions_cap

# Drop temporary column
df = df.drop(columns=['City_Country'])

# Save processed data
df.to_csv(output_csv, index=False)

print(f"Processed data saved to {output_csv}")
print(f"Total rows after processing: {len(df)}")