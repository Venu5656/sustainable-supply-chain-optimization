import pandas as pd
import sqlite3
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

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

# Load the world cities dataset
cities_df = pd.read_csv(cities_path)

# Standardize city and country names for matching
df['Destination_City'] = df['Destination_City'].str.lower().str.strip()
df['Destination_Country'] = df['Destination_Country'].str.lower().str.strip()
cities_df['city_ascii'] = cities_df['city_ascii'].str.lower().str.strip()
cities_df['country'] = cities_df['country'].str.lower().str.strip()

# Create a city-country key for merging
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

# Drop rows where coordinates are missing
df = df.dropna(subset=['Destination_Latitude', 'Destination_Longitude'])

# Calculate distances
df['Distance_km'] = df.apply(
    lambda row: haversine(
        row['Origin_Latitude'], row['Origin_Longitude'],
        row['Destination_Latitude'], row['Destination_Longitude']
    ), axis=1
)

# Calculate emissions
df['Carbon_Emissions_kg'] = df.apply(
    lambda row: row['Distance_km'] * emission_factors.get(row['Shipping_Mode'], 0.1), axis=1
)

# Drop temporary column
df = df.drop(columns=['City_Country'])

# Save processed data
df.to_csv(output_csv, index=False)

print(f"Processed data saved to {output_csv}")