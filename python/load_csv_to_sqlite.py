import pandas as pd
import sqlite3
import os

# Define paths
project_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(project_dir, 'data', 'raw', 'DataCoSupplyChainDataset.csv')
db_path = os.path.join(project_dir, 'data', 'supply_chain.db')

# Read CSV with specified encoding
df = pd.read_csv(csv_path, encoding='latin1')

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table with exact column names from the dataset description
cursor.execute('''
CREATE TABLE IF NOT EXISTS supply_chain (
    "Type" TEXT,
    "Days for shipping (real)" INTEGER,
    "Days for shipment (scheduled)" INTEGER,
    "Benefit per order" REAL,
    "Sales per customer" REAL,
    "Delivery Status" TEXT,
    "Late_delivery_risk" INTEGER,
    "Category Id" INTEGER,
    "Category Name" TEXT,
    "Customer City" TEXT,
    "Customer Country" TEXT,
    "Customer Email" TEXT,
    "Customer Fname" TEXT,
    "Customer Id" INTEGER,
    "Customer Lname" TEXT,
    "Customer Password" TEXT,
    "Customer Segment" TEXT,
    "Customer State" TEXT,
    "Customer Street" TEXT,
    "Customer Zipcode" TEXT,
    "Department Id" INTEGER,
    "Department Name" TEXT,
    "Latitude" REAL,
    "Longitude" REAL,
    "Market" TEXT,
    "Order City" TEXT,
    "Order Country" TEXT,
    "Order Customer Id" INTEGER,
    "order date (DateOrders)" TEXT,
    "Order Id" INTEGER,
    "Order Item Cardprod Id" INTEGER,
    "Order Item Discount" REAL,
    "Order Item Discount Rate" REAL,
    "Order Item Id" INTEGER,
    "Order Item Product Price" REAL,
    "Order Item Profit Ratio" REAL,
    "Order Item Quantity" INTEGER,
    "Sales" REAL,
    "Order Item Total" REAL,
    "Order Profit Per Order" REAL,
    "Order Region" TEXT,
    "Order State" TEXT,
    "Order Status" TEXT,
    "Product Card Id" INTEGER,
    "Product Category Id" INTEGER,
    "Product Description" TEXT,
    "Product Image" TEXT,
    "Product Name" TEXT,
    "Product Price" REAL,
    "Product Status" INTEGER,
    "Shipping date (DateOrders)" TEXT,
    "Shipping Mode" TEXT
)
''')

# Load data into the table
df.to_sql('supply_chain', conn, if_exists='replace', index=False)

# Commit and close
conn.commit()
conn.close()

print(f"Dataset loaded into SQLite database at {db_path}")