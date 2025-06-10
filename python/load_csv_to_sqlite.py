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

# Create table with appropriate columns based on dataset description
cursor.execute('''
CREATE TABLE IF NOT EXISTS supply_chain (
    Type TEXT,
    Days_for_shipping_real INTEGER,
    Days_for_shipment_scheduled INTEGER,
    Benefit_per_order REAL,
    Sales_per_customer REAL,
    Delivery_Status TEXT,
    Late_delivery_risk INTEGER,
    Category_Id INTEGER,
    Category_Name TEXT,
    Customer_City TEXT,
    Customer_Country TEXT,
    Customer_Email TEXT,
    Customer_Fname TEXT,
    Customer_Id INTEGER,
    Customer_Lname TEXT,
    Customer_Password TEXT,
    Customer_Segment TEXT,
    Customer_State TEXT,
    Customer_Street TEXT,
    Customer_Zipcode TEXT,
    Department_Id INTEGER,
    Department_Name TEXT,
    Latitude REAL,
    Longitude REAL,
    Market TEXT,
    Order_City TEXT,
    Order_Country TEXT,
    Order_Customer_Id INTEGER,
    order_date TEXT,
    Order_Id INTEGER,
    Order_Item_Cardprod_Id INTEGER,
    Order_Item_Discount REAL,
    Order_Item_Discount_Rate REAL,
    Order_Item_Id INTEGER,
    Order_Item_Product_Price REAL,
    Order_Item_Profit_Ratio REAL,
    Order_Item_Quantity INTEGER,
    Sales REAL,
    Order_Item_Total REAL,
    Order_Profit_Per_Order REAL,
    Order_Region TEXT,
    Order_State TEXT,
    Order_Status TEXT,
    Product_Card_Id INTEGER,
    Product_Category_Id INTEGER,
    Product_Description TEXT,
    Product_Image TEXT,
    Product_Name TEXT,
    Product_Price REAL,
    Product_Status INTEGER,
    Shipping_date TEXT,
    Shipping_Mode TEXT
)
''')

# Load data into the table
df.to_sql('supply_chain', conn, if_exists='replace', index=False)

# Commit and close
conn.commit()
conn.close()

print(f"Dataset loaded into SQLite database at {db_path}")
