# Sustainable Supply Chain Optimization

# Sustainable Supply Chain Optimization

This project optimizes supply chain logistics with a focus on sustainability, specifically reducing carbon emissions while minimizing costs. It uses SQL and pandas for data preprocessing, scikit-learn for predictive modeling, and Tableau for interactive visualizations, deployed on AWS S3.

## Project Overview
- **Objective**: Optimize supply chain routes to balance cost and carbon emissions.
- **Dataset**: DataCo SMART Supply Chain Dataset (to be augmented with emissions data).
- **Tools**:
  - SQL (PostgreSQL/SQLite) for data extraction and transformation.
  - Python (pandas, scikit-learn, numpy, matplotlib, seaborn) for preprocessing and modeling.
  - Tableau for visualizations.
  - AWS S3 for dashboard hosting.
- **Deliverables**:
  - Cleaned dataset and preprocessing scripts.
  - Predictive model for route optimization.
  - Interactive Tableau dashboard.
  - Comprehensive documentation.

## Folder Structure

sustainable-supply-chain-optimization/
├── data/
│   ├── raw/                   # Raw dataset (e.g., DataCo dataset)
│   ├── processed/             # Cleaned and transformed data
├── sql/
│   ├── extract.sql            # SQL queries for data extraction
│   ├── transform.sql          # SQL queries for data transformation
├── python/
│   ├── preprocess.py          # Data cleaning and feature engineering
│   ├── model.py               # Predictive model training and evaluation
│   ├── eda.ipynb              # Jupyter Notebook for exploratory data analysis
├── tableau/
│   ├── dashboard.twbx        # Tableau workbook file
├── docs/
│   ├── report.md              # Project report (methodology, results, impact)
│   ├── aws_deployment.md      # Instructions for AWS S3 deployment
├── README.md                  # This file
├── LICENSE                    # MIT License file
├── requirements.txt           # Python dependencies


## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sustainable-supply-chain-optimization.git
   cd sustainable-supply-chain-optimization

2. **Download dependencies**:
pip install -r requirements.txt

3. **Download the dataset**:
Place the DataCo SMART Supply Chain Dataset in data/raw/.

4. Set up a database (e.g., SQLite or PostgreSQL) for SQL queries.

5. Run scripts:
Preprocessing: python python/preprocess.py
Modeling: python python/model.py
Tableau and AWS:

Open tableau/dashboard.twbx in Tableau.
Follow docs/aws_deployment.md to deploy the dashboard on AWS S3.

Author - Sikhakolli Venu