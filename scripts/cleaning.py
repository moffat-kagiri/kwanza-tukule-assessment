# Import required modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv('data/raw_data.csv')
df.head()
df.info()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Check for invalid data types
# Convert 'DATE' to datetime
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
# Convert 'QUANTITY' to numeric
df['QUANTITY'] = pd.to_numeric(df['QUANTITY'], errors='coerce')
# Convert 'UNIT PRICE' to numeric (remove commas first)
df['UNIT PRICE'] = df['UNIT PRICE'].str.replace(',', '').astype(float)

# Remove rows with missing 'UNIT PRICE'
df = df.dropna(subset=['UNIT PRICE'])

# Verify the removal
missing_values_after = df.isnull().sum()
print("\nMissing values after removal in each column:")
print(missing_values_after)

# Check for invalid data types (NaT for datetime, NaN for numeric)
invalid_dates = df['DATE'].isnull().sum()
invalid_quantities = df['QUANTITY'].isnull().sum()
invalid_unit_prices = df['UNIT PRICE'].isnull().sum()

print("\nInvalid data types:")
print(f"Invalid dates: {invalid_dates}")
print(f"Invalid quantities: {invalid_quantities}")
print(f"Invalid unit prices: {invalid_unit_prices}")

# Create "Month-Year" feature from 'DATE'
df['Month-Year'] = df['DATE'].dt.strftime('%Y-%m')

# Verify the new feature
print("\nSample of 'Month-Year' feature:")
print(df[['DATE', 'Month-Year']].head())

# Calculate 'VALUE' as 'QUANTITY' * 'UNIT PRICE'
df['VALUE'] = df['QUANTITY'] * df['UNIT PRICE']

# Ensure the directory exists
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the cleaned data to a new CSV file
df.to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)

# Aggregate 'QUANTITY' and 'VALUE' by 'ANONYMIZED CATEGORY' and then by 'ANONYMIZED BUSINESS'
aggregated_data = df.groupby(['ANONYMIZED CATEGORY', 'ANONYMIZED BUSINESS']).agg({'QUANTITY': 'sum', 'VALUE': 'sum'}).reset_index()

# Verify the aggregation
print("\nAggregated data by 'ANONYMIZED CATEGORY' and 'ANONYMIZED BUSINESS':")
print(aggregated_data)

# Aggregate 'QUANTITY' and 'VALUE' by 'ANONYMIZED CATEGORY'
category_aggregated = df.groupby('ANONYMIZED CATEGORY').agg({'QUANTITY': 'sum', 'VALUE': 'sum'}).reset_index()

# Order by 'VALUE' descending
category_aggregated = category_aggregated.sort_values(by='VALUE', ascending=False)

# Display table for 'ANONYMIZED CATEGORY'
print("\nAggregated data by 'ANONYMIZED CATEGORY':")
print(category_aggregated)

# Aggregate 'QUANTITY' and 'VALUE' by 'ANONYMIZED BUSINESS'
business_aggregated = df.groupby('ANONYMIZED BUSINESS').agg({'QUANTITY': 'sum', 'VALUE': 'sum'}).reset_index()

# Order by 'VALUE' descending
business_aggregated = business_aggregated.sort_values(by='VALUE', ascending=False)

# Display table for 'ANONYMIZED BUSINESS'
print("\nAggregated data by 'ANONYMIZED BUSINESS':")
print(business_aggregated)

# Aggregate 'QUANTITY' and 'VALUE' by 'Month-Year'
time_series_data = df.groupby('Month-Year').agg({'QUANTITY': 'sum', 'VALUE': 'sum'}).reset_index()

# Convert 'Month-Year' to datetime for plotting
time_series_data['Month-Year'] = pd.to_datetime(time_series_data['Month-Year'])

# Plot the time series data
plt.figure(figsize=(14, 7))
sns.lineplot(x='Month-Year', y='VALUE', data=time_series_data, marker='o', label='Value')
sns.lineplot(x='Month-Year', y='QUANTITY', data=time_series_data, marker='o', label='Quantity')
plt.title('Time Series of Value and Quantity over Month-Year')
plt.xlabel('Month-Year')
plt.ylabel('Value / Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
