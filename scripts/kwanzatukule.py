# Import required modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
df['Month-Year'] = df['DATE'].dt.strftime('%B %Y')

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
df.to_csv(os.path.join(output_dir, 'cleaned_sales_data.csv'), index=False)

# View Summary Statistics
summary_statistics = df.describe()
print("\nSummary Statistics:")
print(summary_statistics)

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
business_aggregated = df.groupby('ANONYMIZED BUSINESS').agg({'QUANTITY': 'sum', 'VALUE': 'sum', 'DATE': 'count'}).reset_index()
business_aggregated.rename(columns={'DATE': 'FREQUENCY'}, inplace=True)

# Order by 'VALUE' descending
business_aggregated = business_aggregated.sort_values(by='VALUE', ascending=False)

# Display table for 'ANONYMIZED BUSINESS'
print("\nAggregated data by 'ANONYMIZED BUSINESS':")
print(business_aggregated)

# Aggregate 'QUANTITY' and 'VALUE' by 'Month-Year'
time_series_data = df.groupby('Month-Year').agg({'QUANTITY': 'sum', 'VALUE': 'sum'}).reset_index()

# Convert 'Month-Year' to datetime for plotting
time_series_data['Month-Year'] = pd.to_datetime(time_series_data['Month-Year'], format='%B %Y')

# Set the theme for the plots
sns.set_theme(style="darkgrid", palette="dark")

# Plot the time series data
plt.figure(figsize=(14, 7))
sns.lineplot(x='Month-Year', y='VALUE', data=time_series_data, marker='o', label='Value', color='darkgreen')
sns.lineplot(x='Month-Year', y='QUANTITY', data=time_series_data, marker='o', label='Quantity', color='darkblue')
plt.title('Time Series of Value and Quantity over Month-Year')
plt.xlabel('Month-Year')
plt.ylabel('Value / Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Determine the top 5 most frequently purchased products by quantity
top_5_products_by_quantity = df.groupby('ANONYMIZED PRODUCT').agg({'QUANTITY': 'sum'}).reset_index().sort_values(by='QUANTITY', ascending=False).head(5)
print("\nTop 5 most frequently purchased products by quantity:")
print(top_5_products_by_quantity)

# Determine the top 5 most valuable products by value
top_5_products_by_value = df.groupby('ANONYMIZED PRODUCT').agg({'VALUE': 'sum'}).reset_index().sort_values(by='VALUE', ascending=False).head(5)
print("\nTop 5 most valuable products by value:")
print(top_5_products_by_value)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
business_aggregated['CLUSTER'] = kmeans.fit_predict(business_aggregated[['QUANTITY', 'VALUE', 'FREQUENCY']])

# Display the clusters
print("\nBusiness clusters using K-Means:")
print(business_aggregated)

# Rule-based classification
def classify_business(row):
    if row['VALUE'] > business_aggregated['VALUE'].quantile(0.75):
        return 'High Value'
    elif row['VALUE'] > business_aggregated['VALUE'].quantile(0.25):
        return 'Medium Value'
    else:
        return 'Low Value'

business_aggregated['VALUE_CATEGORY'] = business_aggregated.apply(classify_business, axis=1)

# Display the rule-based classification
print("\nBusiness classification using rule-based method:")
print(business_aggregated[['ANONYMIZED BUSINESS', 'QUANTITY', 'VALUE', 'FREQUENCY', 'VALUE_CATEGORY']])

# ARIMA forecasting for VALUE
model_value = ARIMA(time_series_data['VALUE'], order=(5, 1, 0))
model_value_fit = model_value.fit()
print("\nARIMA model summary for VALUE:")
print(model_value_fit.summary())

# Forecast VALUE
forecast_value = model_value_fit.forecast(steps=12)
print("\nForecasted VALUE for the next 12 months:")
print(forecast_value)

# Calculate error metrics for ARIMA
mae_value_arima = mean_absolute_error(time_series_data['VALUE'][-12:], forecast_value)
rmse_value_arima = np.sqrt(mean_squared_error(time_series_data['VALUE'][-12:], forecast_value))
print(f"\nARIMA MAE for VALUE: {mae_value_arima}")
print(f"ARIMA RMSE for VALUE: {rmse_value_arima}")

# ETS forecasting for VALUE
model_ets = ExponentialSmoothing(time_series_data['VALUE'], seasonal='add', seasonal_periods=12, initialization_method='estimated')
model_ets_fit = model_ets.fit()
forecast_value_ets = model_ets_fit.forecast(steps=12)
print("\nETS model summary for VALUE:")
print(model_ets_fit.summary())

# Calculate error metrics for ETS
mae_value_ets = mean_absolute_error(time_series_data['VALUE'][-12:], forecast_value_ets)
rmse_value_ets = np.sqrt(mean_squared_error(time_series_data['VALUE'][-12:], forecast_value_ets))
print(f"\nETS MAE for VALUE: {mae_value_ets}")
print(f"ETS RMSE for VALUE: {rmse_value_ets}")

# Prophet forecasting for VALUE
df_prophet = time_series_data[['Month-Year', 'VALUE']].rename(columns={'Month-Year': 'ds', 'VALUE': 'y'})
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=12, freq='M')
forecast_value_prophet = model_prophet.predict(future)
forecast_value_prophet = forecast_value_prophet[['ds', 'yhat']].tail(12)['yhat']
print("\nProphet forecast for VALUE:")
print(forecast_value_prophet)

# Calculate error metrics for Prophet
mae_value_prophet = mean_absolute_error(time_series_data['VALUE'][-12:], forecast_value_prophet)
rmse_value_prophet = np.sqrt(mean_squared_error(time_series_data['VALUE'][-12:], forecast_value_prophet))
print(f"\nProphet MAE for VALUE: {mae_value_prophet}")
print(f"Prophet RMSE for VALUE: {rmse_value_prophet}")

# Compare models based on error metrics
print("\nModel comparison for VALUE:")
print(f"ARIMA MAE: {mae_value_arima}, RMSE: {rmse_value_arima}")
print(f"ETS MAE: {mae_value_ets}, RMSE: {rmse_value_ets}")
print(f"Prophet MAE: {mae_value_prophet}, RMSE: {rmse_value_prophet}")

# Identify anomalies using Z-score
time_series_data['Z_SCORE'] = (time_series_data['VALUE'] - time_series_data['VALUE'].mean()) / time_series_data['VALUE'].std()
anomalies_z_score = time_series_data[time_series_data['Z_SCORE'].abs() > 3]
print("\nAnomalies detected using Z-score method:")
print(anomalies_z_score[['Month-Year', 'VALUE', 'Z_SCORE']])

# Identify anomalies using IQR
Q1 = time_series_data['VALUE'].quantile(0.25)
Q3 = time_series_data['VALUE'].quantile(0.75)
IQR = Q3 - Q1
anomalies_iqr = time_series_data[(time_series_data['VALUE'] < (Q1 - 1.5 * IQR)) | (time_series_data['VALUE'] > (Q3 + 1.5 * IQR))]
print("\nAnomalies detected using IQR method:")
print(anomalies_iqr[['Month-Year', 'VALUE']])
