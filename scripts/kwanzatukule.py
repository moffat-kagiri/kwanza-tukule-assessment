import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
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
print(df[['Month-Year', 'DATE']].head())

# Calculate 'VALUE' as 'QUANTITY' * 'UNIT PRICE'
df['VALUE'] = df['QUANTITY'] * df['UNIT PRICE']

# Ensure the directory exists
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# Aggregate 'QUANTITY' and 'VALUE' by 'DATE'
time_series_data = df.groupby('DATE').agg({'QUANTITY': 'sum', 'VALUE': 'sum'}).reset_index()

# Convert 'DATE' to string and remove unwanted characters
time_series_data['DATE'] = time_series_data['DATE'].astype(str).str.strip()

# Try multiple date formats to handle inconsistencies
time_series_data['DATE'] = pd.to_datetime(time_series_data['DATE'], errors='coerce')

# Drop any rows where 'DATE' couldn't be converted
time_series_data = time_series_data.dropna(subset=['DATE'])

# Sort by date
time_series_data = time_series_data.sort_values(by='DATE')

# Ensure the data is sorted by date
time_series_data = time_series_data.sort_values(by='DATE')

# Set the theme for the plots
sns.set_theme(style="darkgrid", palette="dark")

# Plot the time series data
plt.figure(figsize=(14, 7))
sns.lineplot(x='DATE', y='VALUE', data=time_series_data, marker='o', label='Value', color='darkgreen')
sns.lineplot(x='DATE', y='QUANTITY', data=time_series_data, marker='o', label='Quantity', color='darkblue')
plt.title('Time Series of Value and Quantity over DATE')
plt.xlabel('DATE')
plt.ylabel('Value / Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_series_plot.png'))
plt.close()

# Determine the top 5 most frequently purchased products by quantity
top_5_products_by_quantity = df.groupby('ANONYMIZED PRODUCT').agg({'QUANTITY': 'sum'}).reset_index().sort_values(by='QUANTITY', ascending=False).head(5)
print("\nTop 5 most frequently purchased products by quantity:")
print(top_5_products_by_quantity)

# Determine the top 5 most valuable products by value
top_5_products_by_value = df.groupby('ANONYMIZED PRODUCT').agg({'VALUE': 'sum'}).reset_index().sort_values(by='VALUE', ascending=False).head(5)
print("\nTop 5 most valuable products by value:")
print(top_5_products_by_value)

# Rule-based classification
def classify_business(row):
    if row['VALUE'] > business_aggregated['VALUE'].quantile(0.75):
        return 'High Value'
    elif row['VALUE'] > business_aggregated['VALUE'].quantile(0.25):
        return 'Medium Value'
    else:
        return 'Low Value'

business_aggregated['VALUE_CATEGORY'] = business_aggregated.apply(classify_business, axis=1)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
business_aggregated['CLUSTER'] = kmeans.fit_predict(business_aggregated[['QUANTITY', 'VALUE', 'FREQUENCY']])

# Display the clusters
print("\nBusiness clusters using K-Means:")
print(business_aggregated)

# Display the rule-based classification
print("\nBusiness classification using rule-based method:")
print(business_aggregated[['ANONYMIZED BUSINESS', 'QUANTITY', 'VALUE', 'FREQUENCY', 'VALUE_CATEGORY']])

# ARIMA forecasting for VALUE
model_value = ARIMA(time_series_data['VALUE'], order=(5, 1, 0))
model_value_fit = model_value.fit()
print("\nARIMA model summary for VALUE:")
print(model_value_fit.summary())

# Forecast VALUE for the next 3 months
forecast_value = model_value_fit.forecast(steps=3)
print("\nForecasted VALUE for the next 3 months using ARIMA:")
print(forecast_value)

# Calculate error metrics for ARIMA
mae_value_arima = mean_absolute_error(time_series_data['VALUE'][-3:], forecast_value)
rmse_value_arima = np.sqrt(mean_squared_error(time_series_data['VALUE'][-3:], forecast_value))
print(f"\nARIMA MAE for VALUE: {mae_value_arima}")
print(f"ARIMA RMSE for VALUE: {rmse_value_arima}")

# ETS forecasting for VALUE
# Adjust the seasonal_periods to match the length of your data if necessary
model_ets = ExponentialSmoothing(time_series_data['VALUE'], seasonal='add', seasonal_periods=3, initialization_method='estimated')
model_ets_fit = model_ets.fit()
forecast_value_ets = model_ets_fit.forecast(steps=3)
print("\nETS model summary for VALUE:")
print(model_ets_fit.summary())

# Calculate error metrics for ETS
mae_value_ets = mean_absolute_error(time_series_data['VALUE'][-3:], forecast_value_ets)
rmse_value_ets = np.sqrt(mean_squared_error(time_series_data['VALUE'][-3:], forecast_value_ets))
print(f"\nETS MAE for VALUE: {mae_value_ets}")
print(f"ETS RMSE for VALUE: {rmse_value_ets}")

# Prophet forecasting for VALUE
df_prophet = time_series_data[['DATE', 'VALUE']].rename(columns={'DATE': 'ds', 'VALUE': 'y'})
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=3, freq='MS')
forecast_value_prophet = model_prophet.predict(future)
forecast_value_prophet = forecast_value_prophet[['ds', 'yhat']].tail(3)['yhat']
print("\nProphet forecast for VALUE:")
print(forecast_value_prophet)

# Calculate error metrics for Prophet
mae_value_prophet = mean_absolute_error(time_series_data['VALUE'][-3:], forecast_value_prophet)
rmse_value_prophet = np.sqrt(mean_squared_error(time_series_data['VALUE'][-3:], forecast_value_prophet))
print(f"\nProphet MAE for VALUE: {mae_value_prophet}")
print(f"Prophet RMSE for VALUE: {rmse_value_prophet}")

# Compare models based on error metrics
print("\nModel comparison for VALUE:")
print(f"ARIMA MAE: {mae_value_arima}, RMSE: {rmse_value_arima}")
print(f"ETS MAE: {mae_value_ets}, RMSE: {rmse_value_ets}")
print(f"Prophet MAE: {mae_value_prophet}, RMSE: {rmse_value_prophet}")

# Create a DataFrame for the forecasts
forecast_df = pd.DataFrame({
    'DATE': pd.date_range(start=time_series_data['DATE'].iloc[-1], periods=4, freq='MS')[1:],
    'ARIMA Forecast': forecast_value,
    'ETS Forecast': forecast_value_ets,
    'Prophet Forecast': forecast_value_prophet
})

# Plot the forecasts
plt.figure(figsize=(14, 7))
plt.plot(time_series_data['DATE'], time_series_data['VALUE'], label='Actual', marker='o')
plt.plot(forecast_df['DATE'], forecast_df['ARIMA Forecast'], label='ARIMA Forecast', marker='o')
plt.plot(forecast_df['DATE'], forecast_df['ETS Forecast'], label='ETS Forecast', marker='o')
plt.plot(forecast_df['DATE'], forecast_df['Prophet Forecast'], label='Prophet Forecast', marker='o')
plt.title('3-Month Forecasts of Value using ARIMA, ETS, and Prophet')
plt.xlabel('DATE')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'forecast_plot.png'))
plt.close()

# Save all outputs to an Excel file
output_excel_path = os.path.join(output_dir, 'cleaned_sales_data.xlsx')
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Save error metrics and model comparison to Excel
    error_metrics_df = pd.DataFrame({
        'Model': ['ARIMA', 'ETS', 'Prophet'],
        'MAE': [mae_value_arima, mae_value_ets, mae_value_prophet],
        'RMSE': [rmse_value_arima, rmse_value_ets, rmse_value_prophet]
    })
    error_metrics_df.to_excel(writer, sheet_name='Error Metrics', index=False)
    
    df.to_excel(writer, sheet_name='Cleaned Data', index=False)
    aggregated_data.to_excel(writer, sheet_name='Aggregated Data', index=False)
    category_aggregated.to_excel(writer, sheet_name='Category Aggregated', index=False)
    business_aggregated.to_excel(writer, sheet_name='Business Aggregated', index=False)
    time_series_data.to_excel(writer, sheet_name='Time Series Data', index=False)
    top_5_products_by_quantity.to_excel(writer, sheet_name='Top 5 Products by Quantity', index=False)
    top_5_products_by_value.to_excel(writer, sheet_name='Top 5 Products by Value', index=False)
    forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)
    
    # Save the plots as images and insert them into the Excel file
    workbook = writer.book
    worksheet = workbook.create_sheet('Time Series Plot')
    worksheet.add_image(openpyxl.drawing.image.Image(os.path.join(output_dir, 'time_series_plot.png')), 'A1')
    worksheet = workbook.create_sheet('Forecast Plot')
    worksheet.add_image(openpyxl.drawing.image.Image(os.path.join(output_dir, 'forecast_plot.png')), 'A1')

print(f"\nAll outputs have been saved to {output_excel_path}")
