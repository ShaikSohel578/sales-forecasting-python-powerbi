# ----------------------------------------------------------
# AI Sales Forecasting using NeuralProphet
# Author: Sohel
# ----------------------------------------------------------
# This project predicts future sales using the Sample Superstore dataset.
# It applies NeuralProphet (an advanced version of Facebook Prophet)
# to model seasonality and trend patterns in historical data.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ----------------------------------------------------------
# 1. Load and prepare dataset
# ----------------------------------------------------------
data = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

# Convert order date to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Aggregate sales by month
monthly_sales = data.groupby(pd.Grouper(key='Order Date', freq='M')).agg({'Sales': 'sum'}).reset_index()

# Rename columns for NeuralProphet
monthly_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)

print("📈 Data sample:")
print(monthly_sales.head())

# ----------------------------------------------------------
# 2. Train NeuralProphet model
# ----------------------------------------------------------
model = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    epochs=200
)

metrics = model.fit(monthly_sales, freq='M')

# ----------------------------------------------------------
# 3. Predict future sales
# ----------------------------------------------------------
future = model.make_future_dataframe(monthly_sales, periods=12, n_historic_predictions=True)
forecast = model.predict(future)

# ----------------------------------------------------------
# 4. Evaluate model performance
# ----------------------------------------------------------
y_true = monthly_sales['y']
y_pred = forecast['yhat1'][:len(y_true)]

mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))

print("\n📊 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# ----------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------
model.plot(forecast)
plt.title("📈 Monthly Sales Forecast using NeuralProphet")
plt.show()

model.plot_components(forecast)
plt.show()

# ----------------------------------------------------------
# 6. Save output
# ----------------------------------------------------------
forecast.to_csv("neuralprophet_forecast_output.csv", index=False)
print("\n✅ Forecast saved to 'neuralprophet_forecast_output.csv'")
