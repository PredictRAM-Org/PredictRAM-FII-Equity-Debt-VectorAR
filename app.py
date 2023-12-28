# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

nifty_data = pd.read_excel("NIFTY.xlsx")

# Check if 'Date' column exists in nifty_data
if 'Date' not in nifty_data.columns:
    raise KeyError("Column 'Date' not found in NIFTY data. Available columns: {}".format(nifty_data.columns))

# Load FII data
fii_data = pd.read_excel("FII.xlsx", sheet_name="Equity")  # Assuming the Equity sheet is used

# Check if 'Date' column exists in fii_data
if 'Date' not in fii_data.columns:
    raise KeyError("Column 'Date' not found in FII data.")

# Merge NIFTY and FII data
merged_data = pd.merge(nifty_data, fii_data, on="Date", how="left")

# Drop missing values
merged_data = merged_data.dropna()

# Feature engineering
merged_data["Year"] = merged_data["Date"].dt.year
merged_data["Month"] = merged_data["Date"].dt.month

# Define features and target variable
features = ["Equity Net Investment"]  # Assuming only Equity Net Investment is used
target = "Percentage Change"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    merged_data[features], merged_data[target], test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_pred = linear_model.predict(X_test_scaled)

# Random Forest Regression model
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# Gradient Boosting Regression model
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)

# LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=(input_shape, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

lstm_model = create_lstm_model(X_train_scaled.shape[1])
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
lstm_pred = lstm_model.predict(X_test_lstm).flatten()

# ARIMA model
arima_model = ARIMA(merged_data[target], order=(5, 1, 0))
arima_result = arima_model.fit()
arima_pred = arima_result.predict(start=len(merged_data), end=len(merged_data) + len(X_test) - 1, typ='levels')

# VAR model
var_model = VAR(merged_data[["Percentage Change", "Equity Net Investment"]])
var_result = var_model.fit()
var_pred = var_result.forecast(merged_data[["Percentage Change", "Equity Net Investment"]].values, steps=len(X_test))

# Plotting
st.title("NIFTY Predictive Modeling")
st.subheader("Linear Regression Model")
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, linear_pred, label="Linear Regression Prediction")
plt.legend()
st.pyplot(plt)

st.subheader("Random Forest Regression Model")
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, rf_pred, label="Random Forest Prediction")
plt.legend()
st.pyplot(plt)

st.subheader("Gradient Boosting Regression Model")
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, gb_pred, label="Gradient Boosting Prediction")
plt.legend()
st.pyplot(plt)

st.subheader("LSTM Model")
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, lstm_pred, label="LSTM Prediction")
plt.legend()
st.pyplot(plt)

st.subheader("ARIMA Model")
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, arima_pred, label="ARIMA Prediction")
plt.legend()
st.pyplot(plt)

st.subheader("VAR Model")
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, var_pred[:, 0], label="VAR Prediction")
plt.legend()
st.pyplot(plt)
