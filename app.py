import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load NIFTY data
nifty_data = pd.read_excel("NIFTY.xlsx")

# Load FII data
equity_data = pd.read_excel("FII.xlsx", sheet_name="Equity")
debt_data = pd.read_excel("FII.xlsx", sheet_name="Debt")

# Merge NIFTY, Equity, and Debt data
merged_data = pd.merge(nifty_data, equity_data, on="Date", how="left")
merged_data = pd.merge(merged_data, debt_data, on="Date", how="left")

# Drop missing values
merged_data = merged_data.dropna()

# Feature engineering
merged_data["Year"] = merged_data["Date"].dt.year
merged_data["Month"] = merged_data["Date"].dt.month

# Define features and target variable
equity_features = ["Equity Net Investment"]
debt_features = ["Debt Net Investment"]
target = "Percentage Change"

# Sidebar with date selector
selected_date = st.sidebar.date_input("Select Date", min_value=merged_data["Date"].min(), max_value=merged_data["Date"].max())

# Filter data for the selected date
selected_data = merged_data[merged_data["Date"] == selected_date]

# Display selected data
st.write("Selected Data:")
st.write(selected_data)

# Linear Regression model for Equity
linear_model_equity = LinearRegression()
linear_model_equity.fit(merged_data[equity_features], merged_data[target])
linear_pred_equity = linear_model_equity.predict(selected_data[equity_features])

# Linear Regression model for Debt
linear_model_debt = LinearRegression()
linear_model_debt.fit(merged_data[debt_features], merged_data[target])
linear_pred_debt = linear_model_debt.predict(selected_data[debt_features])

# Random Forest Regression model for Equity
rf_model_equity = RandomForestRegressor()
rf_model_equity.fit(merged_data[equity_features], merged_data[target])
rf_pred_equity = rf_model_equity.predict(selected_data[equity_features])

# Random Forest Regression model for Debt
rf_model_debt = RandomForestRegressor()
rf_model_debt.fit(merged_data[debt_features], merged_data[target])
rf_pred_debt = rf_model_debt.predict(selected_data[debt_features])

# Gradient Boosting Regression model for Equity
gb_model_equity = GradientBoostingRegressor()
gb_model_equity.fit(merged_data[equity_features], merged_data[target])
gb_pred_equity = gb_model_equity.predict(selected_data[equity_features])

# Gradient Boosting Regression model for Debt
gb_model_debt = GradientBoostingRegressor()
gb_model_debt.fit(merged_data[debt_features], merged_data[target])
gb_pred_debt = gb_model_debt.predict(selected_data[debt_features])

# LSTM model for Equity
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=(input_shape, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

X_train_lstm_equity = np.reshape(merged_data[equity_features], (merged_data.shape[0], len(equity_features), 1))
X_test_lstm_equity = np.reshape(selected_data[equity_features], (1, len(equity_features), 1))

lstm_model_equity = create_lstm_model(len(equity_features))
lstm_model_equity.fit(X_train_lstm_equity, merged_data[target], epochs=50, batch_size=32, verbose=0)
lstm_pred_equity = lstm_model_equity.predict(X_test_lstm_equity).flatten()

# LSTM model for Debt
X_train_lstm_debt = np.reshape(merged_data[debt_features], (merged_data.shape[0], len(debt_features), 1))
X_test_lstm_debt = np.reshape(selected_data[debt_features], (1, len(debt_features), 1))

lstm_model_debt = create_lstm_model(len(debt_features))
lstm_model_debt.fit(X_train_lstm_debt, merged_data[target], epochs=50, batch_size=32, verbose=0)
lstm_pred_debt = lstm_model_debt.predict(X_test_lstm_debt).flatten()

# ARIMA model for Equity
arima_model_equity = ARIMA(merged_data[target], order=(5, 1, 0))
arima_result_equity = arima_model_equity.fit()
arima_pred_equity = arima_result_equity.predict(start=len(merged_data), end=len(merged_data) + 1, typ='levels')

# ARIMA model for Debt
arima_model_debt = ARIMA(merged_data[target], order=(5, 1, 0))
arima_result_debt = arima_model_debt.fit()
arima_pred_debt = arima_result_debt.predict(start=len(merged_data), end=len(merged_data) + 1, typ='levels')

# VAR model for Equity
var_model_equity = VAR(merged_data[["Percentage Change", "Equity Net Investment"]])
var_result_equity = var_model_equity.fit()
var_pred_equity = var_result_equity.forecast(merged_data[["Percentage Change", "Equity Net Investment"]].values, steps=1)

# VAR model for Debt
var_model_debt = VAR(merged_data[["Percentage Change", "Debt Net Investment"]])
var_result_debt = var_model_debt.fit()
var_pred_debt = var_result_debt.forecast(merged_data[["Percentage Change", "Debt Net Investment"]].values, steps=1)

# Plotting
st.title("Comparison of Predictions with NIFTY Percentage Change for Selected Date")

plt.figure(figsize=(10, 6))

# Plotting for Equity
plt.plot(["Linear Regression", "Random Forest", "Gradient Boosting", "LSTM", "ARIMA", "VAR"],
         [linear_pred_equity[0], rf_pred_equity[0], gb_pred_equity[0], lstm_pred_equity[0], arima_pred_equity.iloc[0], var_pred_equity[0, 0]],
         label="Equity Models")

# Plotting for Debt
plt.plot(["Linear Regression", "Random Forest", "Gradient Boosting", "LSTM", "ARIMA", "VAR"],
         [linear_pred_debt[0], rf_pred_debt[0], gb_pred_debt[0], lstm_pred_debt[0], arima_pred_debt.iloc[0], var_pred_debt[0, 0]],
         label="Debt Models")

# Plotting for Actual NIFTY Percentage Change
plt.axhline(y=selected_data[target].values[0], color='r', linestyle='--', label="Actual NIFTY Percentage Change")

plt.title(f"Comparison for Date: {selected_date}")
plt.xlabel("Models")
plt.ylabel("NIFTY Percentage Change")
plt.legend()
st.pyplot(plt)
