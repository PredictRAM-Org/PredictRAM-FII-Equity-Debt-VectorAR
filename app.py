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

# Split data into training and testing sets
X_train_equity, X_test_equity, y_train, y_test = train_test_split(
    merged_data[equity_features], merged_data[target], test_size=0.2, random_state=42
)

X_train_debt, X_test_debt, _, _ = train_test_split(
    merged_data[debt_features], merged_data[target], test_size=0.2, random_state=42
)

# Standardize features
scaler_equity = StandardScaler()
X_train_scaled_equity = scaler_equity.fit_transform(X_train_equity)
X_test_scaled_equity = scaler_equity.transform(X_test_equity)

scaler_debt = StandardScaler()
X_train_scaled_debt = scaler_debt.fit_transform(X_train_debt)
X_test_scaled_debt = scaler_debt.transform(X_test_debt)

# Linear Regression model for Equity
linear_model_equity = LinearRegression()
linear_model_equity.fit(X_train_scaled_equity, y_train)
linear_pred_equity = linear_model_equity.predict(X_test_scaled_equity)

# Linear Regression model for Debt
linear_model_debt = LinearRegression()
linear_model_debt.fit(X_train_scaled_debt, y_train)
linear_pred_debt = linear_model_debt.predict(X_test_scaled_debt)

# Random Forest Regression model for Equity
rf_model_equity = RandomForestRegressor()
rf_model_equity.fit(X_train_scaled_equity, y_train)
rf_pred_equity = rf_model_equity.predict(X_test_scaled_equity)

# Random Forest Regression model for Debt
rf_model_debt = RandomForestRegressor()
rf_model_debt.fit(X_train_scaled_debt, y_train)
rf_pred_debt = rf_model_debt.predict(X_test_scaled_debt)

# Gradient Boosting Regression model for Equity
gb_model_equity = GradientBoostingRegressor()
gb_model_equity.fit(X_train_scaled_equity, y_train)
gb_pred_equity = gb_model_equity.predict(X_test_scaled_equity)

# Gradient Boosting Regression model for Debt
gb_model_debt = GradientBoostingRegressor()
gb_model_debt.fit(X_train_scaled_debt, y_train)
gb_pred_debt = gb_model_debt.predict(X_test_scaled_debt)

# LSTM model for Equity
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=(input_shape, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

X_train_lstm_equity = np.reshape(X_train_scaled_equity, (X_train_scaled_equity.shape[0], X_train_scaled_equity.shape[1], 1))
X_test_lstm_equity = np.reshape(X_test_scaled_equity, (X_test_scaled_equity.shape[0], X_test_scaled_equity.shape[1], 1))

lstm_model_equity = create_lstm_model(X_train_scaled_equity.shape[1])
lstm_model_equity.fit(X_train_lstm_equity, y_train, epochs=50, batch_size=32, verbose=0)
lstm_pred_equity = lstm_model_equity.predict(X_test_lstm_equity).flatten()

# LSTM model for Debt
X_train_lstm_debt = np.reshape(X_train_scaled_debt, (X_train_scaled_debt.shape[0], X_train_scaled_debt.shape[1], 1))
X_test_lstm_debt = np.reshape(X_test_scaled_debt, (X_test_scaled_debt.shape[0], X_test_scaled_debt.shape[1], 1))

lstm_model_debt = create_lstm_model(X_train_scaled_debt.shape[1])
lstm_model_debt.fit(X_train_lstm_debt, y_train, epochs=50, batch_size=32, verbose=0)
lstm_pred_debt = lstm_model_debt.predict(X_test_lstm_debt).flatten()

# ARIMA model for Equity
arima_model_equity = ARIMA(merged_data[target], order=(5, 1, 0))
arima_result_equity = arima_model_equity.fit()
arima_pred_equity = arima_result_equity.predict(start=len(merged_data), end=len(merged_data) + len(X_test_equity) - 1, typ='levels')

# ARIMA model for Debt
arima_model_debt = ARIMA(merged_data[target], order=(5, 1, 0))
arima_result_debt = arima_model_debt.fit()
arima_pred_debt = arima_result_debt.predict(start=len(merged_data), end=len(merged_data) + len(X_test_debt) - 1, typ='levels')

# VAR model for Equity
var_model_equity = VAR(merged_data[["Percentage Change", "Equity Net Investment"]])
var_result_equity = var_model_equity.fit()
var_pred_equity = var_result_equity.forecast(merged_data[["Percentage Change", "Equity Net Investment"]].values, steps=len(X_test_equity))

# VAR model for Debt
var_model_debt = VAR(merged_data[["Percentage Change", "Debt Net Investment"]])
var_result_debt = var_model_debt.fit()
var_pred_debt = var_result_debt.forecast(merged_data[["Percentage Change", "Debt Net Investment"]].values, steps=len(X_test_debt))

# Plotting for Equity and NIFTY (Date-wise)
st.title("Comparison of Equity Predictions with NIFTY Percentage Change (Date-wise)")
plt.figure(figsize=(10, 6))
for i in range(len(y_test)):
    date = y_test.index[i]
    
    plt.plot([date, date], [y_test.iloc[i], linear_pred_equity[i]], label="Linear Regression")
    plt.plot([date, date], [y_test.iloc[i], rf_pred_equity[i]], label="Random Forest")
    plt.plot([date, date], [y_test.iloc[i], gb_pred_equity[i]], label="Gradient Boosting")
    plt.plot([date, date], [y_test.iloc[i], lstm_pred_equity[i]], label="LSTM")
    plt.plot([date, date], [y_test.iloc[i], arima_pred_equity.iloc[i]], label="ARIMA")
    plt.plot([date, date], [y_test.iloc[i], var_pred_equity[i, 0]], label="VAR")

plt.title("Equity Predictions vs. NIFTY Percentage Change (Date-wise)")
plt.xlabel("Date")
plt.ylabel("NIFTY Percentage Change")
plt.legend()
st.pyplot(plt)

# Plotting for Debt and NIFTY (Date-wise)
st.title("Comparison of Debt Predictions with NIFTY Percentage Change (Date-wise)")
plt.figure(figsize=(10, 6))
for i in range(len(y_test)):
    date = y_test.index[i]
    
    plt.plot([date, date], [y_test.iloc[i], linear_pred_debt[i]], label="Linear Regression")
    plt.plot([date, date], [y_test.iloc[i], rf_pred_debt[i]], label="Random Forest")
    plt.plot([date, date], [y_test.iloc[i], gb_pred_debt[i]], label="Gradient Boosting")
    plt.plot([date, date], [y_test.iloc[i], lstm_pred_debt[i]], label="LSTM")
    plt.plot([date, date], [y_test.iloc[i], arima_pred_debt.iloc[i]], label="ARIMA")
    plt.plot([date, date], [y_test.iloc[i], var_pred_debt[i, 0]], label="VAR")

plt.title("Debt Predictions vs. NIFTY Percentage Change (Date-wise)")
plt.xlabel("Date")
plt.ylabel("NIFTY Percentage Change")
plt.legend()
st.pyplot(plt)
