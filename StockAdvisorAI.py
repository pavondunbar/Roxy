# Stock Advisor AI
# An AI Model developed by Pavon Dunbar

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import warnings
from sklearn.exceptions import DataConversionWarning
from datetime import datetime

# Suppress the warning
warnings.filterwarnings(action='ignore', category=UserWarning)

LOOKBACK = 90

def get_data(stock_ticker):
    today = datetime.today().strftime('%Y-%m-%d')  # Get today's date in the format 'YYYY-MM-DD'
    data = yf.download(stock_ticker, start="2010-01-01", end=today, progress=False)
    return data

def add_indicators(data):
    # Add RSI (Relative Strength Index)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

    # Add MACD (Moving Average Convergence Divergence)
    MACD = ta.trend.MACD(data['Close'])
    data['MACD'] = MACD.macd_diff()

    return data

def preprocess_data(data):
    data = add_indicators(data)
    
    # Create a separate scaler for the 'Close' column
    close_scaler = MinMaxScaler()
    data['Close'] = close_scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Scale the entire dataset
    scaler = MinMaxScaler()
    data = data.dropna().values
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x.append(scaled_data[i-LOOKBACK:i])
        y.append(scaled_data[i, 0])  # Predict the 'Close' value
    
    return np.array(x), np.array(y), scaler, close_scaler

def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def get_advice(model, last_sequence, close_scaler):
    prediction = model.predict(last_sequence)
    last_real_value = close_scaler.inverse_transform(last_sequence[0, -1, :1].reshape(-1, 1))[0,0]
    predicted_value = close_scaler.inverse_transform(prediction.reshape(-1, 1))[0,0]

    if predicted_value > last_real_value:
        return "Buy"
    elif predicted_value < last_real_value:
        return "Sell"
    else:
        return "Hold"

def main(stock_name):
    data = get_data(stock_name)
    x, y, scaler, close_scaler = preprocess_data(data)
    
    model = build_model((x.shape[1], x.shape[2]))
    model.fit(x, y, epochs=25, batch_size=32)

    last_sequence = scaler.transform(data[-LOOKBACK:].dropna()).reshape(1, LOOKBACK, -1)
    advice = get_advice(model, last_sequence, close_scaler)
    print(f"Advice for {stock_name}: {advice}")

stock_name = input("Enter the stock ticker symbol of the company you wish to analyze: ")
main(stock_name)

