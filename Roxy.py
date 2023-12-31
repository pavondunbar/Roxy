# Introducing...Roxy.
# An AI Model developed by Pavon Dunbar

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

# Suppress the warning
warnings.filterwarnings(action='ignore', category=UserWarning)

LOOKBACK = 90

def get_data(stock_ticker):
    today = datetime.today().strftime('%Y-%m-%d')  # Get today's date in the format 'YYYY-MM-DD'
    data = yf.download(stock_ticker, start="2010-01-01", end=today, progress=False)
    return data

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    """Compute the MACD and Signal Line indicators."""
    # Compute short and long moving averages
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    # Compute the MACD value
    data['MACD'] = short_ema - long_ema

    # Compute the Signal line value
    data['MACD_signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    return data

def add_indicators(data):
    # Add RSI (Relative Strength Index)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

    # Add Stochastic Oscillator
    data['%K'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Compute MACD and Signal Line
    data = compute_macd(data)

    # Add Moving Averages
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()

    # Add Bollinger Bands
    data['BollingerB_20'] = ta.volatility.BollingerBands(data['Close'], window=20).bollinger_mavg()
    data['BollingerB_20_upper'] = ta.volatility.BollingerBands(data['Close'], window=20).bollinger_hband()
    data['BollingerB_20_lower'] = ta.volatility.BollingerBands(data['Close'], window=20).bollinger_lband()

    # Add Average True Range (ATR)
    data['ATR_14'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14)

    # Add Commodity Channel Index (CCI)
    data['CCI_14'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=14).cci()

    # Add On-Balance Volume (OBV)
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

    return data

def preprocess_data(data):
    data = add_indicators(data)

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Create a separate scaler for the 'Close' column
    close_scaler = MinMaxScaler()
    numeric_data['Close'] = close_scaler.fit_transform(numeric_data['Close'].values.reshape(-1, 1))

    # Scale the entire dataset
    scaler = MinMaxScaler()
    numeric_data = numeric_data.dropna().values
    scaled_data = scaler.fit_transform(numeric_data)

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

def get_advice(model, last_sequence, close_scaler, data):
    prediction = model.predict(last_sequence)
    last_real_value = close_scaler.inverse_transform(last_sequence[0, -1, :1].reshape(-1, 1))[0, 0]
    predicted_value = close_scaler.inverse_transform(prediction.reshape(-1, 1))[0, 0]
    
    last_day = data.iloc[-1]
    rsi = last_day['RSI']
    macd = last_day['MACD']
    signal = last_day['MACD_signal']
    sma_50 = last_day['SMA_50']
    sma_200 = last_day['SMA_200']
    stk = last_day['%K']
    stoch_d = last_day['%D']
    volume = last_day['Volume']
    average_volume = data['Volume'].rolling(window=5).mean().iloc[-1] # 5-day average volume

    advice_reasons = []

    # RSI reasoning
    if rsi < 30:
        advice_reasons.append("RSI indicates the stock might be oversold.")
    elif rsi > 70:
        advice_reasons.append("RSI indicates the stock might be overbought.")
    else:
        advice_reasons.append("RSI is neutral.")
    
    # Additional RSI reasoning
    # RSI divergence
    if rsi < 70 and data['Close'].iloc[-5:].pct_change().mean() < 0:
        advice_reasons.append("RSI divergence suggests potential reversal.")
    
    # MACD reasoning
    if macd > signal:
        advice_reasons.append("MACD is above the signal line, indicating a bullish crossover.")
    elif macd < signal:
        advice_reasons.append("MACD is below the signal line, indicating a bearish crossover.")
    
    # Additional MACD reasoning
    # MACD histogram analysis
    if macd > 0 and data['Close'].iloc[-5:].pct_change().mean() > 0:
        advice_reasons.append("MACD histogram and price trend support bullish momentum.")
    
    # Moving Averages reasoning
    if sma_50 > sma_200:
        advice_reasons.append("The 50-day Simple Moving Average is above the 200-day Simple Moving Average, indicating a bullish trend.")
    else:
        advice_reasons.append("The 50-day Simple Moving Average is below the 200-day Simple Moving Average, indicating a bearish trend.")
    
    # Additional Moving Averages reasoning
    # Golden cross and death cross
    if sma_50 > sma_200 and data['Close'].iloc[-5:].pct_change().mean() > 0:
        advice_reasons.append("Golden cross and recent positive price trend suggest a strong bullish signal.")
    elif sma_50 < sma_200 and data['Close'].iloc[-5:].pct_change().mean() < 0:
        advice_reasons.append("Death cross and recent negative price trend suggest a strong bearish signal.")
    
    # Stochastic Oscillator reasoning
    if stk > stoch_d:
        advice_reasons.append("Stochastic Oscillator shows a bullish momentum.")
    elif stk < stoch_d:
        advice_reasons.append("Stochastic Oscillator shows a bearish momentum.")
    
    # Additional Stochastic Oscillator reasoning
    # Custom overbought/oversold levels
    if stk > 80:
        advice_reasons.append("Stochastic Oscillator indicates overbought conditions.")
    elif stk < 20:
        advice_reasons.append("Stochastic Oscillator indicates oversold conditions.")
    
    # Historical prices reasoning
    if data['Close'].iloc[-5:].pct_change().mean() > 0:
        advice_reasons.append("The stock has been on an upward trend over the last 5 days.")
    else:
        advice_reasons.append("The stock has been on a downward trend over the last 5 days.")
    
    # Volume reasoning
    if volume > 1.5 * average_volume:
        advice_reasons.append("The trading volume is significantly higher than the 5-day average, indicating strong interest in the stock.")
    
    if predicted_value > last_real_value:
        advice = "BUY"
    elif predicted_value < last_real_value:
        advice = "SELL"
    else:
        advice = "HOLD"

    return advice, advice_reasons

def plot_stock_chart(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.title('Historical Stock Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(stock_name):
    data = get_data(stock_name)
    x, y, scaler, close_scaler = preprocess_data(data)
    
    model = build_model((x.shape[1], x.shape[2]))
    model.fit(x, y, epochs=25, batch_size=32)

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    last_sequence = scaler.transform(numeric_data[-LOOKBACK:].dropna()).reshape(1, LOOKBACK, -1)
    advice, reasons = get_advice(model, last_sequence, close_scaler, data)
    
    # Get the current stock price
    current_price = data['Close'].iloc[-1]
    
    # Output the chart
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.scatter(data.index[-1], current_price, label=f'Current Price: ${current_price:.2f}', color='red', marker='o', s=100)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Stock Price Chart for {stock_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the recommendations and reasons in numbered bulleted format
    print("My Post-Analysis Results:")
    for i, reason in enumerate(reasons, start=1):
        print(f"{i}. {reason}")

    print(f"After analyzing {stock_name} as you requested, I recommend the following: {advice}")

stock_name = input("Hello, I'm Roxy. Enter the stock ticker symbol (in all caps, ex: AAPL, TSLA) you want me to analyze: ")
main(stock_name)
