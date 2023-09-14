# Roxy

Roxy is an AI model written in Python that advises investors to BUY, SELL, or HOLD a stock that is submitted to her for analysis.

# Synopsis and Summary

Roxy is using a type of Artificial Neural Network (ANN) known as a Long Short-Term Memory (LSTM) network. LSTM networks are a type of recurrent neural network (RNN) architecture designed for sequence prediction and time-series analysis. They are well-suited for tasks like stock price prediction, where past data points in a time series are used to predict future values.

For Roxy, the LSTM network is being used to analyze historical stock price data and provide trading recommendations (Buy, Sell, or Hold) based on various technical indicators and patterns derived from the data. The LSTM network is trained to learn patterns and relationships in the historical stock price data, and it generates predictions for future stock prices based on these patterns.

Here's a brief overview of the major components of Roxy:

**Data Collection:** Historical stock price data is collected using the Yahoo Finance API (yfinance library) for a specified stock ticker symbol.

**Feature Engineering:** Technical indicators such as Moving Averages (SMA_50 and SMA_200), Relative Strength Index (RSI), Stochastic Oscillator (%K and %D), Moving Average Convergence Divergence (MACD), Bollinger Bands, Average True Range (ATR), Commodity Channel Index (CCI), and On-Balance Volume (OBV) are computed from the raw stock price data. These indicators are used as features for Roxy.

**Data Preprocessing:** The data is preprocessed, including feature scaling using Min-Max scaling to ensure that all features have the same scale. The data is organized into input sequences and corresponding target values for training Roxy.

**LSTM Model:** An LSTM neural network model is constructed using TensorFlow's Keras API. The model architecture includes LSTM layers with dropout for regularization and a final dense layer for regression (predicting stock prices).

**Training the Model:** Roxy is trained on the preprocessed data for a specified number of epochs (iterations).  At this time, Roxy iterates over the dataset 25 times to detect patterns and anomalies that aid in her analysis, decision, recommendations, and reasons.

**Generating Recommendations:** After training, Roxy is used to generate trading recommendations (Buy, Sell, or Hold) based on the last available data point and the reasoning provided by various technical indicators.

Overall, Roxy is using machine learning, specifically LSTM networks, for stock price prediction and trading recommendations. She combines time-series data analysis with neural network techniques to make predictions and provide insights for trading decisions.

# Requirements

Python3.  You can check to see if Python3 is on your system by running the command below:

```
python3 --version
```

If a version number prints out, you are good to go.

# Clone This Repository

Run this command below to clone this repo and begin using it.

```
git clone https://github.com/pavondunbar/Roxy && cd Roxy
```


# Create a Python Virtual Environment

It is recommended to create a Python virtual environment to run Roxy.  Running a virtual environment will prevent library conflicts with other Python projects or applications you may have on your system.

Create a virtual environment named RoxyEnv (or whatever you want to call it) by running the following command:

```
python3 -m venv RoxyEnv
```

If your virtual environment isn't created, you can use this command to create it:

```
virtualenv RoxyEnv
```

# Activate the Virtual Environment

After you've created your Python virtual environment, activate it by running the command below:

```
source RoxyEnv/bin/activate
```

# Install Required Libraries Using PIP

Run the following command to install the libraries you need to run and train Roxy:

```
pip install yfinance numpy scikit-learn tensorflow ta datetime matplotlib
```

# Initialize Roxy

Now the fun begins!  Run the following command to Initialize Roxy:

```
python3 Roxy.py
```

Roxy will initialize and ask you to submit a stock ticker symbol for a company. **Please submit a ticker in ALL CAPS (ex: AAPL, TSLA).**

After you submit the stock ticker symbol, Roxy will do 25 iterations over the dataset.  Roxy is effectively "training herself" to find patterns and anomalies in the dataset so she can make a determination.

Once Roxy finishes training, she will output a decision for you to either BUY, SELL, or HOLD the stock, reasons for her decision, and a visual chart that you can save on your computer.

# Closing Notes

1. Roxy uses the Yahoo Finance (yfinance) dataset to train herself.
2. Roxy trains herself based on the stock's Historical Price Data, Volume, the stock's RSI (Relative Strength Index), and the stock's MACD (Moving Average Convergence Divergence) among other technical indicators.
3. At this time, **Roxy does not use Sentiment Score when analyzing stock data.**
4. Roxy analyzes data from 01/10/2010 to the current day; however, only 3 months of past data is used to make a recommendation.

# Disclaimer

If you enjoy using Roxy to hopefully make you more money and be better informed to make an investment decision, that is awesome and I appreciate it.  But please...**do not use Roxy as a "final decision maker" when analyzing a certain stock.** 

As a human, you should still do your research and due diligence before you make any investment decisions to buy, sell, or hold certain stocks.

By using Roxy, you agree to hold Roxy, its creator Pavon Dunbar, or any affiliates or representatives of Roxy harmless from any financial damages, errors, etc that may result from use or misuse of Roxy.

Roxy is a **work in progress** and will be consistently updated.

# Let's Get Social!

Feel free to connect with me.  My Linktree is https://linktr.ee/pavondunbar

