# StockAdvisorAI

An AI model written in Python that advises investors to BUY, SELL, or HOLD a stock that is submitted to the AI model for analysis.

# Clone This Repository

Run this command below to clone this repo and begin using it.

```
git clone https://github.com/pavondunbar/StockAdvisorAI && cd StockAdvisorAI
```

# Requirements

Python3.  You can check to see if Python3 is on your system by running the command below:

```
python3 --version
```

If a version number prints out, you are good to go.

# Create a Python Virtual Environment

It is recommended to create a Python virtual environment to run this AI model.  Running a virtual environment will prevent library conflicts with other Python projects or applications you may have on your system.

Create a virtual environment named StockAdvisor (or whatever you want to call it) by running the following command:

```
python3 -m venv StockAdvisor
```

If your virtual environment isn't created, you can use this command to create it:

```
virtualenv StockAdvisor
```

# Activate the Virtual Environment

After you've created your Python virtual environment, activate it by running the command below:

```
source StockAdvisor/bin/activate
```

# Install Required libraries

Run the following command to install the libraries you need to run the AI model as well as train it:

```
pip install yfinance numpy scikit-learn tensorflow ta
```

# Start the AI Model

Now the fun begins!  Run the following command to boot up the StockAdvisor AI model:

```
python3 StockAdvisorAI.py
```

The AI model will initialize and ask you to submit a stock ticker symbol for a company

After you submit the stock ticker symbol, the AI will do 25 iterations over the dataset.  StockAdvisorAI is effectively "training itself" to find patterns in the datasets so it can make a determination.

Once StockAdvisorAI finishes training, it will output a decision for you to either BUY, SELL, or HOLD the stock.

# Closing Notes

1. This AI model uses the Yahoo Finance (yfinance) dataset to train itself
2. This AI model trains itself based on these 4 things:
          a. Historical Price Data
          b. Volume
          c. The RSI (Relative Strength Index) technical indicator
          d. The MACD (Moving Average Convergence Divergence) technical indicator
3. StockAdvisorAI does not use Sentiment Score when analyzing stock data
4. StockAdvisorAI analyzes data from 01/10/2010 to the current day; however, only 3 months of past data is used to make a recommendation.

# Disclaimer

If you are enjoying StockAdvisorAI, that is awesome and I appreciate it.  But please...**do not use StockAdvisorAI as a "final decision maker" when analyzing a certain stock.** 

As a human, you should still do your research and due diligence before you amake any investment decisions to buy, sell, or hold certain stocks.

By using StockAdvisorAI, you agree to hold the AI, its creator Pavon Dunbar, or any affiliates or representatives of StockAdvisorAI harmless from any financial damages, errors, etc that may result from using StockAdvisorAI.

StockAdvisorAI is a **work in progress** and will be consistently updated.


