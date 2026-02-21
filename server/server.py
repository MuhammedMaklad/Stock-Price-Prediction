from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pathlib import Path
import pandas as pd

app = FastAPI()

# Model paths
MODEL_PATHS = {
    'AAPL': '../models/AAPL_linear_regression.joblib',
    'TSLA': '../models/TSLA_linear_regression.joblib',
    'GOOG': '../models/GOOG_linear_regression.joblib',
    'MSFT': '../models/MSFT_linear_regression.joblib',
    'AMZN': '../models/AMZN_linear_regression.joblib',
}

# Load models at startup
MODELS = {}
for ticker, path in MODEL_PATHS.items():
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    MODELS[ticker] = joblib.load(full_path)

# Dataset paths
DATASET_PATHS = {
    'AAPL': '../dataset/Pre_Processed_AAPL.csv',
    'TSLA': '../dataset/Pre_Processed_TSLA.csv',
    'GOOG': '../dataset/Pre_Processed_GOOG.csv',
    'MSFT': '../dataset/Pre_Processed_MSFT.csv',
    'AMZN': '../dataset/Pre_Processed_AMZN.csv',
}

class PriceInput(BaseModel):
    prices: list[float]  # last 5 days prices

@app.post("/predict/{ticker}")
def predict_next_price(ticker: str, price_input: PriceInput):
    """
    Predict the next day's adjusted close price for a given stock ticker using its linear regression model.

    Args:
        ticker (str): Stock ticker symbol (AAPL, TSLA, GOOG, MSFT, AMZN).
        price_input (PriceInput): JSON body with 'prices', a list of last 5 days' adjusted close prices.

    Returns:
        dict: Contains ticker and predicted next price, or error message if input is invalid.

    Example request:
        POST /predict/TSLA
        {
            "prices": [123.45, 124.56, 125.67, 126.78, 127.89]
        }
    """
    ticker = ticker.upper()
    if ticker not in MODELS:
        return {"error": f"Model for {ticker} not found."}
    prices = price_input.prices
    if len(prices) != 5:
        return {"error": "Input must be a list of 5 prices (last 5 days)."}
    model = MODELS[ticker]
    X = np.array(prices).reshape(1, -1)
    pred = model.predict(X)[0]
    return {"ticker": ticker, "predicted_next_price": float(pred)}

@app.get("/last5/{ticker}")
def get_last_5_prices(ticker: str):
    """
    Get the last 5 adjusted close prices for a given stock ticker from the dataset folder.

    Args:
        ticker (str): Stock ticker symbol (AAPL, TSLA, GOOG, MSFT, AMZN).

    Returns:
        dict: Contains ticker and last 5 prices, or error message if ticker or file is invalid.
    """
    ticker = ticker.upper()
    if ticker not in DATASET_PATHS:
        return {"error": f"Dataset for {ticker} not found."}
    dataset_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), DATASET_PATHS[ticker])))
    if not dataset_path.exists():
        return {"error": f"Dataset file for {ticker} not found at {dataset_path}"}
    try:
        df = pd.read_csv(dataset_path)
        prices = df['Adj. Close'].tail(5).tolist()
        return {"ticker": ticker, "last_5_prices": prices}
    except Exception as e:
        return {"error": str(e)}

@app.get("/history/{ticker}")
def get_last_20_history(ticker: str):
    """
    Get the last 20 dates and adjusted close prices for a given stock ticker from the dataset folder.
    Also provides a simple buy recommendation based on price trend.

    Args:
        ticker (str): Stock ticker symbol (AAPL, TSLA, GOOG, MSFT, AMZN).

    Returns:
        dict: Contains ticker, last 20 dates and prices, and a recommendation.
    """
    ticker = ticker.upper()
    if ticker not in DATASET_PATHS:
        return {"error": f"Dataset for {ticker} not found."}
    dataset_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), DATASET_PATHS[ticker])))
    if not dataset_path.exists():
        return {"error": f"Dataset file for {ticker} not found at {dataset_path}"}
    try:
        df = pd.read_csv(dataset_path)
        last_20 = df[['Date', 'Adj. Close']].tail(20)
        dates = last_20['Date'].tolist()
        prices = last_20['Adj. Close'].tolist()
        # Simple recommendation: buy if price is trending up in last 5 days
        trend = prices[-1] > prices[-5]
        recommendation = "Buy" if trend else "Do Not Buy"
        return {
            "ticker": ticker,
            "last_20": [{"date": d, "price": p} for d, p in zip(dates, prices)],
            "recommendation": recommendation
        }
    except Exception as e:
        return {"error": str(e)}

