# Stock Market Data Project

## Overview
This project provides stock price prediction and analysis for major tech stocks (AAPL, TSLA, GOOG, MSFT, AMZN) using machine learning models. It includes:
- LSTM models for advanced time series forecasting
- Linear regression models for baseline predictions
- A FastAPI backend for serving predictions and price history

## Folder Structure
- `dataset/`: Pre-processed CSV files for each stock
- `models/`: Saved machine learning models (`.keras` for LSTM, `.joblib` for linear regression)
- `notebook/`: Jupyter notebooks for model development and evaluation
- `server/`: FastAPI backend (`server.py`)

## Backend API Endpoints
- `POST /predict/{ticker}`: Predict next day's price using linear regression. Send JSON `{ "prices": [last_5_prices] }`.
- `GET /last5/{ticker}`: Get last 5 prices from dataset for a ticker.
- `GET /history/{ticker}`: Get last 20 dates and prices, plus a buy recommendation.

## How to Use
1. Start the FastAPI server:
   ```bash
   uvicorn server.server:app --reload
   ```
2. Use the endpoints to:
   - Fetch recent prices
   - Get buy recommendations
   - Predict next day's price

## Model Details
- **LSTM Models**: Built in notebooks, saved as `.keras` files. Used for deep learning price prediction.
- **Linear Regression Models**: Trained for each stock, saved as `.joblib` files. Used for quick baseline predictions.

## Example Prediction Request
```bash
curl -X POST "http://localhost:8000/predict/TSLA" -H "Content-Type: application/json" -d '{"prices": [123.45, 124.56, 125.67, 126.78, 127.89]}'
```

## Recommendation Logic
- The `/history/{ticker}` endpoint recommends "Buy" if the price is trending up in the last 5 days, otherwise "Do Not Buy".

## Requirements
- Python 3.10+
- FastAPI
- scikit-learn
- joblib
- pandas
- numpy

## Credits
Created with GitHub Copilot (GPT-4.1) and FastAPI.
