# 💹 Forex Market Prediction Engine

A powerful, production-ready machine learning system for forecasting Forex prices using a stacked ensemble of **Prophet**, **XGBoost**, and **LSTM** models — with an auto-selected **meta-model** based on performance — deployed via a **Flask web application** that supports forecasting, plotting, trading (MT5), and trade log updates.

## 📌 Features

- **Advanced Feature Engineering**: Technical indicators, lag-based features, volatility, momentum, etc.
- **Triple-Model Forecasting**: Prophet (time-series), XGBoost (gradient boosting), LSTM (deep learning).
- **Meta-Model Stacking**: Best out of Ridge, Lasso, Random Forest, Gradient Boost selected via performance benchmarking.
- **Quantile Prediction**: Median, lower (10%), and upper (90%) quantile forecasts.
- **Flask API**: Routes for forecast, plotting, trading, and trade logging.
- **Live Trading Enabled**: Connected to MetaTrader5 for executing trades.
- **OANDA Support**: API integrated and secured via `.env`.

## 🧠 Model Pipeline Overview

```
Raw Data ──▶ FeatureEngineer ──▶ Train Prophet/XGBoost/LSTM ──▶ Stack with Meta-Model ──▶ Save Quantile Models ──▶ Forecast + Trade
```

## 🧪 ML Stack

| Model       | Purpose               |
|-------------|------------------------|
| **Prophet** | Trend + Seasonality    |
| **XGBoost** | Non-linear patterns    |
| **LSTM**    | Temporal dependencies  |

## 🧰 Tech Stack

- `Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`
- `Prophet`, `XGBoost`, `TensorFlow` / `Keras`
- `Flask`, `MetaTrader5`, `oandapyV20`
- `.env` for environment secrets
- `joblib` for saving models

## 📂 Project Structure

```
├── run.py
├── models/
│   ├── prophet.pkl
│   ├── xgb.pkl
│   ├── lstm.h5
│   └── meta_model_rf.pkl
├── src/
│   ├── forecast.py
│   ├── features.py
│   ├── evaluate.py
│   ├── mt5.py
│   ├── utils.py
│   └── logger.py
├── templates/
├── .env
└── README.md
```

## 🌐 Flask API Routes

| Route                  | Method | Description                                       |
|------------------------|--------|---------------------------------------------------|
| `/`                    | GET    | Health check.                                     |
| `/forecast`            | GET    | Returns next 7-day forecast as JSON.             |
| `/forecast/plot`       | GET    | Renders HTML plot with prediction & confidence.   |
| `/trade`               | GET/POST | Auto-trades based on model signals.             |
| `/update-trades`       | GET    | Updates log of closed trades.                    |

## 🔐 Environment Variables

Create a `.env` file with:

```env
OANDA_API_KEY=your_oanda_api_key
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server_name
```

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/SteveParadox/TF_Analysis.git
 
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file** (see above)

4. **Run Flask app**
   ```bash
   python run.py
   ```

## 📈 Forecast Sample Output

```json
{
  "forecast": [
    {
      "Date": "2025-07-23",
      "Predicted_Close": 1.0965,
      "Lower_Band": 1.0930,
      "Upper_Band": 1.0999
    },
    ...
  ]
}
```

## 🧠 Model Training Summary

1. **Feature Engineering**
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Lag features, rolling statistics
   - Candlestick patterns

2. **Base Models**
   - Trained independently on engineered features.

3. **Meta-Model Selection**
   - Models: Ridge, Lasso, Random Forest, Gradient Boost
   - Best model selected based on lowest validation MAE.

4. **Quantile Stacking**
   - Three meta-models trained for 10%, 50%, 90% quantiles for confidence bands.

## ⚙️ Live Trading

- Uses `MetaTrader5` API.
- Trades only on `EURUSD` symbol.
- Uses forecast quantiles to place cautious limit orders.
- Can be extended for more strategies & risk management.

## 🛠 Extending

You can easily extend the system to:
- Add more base models (LightGBM, CatBoost)
- Use alternative data (news, sentiment)
- Deploy via Docker / AWS / Streamlit
- Enable backtesting or Telegram alerts

## 🧾 License

MIT License © SteveParadox
