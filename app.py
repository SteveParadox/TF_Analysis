from flask import Flask, request, jsonify, Response
import pandas as pd
import joblib
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from indicators import add_technical_indicators , add_candle_patterns

from _features import features_list


app = Flask(__name__)

# === Load Models ===
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model.json")

arima_model = joblib.load("models/arima_model.pkl")
meta_model = joblib.load("models/meta_model.pkl")

with open("models/prophet_model.pkl", "rb") as f:
    prophet = joblib.load(f)

# === Auto Predict Endpoint ===
@app.route('/auto-predict', methods=['GET'])
def auto_predict():
    plot = request.args.get("plot", "false").lower() == "true"

    df = yf.download("EURUSD=X", interval="1d", period="5y", progress=False)
    df.reset_index(inplace=True)
    
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance. Try a shorter period or check your internet.")
    
    df = add_technical_indicators(df)
    df = add_candle_patterns(df)
    df.dropna(inplace=True)

    features = features_list

    # Feature Engineering
    df.dropna(inplace=True)
    latest = df.iloc[-1:].copy()
    latest.columns = [col[0].strip() if isinstance(col, tuple) else col.strip() for col in latest.columns]

    # XGBoost
    dmatrix = xgb.DMatrix(latest[features])
    xgb_pred = xgb_model.predict(dmatrix)[0]

    # ARIMA
    try:
        arima_pred = arima_model.forecast(steps=1)[0]
    except:
        arima_pred = xgb_pred

    # Prophet
    try:
        future = prophet.make_future_dataframe(periods=1)
        forecast = prophet.predict(future)
        prophet_pred = forecast.iloc[-1]['yhat']
    except:
        prophet_pred = xgb_pred

    # Meta-model
    stack = pd.DataFrame({
        "xgb": [xgb_pred],
        "arima": [arima_pred],
        "prophet": [prophet_pred]
    })
    final_pred = meta_model.predict(stack)[0]

    if not plot:
        return jsonify({
            "xgb": float(xgb_pred),
            "arima": float(arima_pred),
            "prophet": float(prophet_pred),
            "final_prediction": float(final_pred)
        })

    # === PLOT ===
    labels = ['XGBoost', 'ARIMA', 'Prophet', 'Meta Model']
    values = [xgb_pred, arima_pred, prophet_pred, final_pred]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color='royalblue')
    plt.title("Forecasted Price by Model")
    plt.ylabel("Predicted Value")
    plt.grid(True, axis='y', linestyle='--')
    plt.tight_layout()

    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    html = f"""
    <html><body>
    <h2>Model Forecast Comparison</h2>
    <img src='data:image/png;base64,{encoded}'/>
    </body></html>
    """

    return Response(html, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
