import os
import joblib
import traceback
import pandas as pd
from flask import Blueprint, jsonify, current_app

from src.mt5 import trade_from_forecast  # <-- Add this at the top

from src.forecast import forecast_next_7_days
from src.utils import retrain_model, load_training_data, load_model_and_features
import MetaTrader5 as mt5

from dotenv import load_dotenv
import oandapyV20

# Load environment variables from .env
load_dotenv()

# Get OANDA API key from environment
OANDA_API_KEY = os.getenv("OANDA_API_KEY")

# Ensure the key is found
if not OANDA_API_KEY:
    raise EnvironmentError("❌ OANDA_API_KEY not found in .env")

client = oandapyV20.API(access_token=OANDA_API_KEY)

main = Blueprint("main", __name__)



@main.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🚀 Forecast API is live."})


@main.route("/forecast", methods=["GET"])
def forecast_endpoint():
    try:
        model, _ = load_model_and_features()
        df = load_training_data()
        df["Date"] = pd.to_datetime(df["Date"])

        forecast_df = forecast_next_7_days(
            df=df,
            mae_history=[],
            model=model,
            retrain_callback=lambda: retrain_model(
                data_path=None,
                model_save_path="models/meta_model_rf.pkl"
            )
        )

        result = forecast_df.to_dict(orient="records")
        return jsonify({"forecast": result})

    except Exception as e:
        current_app.logger.error(f" Forecast failed: {e}")
        current_app.logger.debug(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@main.route("/trade", methods=["POST"])
def trade_endpoint():
    try:
        df_path = os.path.join("models", "original_df.pkl")
        model_path = os.path.join("models", "meta_model_rf.pkl")

        if not os.path.exists(df_path) or not os.path.exists(model_path):
            return jsonify({"error": "📦 Required model or data not found."}), 404

        df = joblib.load(df_path)
        df["Date"] = pd.to_datetime(df["Date"])

        # Load trained model
        model = joblib.load(model_path)

        # Dynamically extract the feature columns from compute_features
        sample_df = compute_features(df.copy())
        feature_columns = sample_df.drop(columns=["Date", "Open", "High", "Low", "Close", "Volume"]).columns.tolist()

        # MT5 credentials from env
        login = int(os.getenv("MT5_LOGIN"))
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")

        if not all([login, password, server]):
            return jsonify({"error": "❌ MT5 credentials missing from environment."}), 500

        # Run trade logic
        trade_from_forecast(
            symbol="EURUSD",  # or make this dynamic later
            model=model,
            feature_columns=feature_columns,
            login=login,
            password=password,
            server=server
        )

        return jsonify({"message": "📈 Trade execution attempted via MT5."})

    except Exception as e:
        current_app.logger.error(f"❌ Trade route failed: {e}")
        current_app.logger.debug(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500