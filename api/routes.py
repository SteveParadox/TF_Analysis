import os
import joblib
import traceback
import pandas as pd
from flask import Blueprint, jsonify, current_app, render_template_string

from src.mt5 import trade_from_forecast  # <-- Add this at the top

from src.forecast import forecast_next_7_days
from src.utils import retrain_model, load_training_data, load_model_and_features
import MetaTrader5 as mt5
from src.evaluate import generate_forecast_plot
from src.features import compute_features


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

        forecast_df, _ = forecast_next_7_days(
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

@main.route("/forecast/plot", methods=["GET"])
def plot_forecast():
    try:
        model, _ = load_model_and_features()
        df = load_training_data()
        
        df["Date"] = pd.to_datetime(df["Date"])

        forecast_df, history = forecast_next_7_days(
            df=df,
            mae_history=[],
            model=model,
            retrain_callback=lambda: retrain_model(
                data_path=None,
                model_save_path="models/meta_model_rf.pkl"
            )
        )
        plot_img = generate_forecast_plot(history, forecast_df)

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>📈 Forecast Preview</title>
            <style>
                body { font-family: sans-serif; text-align: center; background: #f9f9f9; }
                img { max-width: 90vw; margin-top: 20px; border: 1px solid #ccc; }
                h1 { margin-top: 30px; color: #333; }
            </style>
        </head>
        <body>
            <h1>7-Day Forecast Preview</h1>
            <img src="data:image/png;base64,{{ plot }}">
        </body>
        </html>
        """
        return render_template_string(html_template, plot=plot_img)

    except Exception as e:
        current_app.logger.error(f"❌ Plotting failed: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@main.route("/trade", methods=["GET", "POST"])
def trade_endpoint():
    try:
        # === Load Model & Training Data ===
        model, _ = load_model_and_features()
        df = load_training_data()

        if df is None or df.empty:
            return jsonify({"error": "Training data is empty or missing."}), 400

        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        if df["Date"].isnull().any():
            current_app.logger.warning("Unparseable dates found in training data.")

        # === Extract Feature Columns ===
        try:
            sample_df = compute_features(df.copy())
            sample_df.dropna(axis=1, how='all', inplace=True)  # Drop fully-null columns
            feature_columns = [
                col for col in sample_df.columns
                if col not in ["Date", "Open", "High", "Low", "Close", "Volume"]
            ]
        except Exception as fe:
            current_app.logger.exception("Feature extraction failed.")
            return jsonify({"error": "Failed to compute features."}), 500

        if not feature_columns:
            return jsonify({"error": "No feature columns found after extraction."}), 500

        # === Read MT5 Credentials from .env ===
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")

        if not all([login, password, server]):
            return jsonify({"error": "MT5 credentials missing from environment."}), 500

        try:
            login = int(login)
        except ValueError:
            return jsonify({"error": "MT5_LOGIN must be a valid integer."}), 400

        # === Run Auto-Trading Logic ===
        trade_from_forecast(
            symbol="EURUSD",
            model=model,
            feature_columns=feature_columns,
            login=login,
            password=password,
            server=server,
            df=df  # optional override with training data
        )

        return jsonify({"message": "Trade execution attempted on EURUSD via MetaTrader 5."}), 200

    except Exception as e:
        current_app.logger.error(f"Trade route failed: {str(e)}")
        current_app.logger.debug(traceback.format_exc())
        return jsonify({"error": "Internal server error during trading logic."}), 500

