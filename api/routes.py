from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import traceback

from .src.forecast import forecast_next_7_days
from .src.features import compute_features
from .src.utils import load_model_and_features

main = Blueprint("main", __name__)

# Load model and features once at startup
model, feature_columns = load_model_and_features()

@main.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🚀 Forecast API is live."})

@main.route("/forecast", methods=["POST"])
def forecast_endpoint():
    try:
        data = request.get_json()
        if not data or "market_data" not in data:
            return jsonify({"error": "Missing 'market_data' in JSON"}), 400

        df = pd.DataFrame(data["market_data"])
        df["Date"] = pd.to_datetime(df["Date"])

        forecast_df = forecast_next_7_days(
            df=df,
            model=model,
            feature_columns=feature_columns,
            mae_history=data.get("mae_history", []),
            retrain_callback=None  # Add retrain logic if needed
        )

        result = forecast_df.to_dict(orient="records")
        return jsonify({"forecast": result})

    except Exception as e:
        current_app.logger.error(f"❌ Forecast failed: {e}")
        current_app.logger.debug(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500
