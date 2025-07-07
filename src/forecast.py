import os
import numpy as np
import pandas as pd
import logging

from .features import compute_features
from .features import should_retrain
from .utils import load_model_and_features, retrain_model

# === Configure Logging ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/pipeline.log")
formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# === Load model and features once ===
try:
    model, feature_columns = load_model_and_features()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def forecast_next_7_days(df, model, mae_history=None, retrain_callback=None, mae_threshold=0.02):
    """
    Run 7-day recursive forecast with uncertainty bands and optional retraining.

    Args:
        df (DataFrame): Market data with ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        mae_history (List[float], optional): Historical MAEs to evaluate model drift
        retrain_callback (Callable, optional): Function to retrain model if needed
        mae_threshold (float): Threshold for triggering model retraining

    Returns:
        DataFrame with ['Date', 'Predicted_Close', 'Lower_Band', 'Upper_Band']
    """
    logger.info("Starting 7-day forecast pipeline...")
    history = df.sort_values("Date").copy().reset_index(drop=True)
    future_preds = []

    for step in range(7):
        next_date = history.iloc[-1]['Date'] + pd.Timedelta(days=1)
        last_row = history.iloc[-1]

        atr = history.iloc[-2].get('ATR', np.nan)
        if pd.isna(atr):
            atr = last_row['Close'] * 0.01
            logger.warning(f"[{next_date.date()}] ATR missing — using fallback volatility estimate.")

        high_noise = np.random.uniform(0.3, 1.0) * atr
        low_noise = np.random.uniform(0.3, 1.0) * atr

        new_row = {
            'Date': next_date,
            'Open': last_row['Close'],
            'High': last_row['Close'] + high_noise,
            'Low': last_row['Close'] - low_noise,
            'Close': np.nan,
            'Volume': last_row['Volume']
        }

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        tail_window = 100
        tail = compute_features(history.tail(tail_window).copy())
        tail.ffill(inplace=True)
        tail.fillna(0, inplace=True)
        history.update(tail)

        X_next = history.iloc[[-1]].reindex(columns=feature_columns, fill_value=0)
        #y_preds = model.predict(X_next)[0] 
        y_preds = np.array([model.predict(X_next)[0] for _ in range(5)])
        y_mean = y_preds.mean()
        y_std = y_preds.std()
    
        history.at[history.index[-1], 'Close'] = y_mean
        future_preds.append((next_date, y_mean))

        logger.info(f"{next_date.date()} Forecast: {y_mean:.5f} (+/- {y_std:.5f})")

    print(future_preds)
    forecast_df = pd.DataFrame(future_preds, columns=["Date", "Predicted_Close"])
    logger.info(" Forecasting complete.")
    return forecast_df

"""

    if mae_history is not None and should_retrain(mae_history, threshold=mae_threshold):
        logger.warning(" MAE threshold exceeded. Triggering retrain callback.")
        if retrain_callback:
            retrain_callback()
        else:
            logger.warning(" No retrain callback provided.")
"""