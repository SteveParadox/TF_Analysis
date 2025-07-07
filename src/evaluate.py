import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# === Configure Logging ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/evaluation.log")
formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# === 1. Evaluate Core Metrics ===
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    logger.info(f"Evaluating model: {name}")
    logger.info(f"  ▸ MAE :  {mae:.6f}")
    logger.info(f"  ▸ RMSE:  {rmse:.6f}")
    logger.info(f"  ▸ R²  :  {r2:.4f}")

    print(f"📊 {name} Performance:")
    print(f"  ▸ MAE :  {mae:.6f}")
    print(f"  ▸ RMSE:  {rmse:.6f}")
    print(f"  ▸ R²  :  {r2:.4f}")
    return mae, rmse, r2

# === 2. Plot Predictions ===
def plot_predictions(dates, actual, predicted, title="Actual vs Predicted Close"):
    logger.info(f"Plotting prediction comparison: {title}")
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual", color='black')
    plt.plot(dates, predicted, label="Predicted", color='blue', linestyle='--')
    plt.title(f"📈 {title}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === 3. Plot Residuals ===
def plot_residuals(dates, actual, predicted):
    logger.info("Plotting residuals...")
    gap = [e - float(p) for p, e in zip(predicted, actual)]
    labels = dates[-len(gap):] if isinstance(dates, pd.Series) else range(len(gap))

    plt.figure(figsize=(12, 5))
    plt.bar(labels, gap, color='crimson', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Prediction Residuals (Expected - Predicted)")
    plt.xlabel("Date" if isinstance(labels[0], (str, pd.Timestamp)) else "Sample Index")
    plt.ylabel("Error (Gap)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 4. Plot Uncertainty Bands ===
def plot_with_uncertainty_bands(dates, predicted, lower, upper, actual=None):
    logger.info("Plotting uncertainty bands...")
    plt.figure(figsize=(12, 6))
    if actual is not None:
        plt.plot(dates, actual, label='Actual', color='black')
    plt.plot(dates, predicted, label='Prediction', color='blue')
    plt.fill_between(dates, lower, upper, color='skyblue', alpha=0.3, label='Confidence Band ±1σ')
    plt.title("Prediction with Uncertainty Bands")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === 5. Cross-Validation Scores ===
def crossval_scores(model, X, y, cv=5):
    logger.info("Performing cross-validation...")
    rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    logger.info(f"Cross-Validation Results (cv={cv})")
    logger.info(f"  ▸ CV RMSE: {rmse_scores.mean():.6f} ± {rmse_scores.std():.6f}")
    logger.info(f"  ▸ CV MAE : {mae_scores.mean():.6f} ± {mae_scores.std():.6f}")
    logger.info(f"  ▸ CV R²  : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    print("🔁 Cross-Validation Results:")
    print(f"  ▸ CV RMSE: {rmse_scores.mean():.6f} ± {rmse_scores.std():.6f}")
    print(f"  ▸ CV MAE : {mae_scores.mean():.6f} ± {mae_scores.std():.6f}")
    print(f"  ▸ CV R²  : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    return {
        'rmse': rmse_scores,
        'mae': mae_scores,
        'r2': r2_scores
    }
