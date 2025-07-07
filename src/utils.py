import os
import joblib
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .features import compute_features

# === Logging Setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# === Load Data ===
def load_training_data(path=None):
    """
    Load and preprocess training data.
    If no path is provided, loads from default saved DataFrame.

    Args:
        path (str): Optional path to CSV.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with features.
    """
    try:
        if path:
            logger.info(f"Loading data from {path}")
            df = pd.read_csv(path, parse_dates=['Date'])
        else:
            df_path = os.path.join("models", "original_df.pkl")
            logger.info(f"Loading default data from {df_path}")
            df = joblib.load(df_path)
        return df

    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise


# === Load Model ===
def load_model_and_features(model_path="models/meta_model_rf.pkl"):
    """
    Load the trained model and extract feature columns.

    Args:
        model_path (str): Path to model file.

    Returns:
        model, List[str]: Model object and feature names.
    """
    try:
        model = joblib.load(model_path)
        feature_columns = model.feature_names_in_.tolist()
        logger.info(f"Model loaded from {model_path}")
        return model, feature_columns

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# === Retrain Model ===
def retrain_model(data_path=None, feature_columns=None, target_col='Close', model_save_path='models/meta_model_rf.pkl'):
    """
    Retrain RandomForest model using latest data.

    Args:
        data_path (str): Optional path to training CSV.
        feature_columns (List[str]): Feature columns for training.
        target_col (str): Target column.
        model_save_path (str): Where to save the trained model.

    Returns:
        Trained model object.
    """
    logger.info("Starting model retraining...")

    df = load_training_data(data_path)
    if not feature_columns:
        raise ValueError("Feature columns must be provided for retraining.")

    X = df[feature_columns]
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_save_path)
    logger.info(f"Model retrained and saved to: {model_save_path}")

    return model
