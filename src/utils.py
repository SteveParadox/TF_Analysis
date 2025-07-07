import joblib
import pandas as pd
from features import compute_features
from sklearn.ensemble import RandomForestRegressor  # or your preferred model

def load_training_data(path):
    """Load training data from a CSV or data store."""
    df = pd.read_csv(path, parse_dates=['Date'])
    df = compute_features(df)
    df.dropna(inplace=True)
    return df

def retrain_model(data_path, feature_columns, target_col='Close', model_save_path='model.pkl'):
    """
    Retrain model using fresh data and save updated model.

    Args:
        data_path (str): Path to training CSV file.
        feature_columns (List[str]): Columns used as model input.
        target_col (str): Column to predict.
        model_save_path (str): File path to save trained model.

    Returns:
        Trained model object.
    """
    print("🔄 Retraining model from data:", data_path)

    df = load_training_data(data_path)
    X = df[feature_columns]
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_save_path)
    print(f"✅ Model retrained and saved to: {model_save_path}")

    return model

def load_model(model_path='model.pkl'):
    """Load trained model from disk."""
    return joblib.load(model_path)
