# === Feature Engineering Functions (including Indicators) ===

# === Full Feature Set ===
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class FeatureEngineer:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)

        # === Candle Features ===
        df['Candle_Size'] = (df['Close'] - df['Open']).abs()
        df['Range'] = (df['High'] - df['Low']).replace(0, 1e-6)
        df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']
        df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']
        df['Body_to_Range'] = (df['Open'] - df['Close']).abs() / df['Range']
        df.drop(columns=['Range'], inplace=True)

        # Time features
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['is_week_start'] = (df['dayofweek'] == 0).astype(int)
        df['is_week_end'] = (df['dayofweek'] == 4).astype(int)

        # Log Return
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Technical Indicators
        close = df['Close']
        high = df['High']
        low = df['Low']

        df['RSI'] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        bb = BollingerBands(close)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = df['BB_upper'] - df['BB_lower']

        df['SMA_10'] = SMAIndicator(close, window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        df['SMA_100'] = SMAIndicator(close, window=100).sma_indicator()

        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()

        # Lag & Return Features
        df['lag_1'] = close.shift(1)
        df['lag_2'] = close.shift(2)
        df['lag_3'] = close.shift(3)
        df['return_1d'] = close.pct_change(periods=1) * 100

        # Rolling Features
        df['sma_5'] = close.rolling(5).mean()
        df['sma_10'] = close.rolling(10).mean()
        df['sma_21'] = close.rolling(21).mean()
        df['volatility_10'] = close.rolling(10).std()
        df['volatility_21'] = close.rolling(21).std()

        # Price Action
        high_body = df[['Open', 'Close']].max(axis=1)
        low_body = df[['Open', 'Close']].min(axis=1)
        df['upper_shadow'] = df['High'] - high_body
        df['lower_shadow'] = low_body - df['Low']

        # Targets
        df['target_regression'] = df['Close'].shift(-1)
        df['target_classification'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Candle Shape
        df['body'] = (df['Close'] - df['Open']).abs()
        df['range'] = df['High'] - df['Low']

        df['is_doji'] = (df['body'] / df['range'] < 0.1).astype(int)
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < 0.1 * df['body'])).astype(int)
        df['is_inverted_hammer'] = ((df['upper_shadow'] > 2 * df['body']) & (df['lower_shadow'] < 0.1 * df['body'])).astype(int)

        # Momentum & ROC
        df['momentum_3'] = df['Close'] - df['Close'].shift(3)
        df['momentum_7'] = df['Close'] - df['Close'].shift(7)
        df['roc_3'] = df['Close'].pct_change(periods=3)
        df['roc_7'] = df['Close'].pct_change(periods=7)
        df['vol_5'] = df['Close'].rolling(5).std()
        df['vol_10'] = df['Close'].rolling(10).std()
        df['sma_5_slope'] = df['sma_5'].diff()

        # Engulfing Patterns
        df['prev_close'] = df['Close'].shift(1)
        df['prev_open'] = df['Open'].shift(1)
        df['is_bullish_engulfing'] = (
            (df['prev_close'] > df['prev_open']) &
            (df['Close'] > df['Open']) &
            (df['Open'] < df['prev_close']) &
            (df['Close'] > df['prev_open'])
        ).astype(int)
        df['is_bearish_engulfing'] = (
            (df['prev_open'] > df['prev_close']) &
            (df['Close'] < df['Open']) &
            (df['Open'] > df['prev_close']) &
            (df['Close'] < df['prev_open'])
        ).astype(int)

        # Trend Flags
        df['price_above_sma20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['price_above_sma50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['sma_crossover'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        df['log_return_positive'] = (df['Log_Return'] > 0).astype(int)

        df.drop(columns=['prev_open', 'prev_close'], inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df


# === Retrain Trigger ===
def should_retrain(mae_series, threshold=0.02, window=5):
    recent_mae = mae_series[-window:]
    return (recent_mae.mean() > threshold)

# === Prediction Band Calculator ===
def add_uncertainty_bands(preds, std_factor=1.0):
    preds = np.array(preds)
    std = np.std(preds)
    return preds - std * std_factor, preds + std * std_factor
