# === Feature Engineering Functions (including Indicators) ===

import numpy as np
import pandas as pd

# === Relative Strength Index (RSI) ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === Moving Average Convergence Divergence (MACD) ===
def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

# === Bollinger Bands ===
def compute_bbands(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = upper - lower
    return upper, lower, width

# === Retrain Trigger ===
def should_retrain(mae_series, threshold=0.02, window=5):
    recent_mae = mae_series[-window:]
    return (recent_mae.mean() > threshold)

# === Prediction Band Calculator ===
def add_uncertainty_bands(preds, std_factor=1.0):
    preds = np.array(preds)
    std = np.std(preds)
    return preds - std * std_factor, preds + std * std_factor

# === Full Feature Set ===
def compute_features(df):
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['is_week_start'] = (df['dayofweek'] == 0).astype(int)
    df['is_week_end'] = (df['dayofweek'] == 4).astype(int)

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_1d'] = df['Close'].pct_change()
    df['log_return_positive'] = (df['Log_Return'] > 0).astype(int)

    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['lag_3'] = df['Close'].shift(3)

    df['Sma_5'] = df['Close'].rolling(window=5).mean()
    df['Sma_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()

    df['volatility_10'] = df['Log_Return'].rolling(window=10).std()
    df['volatility_21'] = df['Log_Return'].rolling(window=21).std()

    df['RSI'] = compute_rsi(df['Close'])

    macd, macd_signal, macd_diff = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_diff'] = macd_diff

    bb_upper, bb_lower, bb_width = compute_bbands(df['Close'])
    df['BB_upper'] = bb_upper
    df['BB_lower'] = bb_lower
    df['BB_width'] = bb_width

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['momentum_3'] = df['Close'] - df['Close'].shift(3)
    df['momentum_7'] = df['Close'] - df['Close'].shift(7)
    df['roc_3'] = df['Close'].pct_change(periods=3)
    df['roc_7'] = df['Close'].pct_change(periods=7)

    df['vol_5'] = df['Close'].rolling(window=5).std()
    df['vol_10'] = df['Close'].rolling(window=10).std()

    df['sma_5_slope'] = df['sma_5'].diff()

    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body'] = abs(df['Close'] - df['Open'])
    df['range'] = df['High'] - df['Low']

    df['is_doji'] = (df['body'] / df['range'] < 0.1).astype(int)
    df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < df['body'])).astype(int)
    df['is_inverted_hammer'] = ((df['upper_shadow'] > 2 * df['body']) & (df['lower_shadow'] < df['body'])).astype(int)

    df['is_bullish_engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] > df['Open']) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    ).astype(int)

    df['is_bearish_engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open']) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    ).astype(int)

    df['price_above_sma20'] = (df['Close'] > df['SMA_20']).astype(int)
    df['price_above_sma50'] = (df['Close'] > df['SMA_50']).astype(int)
    df['sma_crossover'] = (df['sma_5'] > df['sma_10']).astype(int)

    print("🧼 Feature null check:\n", df.isna().sum().sort_values(ascending=False).head(15))
    return df
