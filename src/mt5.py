import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .forecast import forecast_next_7_days
from .features import compute_features

# === Initialize MT5 ===
def initialize_mt5(login, password, server, path=None):
    if not mt5.initialize(path=path, login=login, password=password, server=server):
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    print("✅ Connected to MT5")

# === Fetch Historical Data from MT5 ===
def fetch_mt5_data(symbol, days=45, timeframe=mt5.TIMEFRAME_D1):
    utc_from = datetime.now() - timedelta(days=days)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, days)
    if rates is None:
        raise ValueError(f"❌ Failed to fetch data for {symbol}")
    df = pd.DataFrame(rates)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'Volume'}, inplace=True)
    return df[['Date', 'open', 'high', 'low', 'close', 'Volume']].rename(
        columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
    )

# === Forecast + Auto-Trading Execution ===
def trade_from_forecast(symbol, model, feature_columns, login, password, server, volume=0.1):
    initialize_mt5(login, password, server)
    df = fetch_mt5_data(symbol)
    forecast = forecast_next_7_days(df, model, feature_columns)

    today_price = df['Close'].iloc[-1]
    tomorrow_forecast = forecast.iloc[0]['Predicted_Close']

    # Determine trade action
    if tomorrow_forecast > today_price:
        action = 'buy'
        order_type = mt5.ORDER_TYPE_BUY
    elif tomorrow_forecast < today_price:
        action = 'sell'
        order_type = mt5.ORDER_TYPE_SELL
    else:
        print("⏸ No clear signal.")
        return

    print(f"🔮 Forecast: {tomorrow_forecast:.5f} | Current: {today_price:.5f} → {action.upper()}")

    # Get trade price
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if action == 'buy' else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 20250707,
        "comment": "ForecastBot Auto-Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Execute trade
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade failed: {result.retcode}")
    else:
        print("✅ Trade executed successfully.")

    mt5.shutdown()
