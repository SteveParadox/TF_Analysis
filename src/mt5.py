import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .forecast import forecast_next_7_days
from .features import compute_features
from flask import current_app

# === 1. Initialize MT5 ===
def initialize_mt5(login, password, server, path=None):
    # Initialize MT5 only with path if explicitly provided
    if path:
        success = mt5.initialize(path=path, login=login, password=password, server=server)
    else:
        success = mt5.initialize(login=login, password=password, server=server)

    if not success:
        error_code, description = mt5.last_error()
        raise RuntimeError(f"MT5 initialization failed ({error_code}): {description}")

    current_app.logger.info("Connected to MetaTrader 5 successfully.")

# === 2. Fetch Historical Market Data ===
def fetch_mt5_data(symbol, days=45, timeframe=mt5.TIMEFRAME_D1):
    utc_from = datetime.now() - timedelta(days=days)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, days)
    if rates is None or len(rates) == 0:
        raise ValueError(f"Failed to fetch MT5 historical data for {symbol}")
    
    df = pd.DataFrame(rates)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'Volume'}, inplace=True)
    
    return df[['Date', 'open', 'high', 'low', 'close', 'Volume']].rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
    })

# === 3. Forecast and Trade ===
def trade_from_forecast(symbol, model, feature_columns, login, password, server, volume=0.1, df=None):
    try:
        initialize_mt5(login, password, server)

        if df is None:
            df = fetch_mt5_data(symbol)
            df = compute_features(df)
            df.ffill(inplace=True)
            df.fillna(0, inplace=True)

        forecast, _ = forecast_next_7_days(
            df=df,
            mae_history=[],
            model=model,
            retrain_callback=lambda: retrain_model(
                data_path=None,
                model_save_path="models/meta_model_rf.pkl"
            )
        )

        today_price = df['Close'].iloc[-1]
        tomorrow_price = forecast.iloc[0]['Predicted_Close']
        tp_price = forecast.iloc[1]['Predicted_Close']  # Set TP to second-day forecast


        current_app.logger.info(f"{symbol} — Current: {today_price:.5f}, Forecast: {tomorrow_price:.5f}")

        if tomorrow_price > today_price:
            order_type = mt5.ORDER_TYPE_BUY
            action = 'BUY'
        elif tomorrow_price < today_price:
            order_type = mt5.ORDER_TYPE_SELL
            action = 'SELL'
        else:
            current_app.logger.info("No trade signal. Forecast equal to current price.")
            return

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol {symbol} not found")

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Failed to select symbol {symbol}")

        # Get latest tick price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            raise RuntimeError(f"Failed to retrieve latest tick for {symbol}")
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        # Prepare a test request to validate filling mode
        test_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 20250707,
            "comment": "ForecastBot AutoTrade - Test",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        filling_mode = None
        for mode in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
            test_request["type_filling"] = mode
            check = mt5.order_check(test_request)
            if check and check.retcode not in (mt5.TRADE_RETCODE_INVALID, 10030):
                filling_mode = mode
                break

        # Fallback to symbol default filling_mode
        if filling_mode is None:
            if hasattr(symbol_info, "filling_mode"):
                filling_mode = symbol_info.filling_mode
                current_app.logger.warning(
                    f"No approved fill mode from order_check; using symbol default: {filling_mode}"
                )
            else:
                raise RuntimeError(f"No supported filling mode accepted by order_check or available for {symbol}")

        # Final order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 20250707,
            "comment": "ForecastBot AutoTrade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode == 10027:
                current_app.logger.error(
                    "Trade failed: AutoTrading (Algo Trading) is disabled in MetaTrader 5. "
                    "Please enable it by clicking the 'Algo Trading' button on the MT5 terminal."
                )
            else:
                current_app.logger.error(
                    f"Trade failed (retcode={result.retcode}): {result._asdict()}"
                )
        else:
            current_app.logger.info(f"Trade executed: {action} {symbol} at {price:.5f}, TP set at {tp_price:.5f}")

    except Exception as ex:
        current_app.logger.exception(f"Trade execution error: {ex}")
    finally:
        mt5.shutdown()
