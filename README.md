# TF_Analysis
# Model	Strengths
# Prophet	Captures seasonality, trend, and holidays
# LSTM	Great for sequence memory, time-dependencies, and nonlinear dynamics
# XGBoost	Captures sharp non-linear patterns and interactions
# Random Forest (meta)	Learns how to weigh the others, adds stability and variance control

### Roadmap to Production Deployment
📅 Week 1 — Model Refinement
Objective: Reduce lag, sharpen responsiveness, boost predictive alpha.

🔧 Tasks:
 Sharpen Spikes:

Add rolling_max, rolling_min, rolling_std (5, 10 days)

Tune max_depth, min_child_weight, gamma in XGBoost

 Lag Reduction:

Add forward indicators: price slope, MACD slope, SMA crossover angle

Use momentum or breakout features (e.g., price > last 10 high?)

 Retrain and re-evaluate Prophet + LSTM + XGBoost

 Check cross-validation over multiple time slices (TimeSeriesSplit)

 #### 2
 📅 Week 2 — Meta Model & Robustness
Objective: Improve stacker logic and ensure model survives edge cases.

🔧 Tasks:
 Replace Random Forest with:

Option A: LightGBM (faster, better leaf-wise splits)

Option B: Shallow Neural Net (e.g., MLPRegressor with Dropout)

 Add input: model disagreement width (max(preds) - min(preds))

 Backtest on:

Calm markets (flat)

Trending spikes

Reversal periods

 Evaluate drift: rolling MAE vs. baseline MAE

 #### 3
📅 Week 3 — External Feature Integration
Objective: Improve foresight by adding non-price signals.

🔧 Tasks:
 Add:

EURUSD as an external input to EURGBP

FinBERT sentiment score for related keywords (optional)

Economic calendar events (e.g., ECB, CPI days as binary flags)

 Retrain all models with these new signals

 Run a 2-week walk-forward test and compare against older version

#### 4
📅 Week 4 — Productionization & Serving
Objective: Wrap, schedule, serve, and store.

🔧 Tasks:
 Package pipeline with:

joblib or ONNX model exports

config.yaml for tunables

 Build predict.py and serve.py scripts:

predict.py → fetch data, make prediction, store to DB/CSV

serve.py → Flask/FastAPI endpoint for predictions

 Add logs + daily retrain scheduling with cron or Airflow

 (Optional) Push predictions to a web dashboard or Telegram bot

 
