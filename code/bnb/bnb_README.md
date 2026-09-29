# BNB Price Prediction System

- Predicts BNB/USDT price 5 minutes ahead.
- Collects price every 5 seconds.
- Trains XGBoost model on 20,000 points (~27.8h), then retrains every 5,000 new points.
- Ensemble of XGBoost, ARIMA, trend extrapolation.
- Modern customizable dashboard with theme, price colors.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_bnb.txt
