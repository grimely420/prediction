# Kalshi Crypto Prediction Terminal

Real-time prediction and decision system for **Kalshi 15-minute crypto markets** —
BTC (`KXBTC15M`), BNB (`KXBNB15M`), and HYPE (`KXHYPE15M`).

The system collects CF Benchmarks RTI prices every ~2–4s, trains strike-aware
contract models on a rolling 60-day window, scores every call against the actual
settlement TWAP, and serves a live dashboard with round charts, calibrated
YES/NO odds in cents, and entry-price caps.

## Architecture

```
predition-app/
├── prediction/
│   ├── shared/               # Core engine
│   │   ├── collector.py      # Price collector (CF Benchmarks RTI primary)
│   │   ├── data_store.py     # SQLite persistence (prices, predictions, signals)
│   │   ├── feature_engine.py # 54–88 features: EMA/MACD/BB/ADX/ATR, BTC lead-lag,
│   │   │                     #   Hyperliquid orderbook microstructure (HYPE)
│   │   ├── model_manager.py  # XGBoost + LightGBM ensembles, Platt calibration,
│   │   │                     #   contract-outcome classifier
│   │   ├── kalshi_engine.py  # Round tracking, Black-Scholes digital + contract
│   │   │                     #   model blend, settlement TWAP engine
│   │   ├── predictor_core.py # Live inference + contract signal logging
│   │   ├── news_engine.py    # Crypto news sentiment features
│   │   ├── api.py            # Flask API + dashboard
│   │   └── dashboard.py      # Live terminal UI (round charts, odds, decisions)
│   ├── services/
│   │   ├── predictor_loop.py # Predict (15s) / validate (30s) / retrain (30min)
│   │   ├── telegram_alerts.py
│   │   └── *.service         # systemd unit templates
│   ├── bitcoin/ bnb/ hype/   # Per-coin SQLite DBs + trained models
│   └── logs/
├── requirements.txt
└── .env.example              # Credential template (copy to .env)
```

### How it works

1. **Collectors** (`prediction-collector@<coin>`) poll the CF Benchmarks RTI
   (Kraken Futures source) every ~2s and store timestamped ticks.
2. **Predictor loops** (`prediction-predictor@<coin>`) run every 15s:
   multi-horizon regressors (5/10/15 min), binary direction classifiers, and a
   strike-aware contract model that answers the actual Kalshi question —
   *will the round settle above the round-open price?*
3. **Kalshi engine** blends a Black-Scholes digital option prior with the
   contract model's calibrated probability, applies EMA smoothing with a
   round-boundary reset, and in the final 75s switches to a settlement-TWAP
   estimator (Kalshi settles on a 60s TWAP).
4. **Scoring** logs one contract call per round per phase (entry/mid/late) and
   resolves it against the true final-60s average — win rates and calibration
   buckets are exposed at `/contract_stats/<coin>`.
5. **Retraining**: every ~30 min when new data arrives (rolling 60-day window,
   ~86k one-minute bars; 70/15/15 train/early-stop/calibration split).

## Requirements

- Linux (developed on Ubuntu), Python 3.10+
- ~4 GB RAM free during retrains, ~2 GB disk per coin per ~3 months of ticks

## Setup

```bash
git clone https://github.com/grimely420/predition-app.git
cd predition-app

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Credentials (`.env`)

```bash
cp .env.example .env
```

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `CF_API_USERNAME` / `CF_API_PASSWORD` | Optional | Authenticated CF Benchmarks API (public Kraken RTI fallback works without) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional | Telegram alert bot |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Optional | Alpaca price feed fallback |

**Never commit `.env`** — it is gitignored.

### systemd services

Unit templates live in `prediction/services/`. Install and enable:

```bash
sudo cp prediction/services/prediction-*.service /etc/systemd/system/
sudo systemctl daemon-reload

sudo systemctl enable --now \
  prediction-api.service \
  prediction-collector@btc.service \
  prediction-collector@bnb.service \
  prediction-collector@hype.service \
  prediction-hl-collector@hype.service \
  prediction-predictor@btc.service \
  prediction-predictor@bnb.service \
  prediction-predictor@hype.service \
  prediction-telegram-alerts.service
```

Adjust `WorkingDirectory`, `User`, and the venv path inside each unit to match
your install location.

## Verify

```bash
curl http://localhost:5000/health            # {"status":"ok"}
curl http://localhost:5000/kalshi            # all three coins, odds + round series
curl http://localhost:5000/contract_stats/btc  # scored win rates + calibration
```

Dashboard: open `http://<host>:5000/` — decision terminal with per-coin
round price-vs-strike charts, YES/NO cent odds, and entry caps.

## API reference

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live dashboard |
| `GET /kalshi` | Round analysis for all coins (odds, decision, chart ticks) |
| `GET /kalshi/<coin>` | Single-coin round analysis |
| `GET /feed/<coin>` | Compact bot feed (price + horizon predictions) |
| `GET /predict/<coin>[/<h>]` | 5/10/15-min predictions |
| `GET /contract_stats/<coin>` | Live-scored win rates, per-phase, calibration buckets |
| `GET /stats/<coin>` | Legacy regressor accuracy stats |
| `GET /news` | Crypto news sentiment summary |
| `GET /hl/<coin>` | Hyperliquid microstructure snapshot (HYPE) |

## Operational notes

- **Data**: ticks accumulate ~28k/day/coin at the ~2s cadence. SQLite handles
  this fine; prune `prices` rows older than ~90 days if disk gets tight —
  training only uses the last 60 days.
- **Honest expectations**: entry-window (min 0–6) edge is modest (~50–57%);
  mid/late-phase calls run 70–95%. Check `/contract_stats/<coin>` — the
  calibration table is the source of truth, not vibes.
- **Round alignment**: Kalshi 15m rounds open at :00/:15/:30/:45 UTC. All
  strikes, charts, and settlement tracking are keyed to those boundaries.
