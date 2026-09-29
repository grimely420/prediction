#!/usr/bin/env python3
"""
BNB prediction system configuration.
"""

import os

# Base directory - use relative path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DB_PATH = os.path.join(BASE_DIR, "bnb", "bnb_prices.db")
MODEL_DIR = os.path.join(BASE_DIR, "bnb", "models")
THRESHOLD_FILE = os.path.join(BASE_DIR, "bnb", ".threshold")

# API Settings
SYMBOL = "BNB"
PREDICTION_HORIZON = 5  # minutes (5 min for BNB vs 15 min for BTC)
COLLECTION_INTERVAL = 5  # seconds
LOOKBACK_MINUTES = 20

# Model Settings
MIN_TRAIN_POINTS = 800
RETRAIN_EVERY_N_POINTS = 5000

# API Server
API_HOST = "0.0.0.0"
API_PORT = 5002

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
