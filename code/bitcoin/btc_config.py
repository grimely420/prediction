#!/usr/bin/env python3
"""
Bitcoin prediction system configuration.
"""

import os

# Base directory - use relative path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DB_PATH = os.path.join(BASE_DIR, "bitcoin", "bitcoin_prices.db")
MODEL_DIR = os.path.join(BASE_DIR, "bitcoin", "models")
THRESHOLD_FILE = os.path.join(BASE_DIR, "bitcoin", ".threshold")

# API Settings
SYMBOL = "BTC"
PREDICTION_HORIZON = 15  # minutes
COLLECTION_INTERVAL = 5  # seconds
LOOKBACK_MINUTES = 30

# Model Settings
MIN_TRAIN_POINTS = 1800
RETRAIN_EVERY_N_POINTS = 10000

# API Server
API_HOST = "0.0.0.0"
API_PORT = 5001

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
