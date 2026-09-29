#!/usr/bin/env python3
"""
HYPE prediction system configuration.
"""

import os

# Base directory - use relative path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DB_PATH = os.path.join(BASE_DIR, "hype", "hype_prices.db")
MODEL_DIR = os.path.join(BASE_DIR, "hype", "models")
THRESHOLD_FILE = os.path.join(BASE_DIR, "hype", ".threshold")

# API Settings
SYMBOL = "HYPE"
PREDICTION_HORIZON = 5  # minutes (primary horizon)
PREDICTION_HORIZONS = [5, 10, 15]  # all supported horizons
COLLECTION_INTERVAL = 6  # seconds
LOOKBACK_MINUTES = 20

# Model Settings
MIN_TRAIN_POINTS = 60
RETRAIN_EVERY_N_POINTS = 10000

# API Server
API_HOST = "0.0.0.0"
API_PORT = 5003

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
