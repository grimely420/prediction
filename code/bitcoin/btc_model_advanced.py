#!/usr/bin/env python3
"""
Bitcoin advanced model training with ensemble and regime detection.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.advanced_trainer import run_advanced_training
from bitcoin.config import DB_PATH, MODEL_DIR, PREDICTION_HORIZON

def main():
    print("="*60)
    print("Bitcoin Advanced Model Training")
    print("="*60)
    print(f"Database: {DB_PATH}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Prediction Horizon: {PREDICTION_HORIZON} minutes")
    print("="*60)
    
    success = run_advanced_training('BTC', DB_PATH, MODEL_DIR, PREDICTION_HORIZON)
    
    if success:
        print("\n✓ Advanced training completed successfully!")
        print("Models saved with ensemble learning and regime detection.")
    else:
        print("\n✗ Training failed or insufficient data.")
        print("Make sure you have at least 1800 data points collected.")

if __name__ == "__main__":
    main()
