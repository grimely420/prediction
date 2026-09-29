#!/bin/bash
echo "Stopping BNB Prediction System..."
pkill -f "bnb/collector.py"
pkill -f "bnb/api.py"
pkill -f "bnb/generate_predictions.py"
pkill -f "bnb/auto_retrain.py"
echo "✅ All stopped."
