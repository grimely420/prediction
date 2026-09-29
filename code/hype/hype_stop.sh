#!/bin/bash
echo "Stopping HYPE Prediction System..."
pkill -f "hype/hype_collector.py"
pkill -f "hype/hype_api.py"
pkill -f "hype/hype_generate_predictions.py"
pkill -f "hype/hype_auto_retrain.py"
echo "All stopped."
