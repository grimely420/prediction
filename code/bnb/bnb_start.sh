#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "Starting BNB Prediction System (port 5002)..."

pkill -f "bnb/collector.py" 2>/dev/null
pkill -f "bnb/api.py" 2>/dev/null
pkill -f "bnb/generate_predictions.py" 2>/dev/null
pkill -f "bnb/auto_retrain.py" 2>/dev/null

$PYTHON bnb/collector.py > /tmp/bnb_collector.log 2>&1 &
echo "Collector started (PID: $!)"

$PYTHON bnb/api.py > /tmp/bnb_api.log 2>&1 &
echo "API started (PID: $!)"

sleep 3
$PYTHON bnb/generate_predictions.py > /tmp/bnb_gen.log 2>&1 &
echo "Generator started (PID: $!)"

$PYTHON bnb/auto_retrain.py > /tmp/bnb_retrain.log 2>&1 &
echo "Auto-retrain started (PID: $!)"

echo "✅ BNB system running. Dashboard: http://localhost:5002/dashboard"
