#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "Starting HYPE Prediction System (port 5003)..."

pkill -f "hype/hype_collector.py" 2>/dev/null
pkill -f "hype/hype_api.py" 2>/dev/null
pkill -f "hype/hype_generate_predictions.py" 2>/dev/null
pkill -f "hype/hype_auto_retrain.py" 2>/dev/null

$PYTHON hype/hype_collector.py > /tmp/hype_collector.log 2>&1 &
echo "Collector started (PID: $!)"

$PYTHON hype/hype_api.py > /tmp/hype_api.log 2>&1 &
echo "API started (PID: $!)"

sleep 3
$PYTHON hype/hype_generate_predictions.py > /tmp/hype_gen.log 2>&1 &
echo "Generator started (PID: $!)"

$PYTHON hype/hype_auto_retrain.py > /tmp/hype_retrain.log 2>&1 &
echo "Auto-retrain started (PID: $!)"

echo "HYPE system running. Dashboard: http://localhost:5003"
