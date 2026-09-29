#!/bin/bash
# Start all prediction system components manually (no systemd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

mkdir -p "$LOG_DIR"

# Load optional environment overrides (e.g. CF Benchmarks credentials)
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a
    source "$SCRIPT_DIR/../.env"
    set +a
fi

echo "=========================================="
echo "Starting Prediction System"
echo "=========================================="

# Save PIDs to file for easy cleanup
PID_FILE="$SCRIPT_DIR/.running_pids"
echo "# Running PIDs $(date)" > "$PID_FILE"

start_component() {
    local name=$1
    local logfile=$2
    shift 2
    echo "Starting $name..."
    nohup "$VENV_PYTHON" "$@" > "$LOG_DIR/$logfile" 2>&1 &
    echo "  PID: $!"
    echo "$!:$name" >> "$PID_FILE"
    sleep 2
}

COINS="btc eth sol bnb xrp doge hype near zec"

echo ""
echo "[1/3] Starting data collectors..."
for coin in $COINS; do
    start_component "${coin^^} Collector" "${coin}-collector.log" -m shared.collector "$coin"
done

echo ""
echo "[2/3] Starting prediction APIs..."
start_component "Unified API (port 5000)" "api.log" -m shared.api

# Wait for API to be ready
echo ""
echo "  Waiting for API to initialize..."
sleep 5

echo ""
echo "[3/3] Starting predictor loops..."
for coin in $COINS; do
    start_component "${coin^^} Predictor Loop" "${coin}-predictor-loop.log" -m services.predictor_loop "$coin"
done

echo ""
echo "=========================================="
echo "All components started!"
echo "=========================================="
echo ""
echo "PIDs saved to: $PID_FILE"
echo "Logs directory: $LOG_DIR"
echo ""
echo "API Endpoint: http://localhost:5000"
echo ""
echo "To stop: ./stop_all.sh"
echo "To status: ./status.sh"
echo ""

