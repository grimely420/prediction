#!/bin/bash
# Stop all prediction system components

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.running_pids"

PATTERNS=(
    "-m shared.collector"
    "-m shared.api"
    "-m services.predictor_loop"
    "bitcoin/collector.py"
    "bitcoin/api.py"
    "bitcoin/generate_predictions.py"
    "bnb/collector.py"
    "bnb/api.py"
    "bnb/generate_predictions.py"
    "hype/collector.py"
    "shared/health_monitor.py"
)

echo "=========================================="
echo "Stopping Prediction System"
echo "=========================================="

if [ -f "$PID_FILE" ]; then
    echo "Stopping processes from PID file..."
    while IFS=: read -r pid name || [ -n "$pid" ]; do
        [[ "$pid" =~ ^# ]] && continue
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping PID $pid ($name)"
            kill "$pid" 2>/dev/null
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing PID $pid"
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done < "$PID_FILE"
    rm "$PID_FILE"
    echo ""
fi

for pattern in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Stopping orphaned $pattern processes: $pids"
        echo "$pids" | xargs kill 2>/dev/null
        sleep 1
        remaining=$(pgrep -f "$pattern" 2>/dev/null)
        if [ -n "$remaining" ]; then
            echo "Force killing remaining: $remaining"
            echo "$remaining" | xargs kill -9 2>/dev/null
        fi
    fi
done

echo ""
echo "=========================================="
echo "System stopped."
echo "=========================================="
