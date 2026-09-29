#!/bin/bash
# Start all Bitcoin Prediction System services

set -e  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "=========================================="
echo "🚀 Starting Bitcoin Prediction System"
echo "=========================================="

# Define services (use bitcoin/ prefix to avoid killing BNB processes)
services=(
    "Collector:bitcoin/collector.py"
    "API:bitcoin/api.py"
    "Prediction Generator:bitcoin/generate_predictions.py"
    "Auto Retrain:bitcoin/auto_retrain.py"
)

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Stop any existing instances first (graceful)
echo "🔍 Checking for existing processes..."
for entry in "${services[@]}"; do
    name="${entry%:*}"
    pattern="${entry#*:}"
    if is_running "$pattern"; then
        echo "⚠️  $name already running – stopping it first"
        pkill -f "$pattern" 2>/dev/null
        sleep 1
    fi
done

echo ""
echo "📡 Starting services..."

# Start collector (silent)
pattern="bitcoin/collector.py"
if ! is_running "$pattern"; then
    $PYTHON bitcoin/collector.py > /tmp/btc_collector.log 2>&1 &
    pid=$!
    sleep 1
    if is_running "$pattern"; then
        echo "   ✅ Collector started (PID: $pid)"
    else
        echo "   ❌ Collector failed to start – check /tmp/btc_collector.log"
        exit 1
    fi
else
    echo "   ✅ Collector already running"
fi

# Start API
pattern="bitcoin/api.py"
if ! is_running "$pattern"; then
    $PYTHON bitcoin/api.py > /tmp/adv_api.log 2>&1 &
    pid=$!
    sleep 2  # give it time to bind port
    if is_running "$pattern"; then
        echo "   ✅ API started (PID: $pid, port: 5001)"
    else
        echo "   ❌ API failed to start – check /tmp/adv_api.log"
        exit 1
    fi
else
    echo "   ✅ API already running"
fi

# Start prediction generator
pattern="bitcoin/generate_predictions.py"
if ! is_running "$pattern"; then
    $PYTHON bitcoin/generate_predictions.py > /tmp/gen.log 2>&1 &
    pid=$!
    sleep 1
    if is_running "$pattern"; then
        echo "   ✅ Prediction Generator started (PID: $pid)"
    else
        echo "   ❌ Prediction Generator failed – check /tmp/gen.log"
        exit 1
    fi
else
    echo "   ✅ Prediction Generator already running"
fi

# Start auto retrain
pattern="bitcoin/auto_retrain.py"
if ! is_running "$pattern"; then
    $PYTHON bitcoin/auto_retrain.py > /tmp/retrain.log 2>&1 &
    pid=$!
    sleep 1
    if is_running "$pattern"; then
        echo "   ✅ Auto Retrain started (PID: $pid)"
    else
        echo "   ⚠️  Auto Retrain failed – will retry later (not critical)"
    fi
else
    echo "   ✅ Auto Retrain already running"
fi

echo ""
echo "=========================================="
echo "✅ All services started successfully"
echo "=========================================="
echo ""
echo "📊 Access the dashboard: http://localhost:5001/dashboard"
echo ""
echo "📝 Monitor logs:"
echo "   Collector:  tail -f /tmp/btc_collector.log"
echo "   API:        tail -f /tmp/adv_api.log"
echo "   Generator:  tail -f /tmp/gen.log"
echo ""
echo "🛑 To stop: ./stop.sh"
