#!/bin/bash
# Stop all Bitcoin Prediction System services

echo "=========================================="
echo "🛑 Stopping Bitcoin Prediction System"
echo "=========================================="

# Define services (use bitcoin/ prefix to avoid killing BNB)
declare -A services=(
    ["Collector"]="bitcoin/collector.py"
    ["API"]="bitcoin/api.py"
    ["Prediction Generator"]="bitcoin/generate_predictions.py"
    ["Auto Retrain"]="bitcoin/auto_retrain.py"
)

# Function to stop a service gracefully then force if needed
stop_service() {
    local name="$1"
    local pattern="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -n "$pids" ]; then
        echo "🔍 $name – PIDs: $(echo $pids | tr '\n' ' ')"
        echo "   Sending SIGTERM..."
        pkill -f "$pattern" 2>/dev/null
        sleep 2
        
        # Check if still alive
        local still_running=$(pgrep -f "$pattern" 2>/dev/null)
        if [ -n "$still_running" ]; then
            echo "   ⚠️  $name still running – force killing..."
            pkill -9 -f "$pattern" 2>/dev/null
            sleep 1
        fi
        
        # Verify stopped
        local final_check=$(pgrep -f "$pattern" 2>/dev/null)
        if [ -z "$final_check" ]; then
            echo "   ✅ $name stopped"
        else
            echo "   ❌ Failed to stop $name – manual intervention may be needed"
        fi
    else
        echo "⚠️  $name not running"
    fi
}

# Stop each service
for name in "${!services[@]}"; do
    stop_service "$name" "${services[$name]}"
done

# Clean up lock files
echo ""
echo "🧹 Cleaning up temporary files..."
rm -f /tmp/prediction_generator.lock 2>/dev/null
echo "   ✅ Lock files removed"

# Double-check for any stray Python processes (optional)
echo ""
echo "🔍 Final verification..."
remaining=$(pgrep -f "bitcoin/collector.py|bitcoin/api.py|bitcoin/generate_predictions.py|bitcoin/auto_retrain.py" 2>/dev/null)
if [ -n "$remaining" ]; then
    echo "⚠️  Unexpected remaining processes: $remaining"
    echo "   Force killing..."
    pkill -9 -f "bitcoin/collector.py|bitcoin/api.py|bitcoin/generate_predictions.py|bitcoin/auto_retrain.py" 2>/dev/null
else
    echo "✅ No stray processes found"
fi

echo ""
echo "=========================================="
echo "✅ All services stopped"
echo "=========================================="
