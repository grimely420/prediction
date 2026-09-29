#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

echo "=========================================="
echo "Prediction System Status"
echo "=========================================="
echo ""
echo "Processes:"
ps aux | grep -E "shared.collector|shared.api|predictor_loop" | grep -v grep | awk '{print "  " $11 " " $12 " " $13 " " $14}'
echo ""
for coin in btc bnb hype; do
    count=$("$VENV_PYTHON" -c "from shared.coin_config import get_coin_config; cfg=get_coin_config('$coin'); import sqlite3; print(sqlite3.connect(cfg.db_path).cursor().execute('SELECT COUNT(*) FROM prices').fetchone()[0])" 2>/dev/null)
    echo "${coin^^} Count: ${count:-0}"
done
echo ""
echo "API Health: $(curl -s http://localhost:5000/health | "$VENV_PYTHON" -c "import sys,json; print(json.load(sys.stdin).get('status', 'offline'))" 2>/dev/null)"
echo ""
echo "Endpoints:"
echo "  http://localhost:5000/coins"
echo "  http://localhost:5000/predict/<coin>/<horizon>"
echo "  http://localhost:5000/predict/<coin>"
echo "  http://localhost:5000/stats/<coin>"
