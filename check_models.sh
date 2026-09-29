#!/bin/bash
date; date -u
echo "==== PREDICTOR LOG TAIL (btc) ===="
journalctl -u prediction-predictor@btc.service --no-pager -n 12 2>/dev/null
echo "==== RETRAIN EVENTS SINCE BOOT ===="
journalctl -u "prediction-predictor@*" --since today --no-pager 2>/dev/null | grep -iE "trained|retrain" | tail -20
echo "==== MODEL FILE AGES ===="
python3 - <<'EOF'
import os, time
base = "/home/chain-deaction/local_prediction/predition-app/prediction"
now = time.time()
for coin, d in [("btc", "bitcoin"), ("bnb", "bnb"), ("hype", "hype")]:
    for f in ["kalshi_contract_latest.pkl", "xgb_5min_clf.pkl", "xgb_5min_latest.pkl"]:
        p = f"{base}/{d}/models/{f}"
        if os.path.exists(p):
            print(f"{coin} {f}: {(now - os.path.getmtime(p))/60:.0f} min ago")
EOF
