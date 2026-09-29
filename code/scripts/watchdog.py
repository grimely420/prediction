#!/usr/bin/env python3
"""Prediction-system watchdog.

Runs every 5 min via systemd timer (as root so it can restart units).
Detects *hung* services — ones that are 'active' but no longer producing
output — which Restart=always cannot catch.

Checks:
  - newest price tick per coin < 180s old  -> restart prediction-collector@<coin>
  - newest news_features row < 300s old     -> restart prediction-predictor@<coin>
  - GET /health responds 200                -> restart prediction-api

A 15-min cooldown per unit prevents restart loops while a coin is
mid-retrain or the host is under load.
"""

import glob
import json
import os
import sqlite3
import subprocess
import urllib.request
from datetime import datetime, timezone

APP = '/home/chain-deaction/local_prediction/predition-app/prediction'
STATE_FILE = '/var/lib/prediction-system/watchdog_state.json'
TICK_MAX_AGE_S = 180        # collector writes every ~3s; 3 min = hung
PRED_MAX_AGE_S = 300        # predictor logs news every ~15s; 5 min = hung
RESTART_COOLDOWN_S = 900

DIR_TO_COIN = {'bitcoin': 'btc'}


def log(msg):
    print(f"[watchdog] {msg}", flush=True)


def parse_ts(ts):
    return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))


def age_seconds(ts):
    try:
        return (datetime.now(timezone.utc) - parse_ts(ts)).total_seconds()
    except Exception:
        return 1e9


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def restart(unit, state, now_ts):
    last = state.get(unit, 0)
    if now_ts - last < RESTART_COOLDOWN_S:
        log(f"{unit} stale but in cooldown "
            f"({int(now_ts - last)}s ago) - skipping")
        return
    status = subprocess.run(
        ['systemctl', 'is-active', unit],
        capture_output=True, text=True).stdout.strip()
    if status != 'active':
        log(f"{unit} is {status} - not restarting (restart loop handled "
            f"by systemd)")
        return
    log(f"restarting {unit}")
    subprocess.run(['systemctl', 'restart', unit])
    state[unit] = now_ts


def main():
    state = load_state()
    now_ts = datetime.now(timezone.utc).timestamp()

    for db in glob.glob(os.path.join(APP, '*', '*_prices.db')):
        coin = DIR_TO_COIN.get(os.path.basename(os.path.dirname(db)),
                               os.path.basename(os.path.dirname(db)))
        try:
            con = sqlite3.connect(db, timeout=10)
            tick = con.execute(
                "SELECT timestamp FROM prices ORDER BY id DESC LIMIT 1"
            ).fetchone()
            news = con.execute(
                "SELECT timestamp FROM news_features ORDER BY id DESC "
                "LIMIT 1").fetchone()
            con.close()
        except Exception as e:
            log(f"{coin}: db read failed: {e}")
            continue

        if tick and age_seconds(tick[0]) > TICK_MAX_AGE_S:
            log(f"{coin}: last tick {age_seconds(tick[0]):.0f}s old")
            restart(f'prediction-collector@{coin}.service', state, now_ts)
        if news and age_seconds(news[0]) > PRED_MAX_AGE_S:
            log(f"{coin}: last news log {age_seconds(news[0]):.0f}s old")
            restart(f'prediction-predictor@{coin}.service', state, now_ts)

    try:
        r = urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=10)
        if r.status != 200:
            raise RuntimeError(f"health status {r.status}")
    except Exception as e:
        log(f"api unhealthy: {e}")
        restart('prediction-api.service', state, now_ts)

    save_state(state)
    log("sweep complete")


if __name__ == '__main__':
    main()
