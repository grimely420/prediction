#!/usr/bin/env python3
"""
Watchdog / health checker for the Bitcoin Prediction System.

Run by cron every few minutes. It checks:
  1. All 4 services are running (collector, api, generate_predictions, auto_retrain).
  2. The price feed is FRESH (a tick within the last STALE_PRICE_SEC seconds).
  3. Predictions are still being generated (one within last STALE_PRED_MIN minutes).
  4. No CORRUPT data: a run of >= DUP_RUN identical actual_price values
     among recent validated predictions (the bug we just fixed).

It auto-restarts any dead service via start.sh, and prints a structured
report. Exit code 0 = all OK, 1 = problems found (and acted on).

Intended to be invoked by the OpenClaw admin agent; the report text is what
the agent relays to Mark only when something is wrong.
"""
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "bitcoin_prices.db")
START_SH = os.path.join(SCRIPT_DIR, "start.sh")

SERVICES = {
    "collector": "collector.py",
    "api": "api.py",
    "generator": "generate_predictions.py",
    "auto_retrain": "auto_retrain.py",
}

STALE_PRICE_SEC = 120      # feed should tick ~every 5s; 2 min = dead
STALE_PRED_MIN = 25        # predictions ~every 15 min; 25 min = stalled
DUP_RUN = 5                # >=5 identical consecutive actual_price = corrupt


def is_running(pattern):
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        )
        # Filter out this healthcheck process itself
        pids = [p for p in out.stdout.split() if p]
        return len(pids) > 0
    except Exception:
        return False


def check_services():
    dead = [name for name, pat in SERVICES.items() if not is_running(pat)]
    return dead


def restart_services():
    try:
        subprocess.run(["bash", START_SH], capture_output=True, text=True, timeout=60)
        return True
    except Exception as e:
        return f"restart failed: {e}"


def check_price_freshness():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT timestamp FROM prices ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        return False, None, "no price rows"
    last = datetime.fromisoformat(row[0])
    age = (datetime.now(timezone.utc) - last).total_seconds()
    return age <= STALE_PRICE_SEC, age, row[0]


def check_prediction_freshness():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT prediction_time FROM predictions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        return False, None, "no predictions"
    last = datetime.fromisoformat(row[0])
    age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
    return age_min <= STALE_PRED_MIN, age_min, row[0]


def check_corrupt_data():
    """Detect a run of identical actual_price among the most recent validated
    predictions — the signature of the frozen-feed poisoning bug."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT id, actual_price FROM predictions
           WHERE checked=1 AND actual_price IS NOT NULL
           ORDER BY id DESC LIMIT 60"""
    ).fetchall()
    conn.close()
    if not rows:
        return True, None
    run = 1
    worst = 1
    worst_val = None
    for i in range(1, len(rows)):
        if rows[i][1] == rows[i - 1][1]:
            run += 1
            if run > worst:
                worst = run
                worst_val = rows[i][1]
        else:
            run = 1
    ok = worst < DUP_RUN
    detail = None if ok else f"{worst} identical actual_price={worst_val}"
    return ok, detail


def main():
    now = datetime.now(timezone.utc).isoformat()
    problems = []
    actions = []

    dead = check_services()
    if dead:
        problems.append(f"DEAD services: {', '.join(dead)}")
        res = restart_services()
        actions.append(f"ran start.sh to restart ({res})")
        # re-check
        still = check_services()
        if still:
            problems.append(f"STILL dead after restart: {', '.join(still)}")
        else:
            actions.append("all services back up after restart")

    fresh_p, age_s, p_ts = check_price_freshness()
    if not fresh_p:
        problems.append(f"PRICE FEED STALE: last tick {p_ts} (age {age_s:.0f}s)")
        if "collector" not in dead:
            actions.append("restarting collector (feed stale but process alive)")
            subprocess.run(["pkill", "-f", "collector.py"], capture_output=True)
            subprocess.run(["bash", START_SH], capture_output=True, timeout=60)

    fresh_pred, age_m, pred_ts = check_prediction_freshness()
    if not fresh_pred:
        problems.append(f"PREDICTIONS STALLED: last {pred_ts} (age {age_m:.0f}m)")

    ok_corrupt, detail = check_corrupt_data()
    if not ok_corrupt:
        problems.append(f"CORRUPT DATA: {detail}")

    status = "OK" if not problems else "PROBLEM"
    print(f"[{now}] HEALTH={status}")
    for p in problems:
        print(f"  ! {p}")
    for a in actions:
        print(f"  -> {a}")
    if not problems:
        print("  collector, api, generator, auto_retrain all running; feed + predictions fresh; no corrupt runs.")

    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
