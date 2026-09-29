#!/usr/bin/env python3
"""
Unified launcher for the multi-horizon prediction system.

Starts the shared Flask API, one price collector per coin, and one
predictor loop per coin. Each component is started as a child process
and logs to logs/<component>.log. Stop with Ctrl+C.

Supported coins and horizons:
  - BTC, BNB, HYPE
  - 5, 10, 15 minute predictions
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

PREDICTION_DIR = Path(__file__).resolve().parent
BASE_DIR = PREDICTION_DIR.parent
LOG_DIR = BASE_DIR / "logs"
PID_FILE = BASE_DIR / ".running_pids"

COINS = ["btc", "eth", "sol", "bnb", "xrp", "doge", "hype", "near", "zec"]

VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)

LOG_DIR.mkdir(parents=True, exist_ok=True)

processes = []


def start_process(name, cmd, logfile):
    log_path = LOG_DIR / logfile
    print(f"Starting {name}... {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PREDICTION_DIR),
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    processes.append({"name": name, "proc": proc, "cmd": cmd})
    print(f"  {name} PID: {proc.pid}")
    time.sleep(1)
    return proc


def write_pids():
    try:
        with open(PID_FILE, "w") as f:
            f.write(f"# Running PIDs {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for item in processes:
                f.write(f"{item['proc'].pid}:{item['name']}\n")
    except Exception as e:
        print(f"Could not write PID file: {e}")


def cleanup(signum=None, frame=None):
    print("\nShutting down prediction system...")
    for item in processes:
        try:
            item["proc"].terminate()
        except Exception:
            pass

    for _ in range(10):
        if all(item["proc"].poll() is not None for item in processes):
            break
        time.sleep(0.5)

    for item in processes:
        if item["proc"].poll() is None:
            try:
                item["proc"].kill()
            except Exception:
                pass

    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass

    print("All components stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def main():
    print("=" * 60)
    print("STARTING MULTI-HORIZON PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Python: {PYTHON}")
    print(f"Coins: {', '.join(c.upper() for c in COINS)}")
    print("Horizons: 5, 10, 15 minutes")
    print("=" * 60)

    print("\n[1/4] Starting unified prediction API...")
    start_process("Unified API", [PYTHON, "-m", "shared.api"], "api.log")

    print("\n[2/4] Starting price collectors...")
    for coin in COINS:
        start_process(
            f"{coin.upper()} Collector",
            [PYTHON, "-m", "shared.collector", coin],
            f"{coin}-collector.log",
        )

    print("\n[3/4] Starting predictor loops...")
    for coin in COINS:
        start_process(
            f"{coin.upper()} Predictor Loop",
            [PYTHON, "-m", "services.predictor_loop", coin],
            f"{coin}-predictor.log",
        )

    write_pids()

    print("\n" + "=" * 60)
    print("SYSTEM STARTED SUCCESSFULLY")
    print("=" * 60)
    print("API: http://localhost:5000")
    print("Prediction endpoints:")
    for coin in COINS:
        print(f"  /predict/{coin}")
        print(f"  /predict/{coin}/<5|10|15>")
    print(f"Validation: POST /validate/<coin>")
    print(f"Stats:      GET /stats/<coin>")
    print(f"Health:     GET /health")
    print(f"Logs:       {LOG_DIR}")
    print("Stop with Ctrl+C")
    print("=" * 60)

    while True:
        try:
            for item in processes:
                proc = item["proc"]
                if proc.poll() is not None:
                    print(
                        f"WARNING: {item['name']} (PID {proc.pid}) exited with code {proc.returncode}"
                    )
            time.sleep(5)
        except KeyboardInterrupt:
            cleanup()
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
