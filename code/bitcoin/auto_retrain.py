#!/usr/bin/env python3
"""
Bitcoin auto-retrain daemon.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import setup_logging, get_db_connection
from bitcoin.config import DB_PATH, MODEL_DIR, RETRAIN_EVERY_N_POINTS, SYMBOL

# Setup logging
logger = setup_logging("BTC-AutoRetrain")

# Track last training
LAST_TRAIN_FILE = os.path.join(MODEL_DIR, ".last_train_count")


def get_price_count() -> int:
    """Get total number of price records."""
    try:
        conn = get_db_connection(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prices")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get price count: {e}")
        return 0


def should_retrain() -> tuple:
    """Check if retraining is needed."""
    current_count = get_price_count()
    
    if not os.path.exists(LAST_TRAIN_FILE):
        return True, current_count
    
    try:
        with open(LAST_TRAIN_FILE, 'r') as f:
            last_count = int(f.read().strip())
        return (current_count - last_count) >= RETRAIN_EVERY_N_POINTS, current_count
    except Exception:
        return True, current_count


def save_last_train_count(count: int) -> None:
    """Save last training count."""
    try:
        with open(LAST_TRAIN_FILE, 'w') as f:
            f.write(str(count))
    except Exception as e:
        logger.error(f"Failed to save last train count: {e}")


def run_retrain() -> bool:
    """Run the retraining script."""
    try:
        script_path = os.path.join(os.path.dirname(__file__), "model.py")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            logger.info("Retraining completed successfully")
            logger.debug(f"Output: {result.stdout[-500:]}")
            return True
        else:
            logger.error(f"Retraining failed: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Retraining timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return False


def main():
    """Main loop for auto-retrain daemon."""
    logger.info(f"{SYMBOL} Auto-Retrain Daemon Started")
    logger.info(f"Retrain every {RETRAIN_EVERY_N_POINTS} new price points")
    logger.info(f"Database: {DB_PATH}")
    
    running = True
    
    while running:
        try:
            need_retrain, new_count = should_retrain()
            
            if need_retrain:
                logger.info(f"Triggering retrain (current points: {new_count})")
                
                if run_retrain():
                    save_last_train_count(new_count)
                    logger.info("Retrain completed and saved")
                else:
                    logger.warning("Retrain failed, will retry later")
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            running = False
        except Exception as e:
            logger.error(f"Auto-retrain loop error: {e}")
            time.sleep(60)
    
    logger.info("Auto-retrain daemon stopped")


if __name__ == "__main__":
    main()
