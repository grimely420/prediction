#!/usr/bin/env python3
"""
Bitcoin prediction generator - calls API every 15 minutes.
"""

import os
import sys
import time
import requests
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import setup_logging
from bitcoin.config import API_HOST, API_PORT, PREDICTION_HORIZON, SYMBOL

# Setup logging
logger = setup_logging("BTC-Generator")

API_URL = f"http://{API_HOST}:{API_PORT}/predict"
RUNNING = True


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global RUNNING
    logger.info("Received shutdown signal")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def make_prediction() -> bool:
    """Make a prediction by calling the API."""
    try:
        response = requests.get(API_URL, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                logger.info(
                    f"Prediction: ${data['predicted_price_15min']:,.2f} "
                    f"(Current: ${data['current_price']:,.2f}, "
                    f"Change: {data['change_percent']:+.2f}%, "
                    f"Model: {data.get('model_used', 'ensemble')})"
                )
                return True
            else:
                logger.warning(f"API error: {data.get('error', 'Unknown')}")
                return False
        else:
            logger.warning(f"HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API at {API_URL}")
        return False
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return False


def main():
    """Main prediction generation loop."""
    logger.info(f"{SYMBOL} Prediction Generator Started")
    logger.info(f"API URL: {API_URL}")
    logger.info(f"Prediction interval: {PREDICTION_HORIZON} minutes")
    
    prediction_count = 0
    wait_seconds = PREDICTION_HORIZON * 60
    
    while RUNNING:
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"[{current_time}] Making prediction #{prediction_count + 1}...")
            
            if make_prediction():
                prediction_count += 1
            
            # Wait for next prediction
            for _ in range(wait_seconds):
                if not RUNNING:
                    break
                time.sleep(1)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)
    
    logger.info(f"Prediction generator stopped. Total predictions: {prediction_count}")


if __name__ == "__main__":
    main()
