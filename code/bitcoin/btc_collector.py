#!/usr/bin/env python3
import requests
import sqlite3
import time
from datetime import datetime, timezone, timedelta
import os
import sys
import logging

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitcoin.config import DB_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BTC-Collector")

def get_price():
    """Get Bitcoin price with multiple API fallbacks and data validation."""
    price_sources = [
        {
            'name': 'Coinbase',
            'url': 'https://api.coinbase.com/v2/prices/BTC-USD/spot',
            'parser': lambda r: float(r.json()['data']['amount'])
        },
        {
            'name': 'Kraken',
            'url': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
            'parser': lambda r: float(r.json()['result']['XXBTZUSD']['c'][0])
        },
        {
            'name': 'Binance',
            'url': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
            'parser': lambda r: float(r.json()['price'])
        }
    ]
    
    for source in price_sources:
        try:
            r = requests.get(source['url'], timeout=10)
            if r.status_code == 200:
                price = source['parser'](r)
                # Validate price is reasonable
                if price and 1000 < price < 1000000:  # Sanity check for BTC price
                    logger.debug(f"Got price from {source['name']}: ${price:,.2f}")
                    return price
                else:
                    logger.warning(f"Invalid price from {source['name']}: ${price}")
        except Exception as e:
            logger.debug(f"{source['name']} API failed: {e}")
            continue
    
    logger.warning("All price sources failed")
    return None

def init_db():
    """Initialize database with optimized schema and indexes."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create prices table with optimized schema
    c.execute('''CREATE TABLE IF NOT EXISTS prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        price REAL,
        source TEXT DEFAULT 'collector'
    )''')
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_time TEXT,
        predicted_price REAL,
        actual_price REAL,
        error REAL,
        checked INTEGER DEFAULT 0,
        model_used TEXT,
        is_correct INTEGER
    )''')
    
    # Create indexes for better query performance
    c.execute('''CREATE INDEX IF NOT EXISTS idx_prices_timestamp 
                ON prices(timestamp DESC)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_time 
                ON predictions(prediction_time DESC)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_checked 
                ON predictions(checked)''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized with indexes")

def cleanup_old_data(days_to_keep=7):
    """Remove data older than specified days to optimize storage."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - 
                 timedelta(days=days_to_keep)).isoformat()
        
        # Count old records
        c.execute("SELECT COUNT(*) FROM prices WHERE timestamp < ?", (cutoff,))
        old_count = c.fetchone()[0]
        
        if old_count > 1000:  # Only cleanup if significant data
            c.execute("DELETE FROM prices WHERE timestamp < ?", (cutoff,))
            deleted = c.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted} old price records")
        
        conn.close()
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def main():
    """Enhanced main collector loop with error handling and monitoring."""
    init_db()
    
    count = 0
    consecutive_failures = 0
    max_consecutive_failures = 12  # 1 minute of failures
    
    logger.info("Bitcoin Collector Started")
    
    while True:
        try:
            price = get_price()
            
            if price:
                ts = datetime.now(timezone.utc).isoformat()
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO prices (timestamp, price, source) VALUES (?, ?, ?)",
                         (ts, price, 'collector'))
                conn.commit()
                conn.close()
                
                count += 1
                consecutive_failures = 0
                
                if count % 12 == 0:  # Log every minute
                    logger.info(f"BTC: ${price:,.2f} (total: {count})")
                    
                # Periodic cleanup (every 1000 records)
                if count % 1000 == 0:
                    cleanup_old_data()
                    
            else:
                consecutive_failures += 1
                logger.warning(f"Failed to get price (consecutive failures: {consecutive_failures})")
                
                # Exponential backoff if many consecutive failures
                if consecutive_failures > max_consecutive_failures:
                    sleep_time = min(60, 5 * (2 ** (consecutive_failures // max_consecutive_failures)))
                    logger.warning(f"Backing off for {sleep_time}s due to repeated failures")
                    time.sleep(sleep_time)
                    consecutive_failures = 0  # Reset after backoff
        
        except KeyboardInterrupt:
            logger.info("Shutting down collector")
            break
        except Exception as e:
            logger.error(f"Collector error: {e}")
            consecutive_failures += 1
        
        # Normal sleep between collection attempts
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped")
