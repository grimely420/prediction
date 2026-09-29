#!/usr/bin/env python3
"""
HYPE feature engineering - uses ALL available data properly.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, timedelta

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging
from hype.config import DB_PATH

# Setup logging
logger = setup_logging("HYPE-Engine")


def load_price_data(limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load ALL price data from database without losing points.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Load all data ordered by timestamp
        query = "SELECT timestamp, price FROM prices ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("No price data found")
            return None
        
        # Convert to datetime and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Sort index
        df = df.sort_index()
        
        # Apply limit if specified (from the end)
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        logger.info(f"Loaded {len(df)} price points from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load price data: {e}")
        return None


def compute_features(target_timeframes: List[int] = [1, 5], 
                     future_minutes: int = 5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Compute features using ALL available data efficiently.
    """
    try:
        # Load all data
        df_raw = load_price_data()
        if df_raw is None or len(df_raw) < 60:
            logger.warning(f"Insufficient data: {len(df_raw) if df_raw is not None else 0} points")
            return None, None, None
        
        logger.info(f"Processing {len(df_raw)} raw price points")
        
        # Resample to 1-minute OHLC (this aggregates but preserves info)
        df_1min = df_raw.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last']
        })
        df_1min.columns = ['open', 'high', 'low', 'close']
        df_1min = df_1min.dropna()
        
        logger.info(f"Resampled to {len(df_1min)} 1-minute bars")
        
        # Use closing price for calculations
        close = df_1min['close']
        
        # Create feature DataFrame
        features = pd.DataFrame(index=df_1min.index)
        features['close'] = close
        
        # Price returns for multiple windows
        for window in [1, 2, 3, 5, 10]:
            features[f'ret_{window}m'] = close.pct_change(window)
        
        # Volatility (rolling std of 1-min returns)
        returns_1m = close.pct_change()
        features['volatility_5m'] = returns_1m.rolling(window=5, min_periods=2).std()
        features['volatility_10m'] = returns_1m.rolling(window=10, min_periods=3).std()
        
        # Price momentum
        for window in [3, 5, 10]:
            features[f'momentum_{window}m'] = close - close.shift(window)
        
        # Price position relative to moving averages
        for window in [5, 10, 20]:
            ma = close.rolling(window=window, min_periods=window//2).mean()
            features[f'price_vs_ma_{window}'] = (close - ma) / ma
        
        # High-low range
        features['hl_range'] = (df_1min['high'] - df_1min['low']) / df_1min['low']
        
        # Target: future price change (future_minutes ahead)
        future_close = close.shift(-future_minutes)
        features['target'] = (future_close - close) / close
        
        # Drop rows with NaN values
        features = features.dropna()
        
        if len(features) < 30:
            logger.warning(f"Only {len(features)} rows after dropping NaN")
            return None, None, None
        
        logger.info(f"Final feature matrix: {len(features)} rows")
        
        # Prepare feature columns (exclude target and close)
        feature_cols = [c for c in features.columns if c not in ['target', 'close']]
        
        X = features[feature_cols].values
        y = features['target'].values
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Features: {feature_cols[:10]}...")
        
        return X, y, feature_cols
        
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_data_info() -> dict:
    """Get information about available data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM prices")
        price_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prices")
        row = cursor.fetchone()
        min_ts, max_ts = row[0], row[1] if row else (None, None)
        
        conn.close()
        
        return {
            'price_points': price_count,
            'earliest_timestamp': min_ts,
            'latest_timestamp': max_ts,
            'min_required_for_training': 60,
            'ready_for_training': price_count >= 60
        }
    except Exception as e:
        logger.error(f"Failed to get data info: {e}")
        return {'price_points': 0, 'ready_for_training': False}


def get_current_features(future_minutes: int = 5) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Get current features for real-time prediction.
    
    Args:
        future_minutes: Prediction horizon in minutes
        
    Returns:
        Tuple of (current_features, feature_names) or (None, None) if insufficient data
    """
    try:
        X, y, feature_names = compute_features(
            target_timeframes=[1, 5],
            future_minutes=future_minutes
        )
        
        if X is None or len(X) == 0:
            logger.warning("No features available for current prediction")
            return None, None
        
        # Return the most recent feature row
        current_features = X[-1:].reshape(1, -1)
        return current_features, feature_names
        
    except Exception as e:
        logger.error(f"Failed to get current features: {e}")
        return None, None


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing HYPE feature engineering...")
    X, y, cols = compute_features()
    if X is not None:
        print(f"Success! X shape: {X.shape}")
        print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    else:
        print("Failed to compute features")
