#!/usr/bin/env python3
"""
Feature engineering for Bitcoin price prediction.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging, get_db_connection
from bitcoin.config import DB_PATH

# Setup logging
logger = setup_logging("BTC-Engine")


def load_price_data(limit: Optional[int] = None, since_timestamp: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load price data from database.
    
    Args:
        limit: Maximum number of records to load
        since_timestamp: Only load records after this timestamp
        
    Returns:
        DataFrame with timestamp index and price column, or None if insufficient data
    """
    try:
        conn = get_db_connection(DB_PATH)
        
        query = "SELECT timestamp, price FROM prices"
        params = []
        
        if since_timestamp:
            query += " WHERE timestamp >= ?"
            params.append(since_timestamp)
        
        query += " ORDER BY id ASC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load price data: {e}")
        return None


def resample_to_timeframe(df: pd.DataFrame, timeframe_min: int) -> pd.DataFrame:
    """
    Resample price data to a specific timeframe.
    
    Args:
        df: DataFrame with price data
        timeframe_min: Timeframe in minutes
        
    Returns:
        OHLC DataFrame for the specified timeframe
    """
    rule = f'{timeframe_min}min'
    ohlc = df['price'].resample(rule).ohlc()
    ohlc.columns = [f'open_{timeframe_min}m', f'high_{timeframe_min}m', f'low_{timeframe_min}m', f'close_{timeframe_min}m']
    return ohlc


def compute_features(target_timeframes: List[int] = [1, 3, 5, 10, 15], 
                     lookback_minutes: int = 60, 
                     future_minutes: int = 15) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Compute features for model training.
    
    Args:
        target_timeframes: List of timeframes to compute features for
        lookback_minutes: Lookback period for rolling features
        future_minutes: Number of minutes ahead to predict
        
    Returns:
        Tuple of (X_features, y_targets, feature_names) or (None, None, None) if insufficient data
    """
    try:
        df_raw = load_price_data()
        if df_raw is None or len(df_raw) < 5000:
            logger.warning(f"Insufficient data for feature computation: {len(df_raw) if df_raw is not None else 0} points")
            return None, None, None
        
        # Resample to required timeframes
        resampled = {}
        for tf in target_timeframes:
            resampled[tf] = resample_to_timeframe(df_raw, tf)
        
        # Base on 1-minute timeframe
        base = resampled[1].copy()
        base.index = base.index.floor('1min')
        base['close_1m'] = base['close_1m']
        
        # Add features from other timeframes
        for tf in target_timeframes:
            if tf == 1:
                continue
            ohlc = resampled[tf]
            ohlc = ohlc.reindex(base.index, method='ffill')
            for col in ohlc.columns:
                base[col] = ohlc[col]
        
        # Calculate returns for various windows
        current_price = base['close_1m']
        for window in [1, 3, 5, 10, 15, 30, 60]:
            base[f'ret_{window}m'] = (current_price - current_price.shift(window)) / current_price.shift(window)
        
        # Volatility features
        returns_1m = current_price.pct_change()
        base['volatility_15m'] = returns_1m.rolling(15).std()
        base['volatility_60m'] = returns_1m.rolling(60).std()
        
        # RSI (14 periods)
        delta = returns_1m
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        base['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target: price change in future_minutes
        future_price = current_price.shift(-future_minutes)
        base['target_pct'] = (future_price - current_price) / current_price
        
        # Drop NaN values
        base.dropna(inplace=True)
        
        if len(base) < 100:
            logger.warning(f"Insufficient data after feature engineering: {len(base)} rows")
            return None, None, None
        
        # Prepare feature matrix
        exclude = ['target_pct', 'open_1m', 'high_1m', 'low_1m']
        feature_cols = [c for c in base.columns if c not in exclude and base[c].dtype in [float, int]]
        
        X = base[feature_cols].values
        y = base['target_pct'].values
        
        logger.info(f"Features computed: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_cols
        
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        return None, None, None


def get_data_info() -> dict:
    """Get information about available data."""
    try:
        conn = get_db_connection(DB_PATH)
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
            'min_required_for_training': 1800,
            'ready_for_training': price_count >= 1800
        }
    except Exception as e:
        logger.error(f"Failed to get data info: {e}")
        return {'price_points': 0, 'ready_for_training': False}


def get_current_features(future_minutes: int = 15) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Get current features for real-time prediction.
    
    Args:
        future_minutes: Prediction horizon in minutes
        
    Returns:
        Tuple of (current_features, feature_names) or (None, None) if insufficient data
    """
    try:
        X, y, feature_names = compute_features(
            target_timeframes=[1, 3, 5, 10, 15],
            lookback_minutes=60,
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
