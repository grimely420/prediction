#!/usr/bin/env python3
"""
Advanced feature engineering with comprehensive technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy import stats
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger("Advanced-Features")


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering with 50+ technical indicators.
    """
    
    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [5, 10, 15, 30, 60, 120]
        self.scaler = RobustScaler()
        self.feature_names = []
        
    def compute_all_features(self, df: pd.DataFrame, prediction_horizon: int = 15) -> pd.DataFrame:
        """
        Compute comprehensive feature set from OHLCV data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close'] + optionally 'volume'
            prediction_horizon: Minutes ahead to predict
            
        Returns:
            DataFrame with all features and target
        """
        if len(df) < max(self.lookback_windows) + prediction_horizon:
            logger.warning(f"Insufficient data: {len(df)} rows, need {max(self.lookback_windows) + prediction_horizon}")
            return None
        
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 1. Price-based features
        features = self._add_price_features(features, close, high, low)
        
        # 2. Returns and momentum
        features = self._add_return_features(features, close)
        
        # 3. Volatility features
        features = self._add_volatility_features(features, close, high, low)
        
        # 4. Moving averages and trends
        features = self._add_ma_features(features, close)
        
        # 5. Technical indicators
        features = self._add_technical_indicators(features, close, high, low)
        
        # 6. Pattern features
        features = self._add_pattern_features(features, df)
        
        # 7. Statistical features
        features = self._add_statistical_features(features, close)
        
        # 8. Time features
        features = self._add_time_features(features, df.index)
        
        # 9. Target variable
        future_return = close.shift(-prediction_horizon).pct_change(prediction_horizon)
        features['target'] = future_return
        
        # Drop NaN values
        features = features.dropna()
        
        logger.info(f"Computed {len(features.columns)-1} features from {len(df)} rows -> {len(features)} valid samples")
        
        return features
    
    def _add_price_features(self, features: pd.DataFrame, close: pd.Series, 
                           high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """Add basic price features."""
        features['close'] = close
        features['high'] = high
        features['low'] = low
        features['hl_range'] = (high - low) / close
        features['hl_range_pct'] = (high - low) / low
        
        # Price position in range
        features['price_position'] = (close - low) / (high - low + 1e-10)
        
        # Price velocity (rate of change)
        for window in [3, 5, 10]:
            features[f'price_velocity_{window}'] = close.diff(window) / window
        
        return features
    
    def _add_return_features(self, features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add return-based features."""
        returns = close.pct_change()
        
        # Simple returns
        for window in self.lookback_windows:
            features[f'return_{window}m'] = close.pct_change(window)
        
        # Cumulative returns
        features['return_24h'] = close.pct_change(1440) if len(close) > 1440 else np.nan
        features['return_7d'] = close.pct_change(10080) if len(close) > 10080 else np.nan
        
        # Acceleration (change in returns)
        for window in [5, 10, 20]:
            ret = close.pct_change(window)
            features[f'return_accel_{window}'] = ret.diff(window)
        
        # Log returns (better statistical properties)
        log_returns = np.log(close / close.shift(1))
        for window in [5, 10, 20, 60]:
            features[f'log_return_{window}'] = log_returns.rolling(window).sum()
        
        # Direction features
        features['direction_5m'] = np.sign(returns.rolling(5).sum())
        features['direction_15m'] = np.sign(returns.rolling(15).sum())
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, close: pd.Series,
                                 high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """Add volatility features including ATR."""
        returns = close.pct_change()
        
        # Standard volatility measures
        for window in [5, 10, 15, 30, 60]:
            features[f'volatility_{window}'] = returns.rolling(window).std()
            features[f'volatility_ma_{window}'] = features[f'volatility_{window}'].rolling(window).mean()
        
        # True Range and ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for window in [5, 14, 20]:
            features[f'true_range_{window}'] = true_range.rolling(window).mean()
            features[f'atr_{window}'] = true_range.ewm(span=window, adjust=False).mean()
        
        # Normalized ATR
        features['atr_14_pct'] = features['atr_14'] / close
        
        # Volatility regimes
        features['vol_ratio_10_60'] = features['volatility_10'] / (features['volatility_60'] + 1e-10)
        
        # Parkinson volatility (uses high-low)
        for window in [10, 20]:
            hl_log = np.log(high / low)
            features[f'parkinson_vol_{window}'] = np.sqrt(
                hl_log.rolling(window).mean() / (4 * np.log(2))
            )
        
        return features
    
    def _add_ma_features(self, features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add moving average features."""
        # Simple MAs
        for window in [5, 10, 15, 20, 30, 50, 100, 200]:
            if len(close) >= window:
                ma = close.rolling(window).mean()
                features[f'ma_{window}'] = ma
                features[f'price_vs_ma_{window}'] = (close - ma) / ma
                features[f'ma_slope_{window}'] = ma.diff(5) / ma
        
        # Exponential MAs
        for span in [12, 26, 50]:
            if len(close) >= span:
                ema = close.ewm(span=span, adjust=False).mean()
                features[f'ema_{span}'] = ema
                features[f'price_vs_ema_{span}'] = (close - ema) / ema
        
        # MA crossovers
        if len(close) >= 50:
            features['ma_cross_10_30'] = features['ma_10'] - features['ma_30']
            features['ma_cross_golden'] = ((features['ma_50'] > features['ma_200']) & 
                                          (features['ma_50'].shift(1) <= features['ma_200'].shift(1))).astype(int)
            features['ma_cross_death'] = ((features['ma_50'] < features['ma_200']) & 
                                         (features['ma_50'].shift(1) >= features['ma_200'].shift(1))).astype(int)
        
        # Distance from moving averages
        for window in [20, 50]:
            if f'ma_{window}' in features.columns:
                ma = features[f'ma_{window}']
                features[f'distance_ma_{window}_std'] = (close - ma) / close.rolling(window).std()
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, close: pd.Series,
                                 high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """Add technical indicators."""
        returns = close.pct_change()
        
        # RSI (multiple periods)
        for period in [6, 14, 21]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Stochastic Oscillator
        for window in [14, 21]:
            lowest_low = low.rolling(window).min()
            highest_high = high.rolling(window).max()
            features[f'stoch_k_{window}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            features[f'stoch_d_{window}'] = features[f'stoch_k_{window}'].rolling(3).mean()
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        for window in [14, 20]:
            tp_ma = tp.rolling(window).mean()
            tp_md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            features[f'cci_{window}'] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)
        
        # Williams %R
        for window in [14, 21]:
            highest_high = high.rolling(window).max()
            lowest_low = low.rolling(window).min()
            features[f'williams_r_{window}'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        
        # Bollinger Bands
        for window in [20, 50]:
            ma = close.rolling(window).mean()
            std = close.rolling(window).std()
            features[f'bb_upper_{window}'] = ma + (2 * std)
            features[f'bb_lower_{window}'] = ma - (2 * std)
            features[f'bb_pct_{window}'] = (close - features[f'bb_lower_{window}']) / (
                features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'] + 1e-10
            )
            features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / ma
        
        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr_14 = features['atr_14'] if 'atr_14' in features else returns.rolling(14).std()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        return features
    
    def _add_pattern_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        open_price = df.get('open', df['close'].shift(1))
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadows
        body = close - open_price
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        
        features['body_size'] = abs(body) / close
        features['upper_shadow_pct'] = upper_shadow / close
        features['lower_shadow_pct'] = lower_shadow / close
        features['body_to_shadow_ratio'] = abs(body) / (upper_shadow + lower_shadow + 1e-10)
        
        # Doji detection
        features['is_doji'] = (abs(body) / (high - low + 1e-10) < 0.1).astype(int)
        
        # Trend detection
        features['higher_highs'] = (high > high.shift(1)).astype(int)
        features['lower_lows'] = (low < low.shift(1)).astype(int)
        features['uptrend_5'] = features['higher_highs'].rolling(5).sum()
        features['downtrend_5'] = features['lower_lows'].rolling(5).sum()
        
        # Support/Resistance proximity (simple version)
        rolling_min = low.rolling(20).min()
        rolling_max = high.rolling(20).max()
        features['proximity_to_support'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
        features['proximity_to_resistance'] = (rolling_max - close) / (rolling_max - rolling_min + 1e-10)
        
        return features
    
    def _add_statistical_features(self, features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add statistical features."""
        returns = close.pct_change()
        
        # Distribution moments
        for window in [10, 20, 60]:
            ret_window = returns.rolling(window)
            features[f'skew_{window}'] = ret_window.skew()
            features[f'kurt_{window}'] = ret_window.kurt()
        
        # Entropy (measure of randomness)
        for window in [20, 60]:
            ret_sign = np.sign(returns.rolling(window).sum())
            features[f'entropy_{window}'] = np.where(
                ret_sign == 0, 1.0, 
                -ret_sign * np.log(abs(ret_sign) + 1e-10)
            )
        
        # Z-scores
        for window in [20, 50]:
            ma = close.rolling(window).mean()
            std = close.rolling(window).std()
            features[f'zscore_{window}'] = (close - ma) / (std + 1e-10)
        
        # Quantile positions
        for window in [20, 60]:
            features[f'quantile_pos_{window}'] = close.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            features[f'autocorr_{lag}'] = returns.rolling(30).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add time-based features."""
        features['hour'] = index.hour
        features['minute'] = index.minute
        features['day_of_week'] = index.dayofweek
        features['is_weekend'] = (index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding of time
        features['hour_sin'] = np.sin(2 * np.pi * index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * index.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * index.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * index.dayofweek / 7)
        
        # Market session (simplified - assumes UTC)
        features['is_asia_session'] = ((index.hour >= 0) & (index.hour < 9)).astype(int)
        features['is_europe_session'] = ((index.hour >= 9) & (index.hour < 17)).astype(int)
        features['is_us_session'] = ((index.hour >= 14) & (index.hour < 22)).astype(int)
        
        return features
    
    def get_feature_matrix(self, features_df: pd.DataFrame, 
                          target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract feature matrix and target from features DataFrame."""
        exclude_cols = [target_col, 'close', 'high', 'low']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values if target_col in features_df.columns else None
        
        return X, y, feature_cols


def engineer_features(df: pd.DataFrame, prediction_horizon: int = 15) -> pd.DataFrame:
    """
    Convenience function for advanced feature engineering.
    
    Args:
        df: DataFrame with OHLCV data
        prediction_horizon: Minutes to predict ahead
        
    Returns:
        DataFrame with all features
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.compute_all_features(df, prediction_horizon)


if __name__ == "__main__":
    # Test with sample data
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample OHLC data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    trend = np.cumsum(np.random.randn(n) * 0.1)
    close = 50000 + trend * 100
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_price = close + np.random.randn(n) * 20
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }, index=dates)
    
    # Test feature engineering
    engineer = AdvancedFeatureEngineer()
    features = engineer.compute_all_features(df, prediction_horizon=15)
    
    if features is not None:
        print(f"\nFeature Engineering Complete!")
        print(f"Total features: {len(features.columns) - 1}")
        print(f"Samples: {len(features)}")
        print(f"\nFeature categories:")
        print(f"  Price features: {len([c for c in features.columns if any(x in c for x in ['close', 'high', 'low', 'hl'])])}")
        print(f"  Returns: {len([c for c in features.columns if 'return' in c])}")
        print(f"  Volatility: {len([c for c in features.columns if 'vol' in c or 'atr' in c])}")
        print(f"  Moving Averages: {len([c for c in features.columns if 'ma' in c or 'ema' in c])}")
        print(f"  Technical Indicators: {len([c for c in features.columns if any(x in c for x in ['rsi', 'macd', 'stoch', 'cci', 'williams', 'bb', 'adx'])])}")
        print(f"  Patterns: {len([c for c in features.columns if any(x in c for x in ['body', 'shadow', 'doji', 'trend'])])}")
        print(f"  Statistical: {len([c for c in features.columns if any(x in c for x in ['skew', 'kurt', 'zscore', 'autocorr'])])}")
        print(f"  Time: {len([c for c in features.columns if any(x in c for x in ['hour', 'minute', 'day', 'session'])])}")
        print(f"\nFeature names (first 20): {list(features.columns[:20])}")
