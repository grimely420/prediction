#!/usr/bin/env python3
"""
Market regime detection for adaptive prediction strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger("Market-Regime")


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    BULLISH = "bullish"
    RANGING = "ranging"
    BEARISH = "bearish"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Metrics for a specific market regime."""
    regime: MarketRegime
    trend_strength: float  # 0 to 1
    volatility: float
    momentum: float
    support_level: float
    resistance_level: float
    avg_true_range: float
    timestamp: pd.Timestamp


class MarketRegimeDetector:
    """
    Detects current market regime using multiple indicators.
    """
    
    def __init__(self, 
                 trend_window: int = 50,
                 volatility_window: int = 20,
                 regime_history_size: int = 100):
        """
        Initialize regime detector.
        
        Args:
            trend_window: Window for trend detection
            volatility_window: Window for volatility calculation
            regime_history_size: How many past regimes to remember
        """
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.regime_history = deque(maxlen=regime_history_size)
        self.price_history = deque(maxlen=500)
        self.current_regime = MarketRegime.UNKNOWN
        
    def detect_regime(self, 
                      close: pd.Series, 
                      high: pd.Series = None, 
                      low: pd.Series = None,
                      volume: pd.Series = None) -> RegimeMetrics:
        """
        Detect current market regime from price data.
        
        Args:
            close: Close prices
            high: High prices (optional)
            low: Low prices (optional)
            volume: Volume data (optional)
            
        Returns:
            RegimeMetrics with current regime classification
        """
        if len(close) < self.trend_window:
            logger.warning(f"Insufficient data for regime detection: {len(close)} < {self.trend_window}")
            return RegimeMetrics(
                regime=MarketRegime.UNKNOWN,
                trend_strength=0,
                volatility=0,
                momentum=0,
                support_level=close.iloc[-1] * 0.95,
                resistance_level=close.iloc[-1] * 1.05,
                avg_true_range=0,
                timestamp=close.index[-1] if hasattr(close, 'index') else pd.Timestamp.now()
            )
        
        # Calculate indicators
        trend_strength = self._calculate_trend_strength(close)
        volatility = self._calculate_volatility(close)
        momentum = self._calculate_momentum(close)
        atr = self._calculate_atr(high, low, close)
        
        # Support and resistance levels
        support, resistance = self._calculate_support_resistance(close)
        
        # Classify regime
        regime = self._classify_regime(trend_strength, volatility, momentum)
        
        metrics = RegimeMetrics(
            regime=regime,
            trend_strength=abs(trend_strength),
            volatility=volatility,
            momentum=momentum,
            support_level=support,
            resistance_level=resistance,
            avg_true_range=atr,
            timestamp=close.index[-1] if hasattr(close, 'index') else pd.Timestamp.now()
        )
        
        self.current_regime = regime
        self.regime_history.append(metrics)
        
        return metrics
    
    def _calculate_trend_strength(self, close: pd.Series) -> float:
        """
        Calculate trend strength using linear regression slope normalized by volatility.
        
        Returns:
            Normalized trend strength (-1 to 1, where > 0.5 is strong trend)
        """
        prices = close.tail(self.trend_window).values
        x = np.arange(len(prices))
        
        # Linear regression
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Normalize by average price
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price
        
        # Calculate R-squared (trend consistency)
        predicted = slope * x + intercept
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Trend strength = slope direction * consistency
        trend_strength = np.sign(normalized_slope) * r_squared
        
        return trend_strength
    
    def _calculate_volatility(self, close: pd.Series) -> float:
        """
        Calculate normalized volatility.
        
        Returns:
            Volatility as coefficient of variation
        """
        recent_prices = close.tail(self.volatility_window)
        returns = recent_prices.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252 * 24 * 60)  # Assuming minute data
        
        return volatility
    
    def _calculate_momentum(self, close: pd.Series) -> float:
        """
        Calculate momentum using multiple timeframes.
        
        Returns:
            Momentum score (-1 to 1)
        """
        # Short, medium, and long-term momentum
        mom_short = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0
        mom_medium = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
        mom_long = (close.iloc[-1] - close.iloc[-50]) / close.iloc[-50] if len(close) >= 50 else 0
        
        # Weighted combination
        momentum = (mom_short * 0.5 + mom_medium * 0.3 + mom_long * 0.2)
        
        # Normalize to [-1, 1]
        momentum = np.clip(momentum, -1, 1)
        
        return momentum
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """
        Calculate Average True Range.
        """
        if high is None or low is None:
            # Use close price volatility as proxy
            return close.tail(self.volatility_window).pct_change().std() * close.iloc[-1]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = true_range.tail(self.volatility_window).mean()
        
        return atr
    
    def _calculate_support_resistance(self, close: pd.Series) -> Tuple[float, float]:
        """
        Calculate dynamic support and resistance levels.
        """
        recent_prices = close.tail(self.trend_window)
        
        # Simple method: use percentiles
        support = recent_prices.quantile(0.05)
        resistance = recent_prices.quantile(0.95)
        
        return support, resistance
    
    def _classify_regime(self, trend_strength: float, volatility: float, momentum: float) -> MarketRegime:
        """
        Classify market regime based on indicators.
        """
        # High volatility regimes
        if volatility > 0.8:  # 80% annualized volatility
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility
        if volatility < 0.2:
            return MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        if abs(trend_strength) > 0.7:
            if trend_strength > 0:
                return MarketRegime.STRONG_UPTREND if momentum > 0.3 else MarketRegime.BULLISH
            else:
                return MarketRegime.STRONG_DOWNTREND if momentum < -0.3 else MarketRegime.BEARISH
        
        # Weak trends
        if abs(trend_strength) > 0.3:
            if trend_strength > 0:
                return MarketRegime.WEAK_UPTREND
            else:
                return MarketRegime.WEAK_DOWNTREND
        
        # No clear trend
        return MarketRegime.RANGING
    
    def get_regime_distribution(self, window: int = 50) -> Dict[MarketRegime, float]:
        """
        Get distribution of regimes over recent history.
        
        Returns:
            Dictionary mapping regime to frequency
        """
        if len(self.regime_history) < window:
            window = len(self.regime_history)
        
        recent_regimes = list(self.regime_history)[-window:]
        regime_counts = {}
        
        for metrics in recent_regimes:
            regime_counts[metrics.regime] = regime_counts.get(metrics.regime, 0) + 1
        
        # Convert to frequencies
        total = len(recent_regimes)
        return {regime: count / total for regime, count in regime_counts.items()}
    
    def is_trending(self) -> bool:
        """Check if market is currently trending."""
        trending_regimes = [
            MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND,
            MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND
        ]
        return self.current_regime in trending_regimes
    
    def is_volatile(self) -> bool:
        """Check if market is currently volatile."""
        return self.current_regime == MarketRegime.HIGH_VOLATILITY
    
    def is_ranging(self) -> bool:
        """Check if market is ranging/sideways."""
        return self.current_regime in [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY]
    
    def get_recommended_strategy(self) -> str:
        """
        Get recommended trading/prediction strategy based on current regime.
        """
        strategy_map = {
            MarketRegime.STRONG_UPTREND: "trend_following_long",
            MarketRegime.WEAK_UPTREND: "momentum_long",
            MarketRegime.BULLISH: "momentum_long",
            MarketRegime.RANGING: "mean_reversion",
            MarketRegime.BEARISH: "momentum_short",
            MarketRegime.WEAK_DOWNTREND: "momentum_short",
            MarketRegime.STRONG_DOWNTREND: "trend_following_short",
            MarketRegime.HIGH_VOLATILITY: "volatility_breakout",
            MarketRegime.LOW_VOLATILITY: "range_trading",
            MarketRegime.UNKNOWN: "conservative"
        }
        
        return strategy_map.get(self.current_regime, "conservative")
    
    def get_prediction_adjustment(self, base_prediction: float) -> float:
        """
        Adjust prediction based on current market regime.
        
        Args:
            base_prediction: Original prediction value
            
        Returns:
            Adjusted prediction
        """
        # Regime-specific adjustments
        adjustments = {
            MarketRegime.STRONG_UPTREND: 1.2,      # Increase bullish predictions
            MarketRegime.WEAK_UPTREND: 1.1,
            MarketRegime.BULLISH: 1.05,
            MarketRegime.RANGING: 1.0,           # No adjustment
            MarketRegime.BEARISH: 0.95,
            MarketRegime.WEAK_DOWNTREND: 0.9,
            MarketRegime.STRONG_DOWNTREND: 0.8,  # Decrease bullish predictions
            MarketRegime.HIGH_VOLATILITY: 0.7,   # Reduce confidence in volatile markets
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.UNKNOWN: 0.9
        }
        
        adjustment = adjustments.get(self.current_regime, 1.0)
        return base_prediction * adjustment


class RegimeAwarePredictor:
    """
    Predictor that adapts based on market regime.
    """
    
    def __init__(self, base_predictor, regime_detector: MarketRegimeDetector = None):
        """
        Initialize regime-aware predictor.
        
        Args:
            base_predictor: Base prediction model
            regime_detector: Regime detector (creates default if None)
        """
        self.base_predictor = base_predictor
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.regime_performance = {}  # Track performance per regime
        
    def predict(self, features: np.ndarray, price_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Make regime-aware prediction.
        
        Args:
            features: Feature matrix for prediction
            price_data: Recent price data for regime detection
            
        Returns:
            Prediction with regime information
        """
        # Detect current regime
        if price_data is not None:
            regime_metrics = self.regime_detector.detect_regime(
                price_data['close'],
                price_data.get('high'),
                price_data.get('low'),
                price_data.get('volume')
            )
        else:
            regime_metrics = None
        
        # Get base prediction
        if hasattr(self.base_predictor, 'predict'):
            if isinstance(self.base_predictor.predict(features), dict):
                result = self.base_predictor.predict(features)
                base_prediction = result['prediction']
            else:
                base_prediction = self.base_predictor.predict(features)
                result = {'prediction': base_prediction}
        else:
            base_prediction = self.base_predictor(features)
            result = {'prediction': base_prediction}
        
        # Adjust based on regime
        if regime_metrics:
            adjusted_prediction = self.regime_detector.get_prediction_adjustment(base_prediction)
            
            result.update({
                'regime': regime_metrics.regime.value,
                'trend_strength': regime_metrics.trend_strength,
                'volatility': regime_metrics.volatility,
                'momentum': regime_metrics.momentum,
                'support': regime_metrics.support_level,
                'resistance': regime_metrics.resistance_level,
                'base_prediction': base_prediction,
                'adjusted_prediction': adjusted_prediction,
                'strategy': self.regime_detector.get_recommended_strategy()
            })
            
            # Use adjusted prediction
            result['prediction'] = adjusted_prediction
        
        return result
    
    def update_performance(self, regime: MarketRegime, predicted: float, actual: float):
        """Update performance tracking for regime."""
        error = abs(predicted - actual)
        
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {'errors': [], 'count': 0}
        
        self.regime_performance[regime]['errors'].append(error)
        self.regime_performance[regime]['count'] += 1
        
        # Keep only recent errors
        if len(self.regime_performance[regime]['errors']) > 100:
            self.regime_performance[regime]['errors'].pop(0)


if __name__ == "__main__":
    # Test regime detection
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic price data for different regimes
    np.random.seed(42)
    
    # 1. Strong uptrend
    n = 200
    trend = np.cumsum(np.ones(n) * 0.001 + np.random.randn(n) * 0.0005)
    uptrend_prices = 100 * np.exp(trend)
    
    # 2. Ranging market
    ranging_prices = 100 + 5 * np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.5
    
    # 3. High volatility
    volatile_prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    
    detector = MarketRegimeDetector()
    
    for name, prices in [
        ("Strong Uptrend", uptrend_prices),
        ("Ranging", ranging_prices),
        ("High Volatility", volatile_prices)
    ]:
        close = pd.Series(prices, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
        metrics = detector.detect_regime(close)
        
        print(f"\n{name}:")
        print(f"  Regime: {metrics.regime.value}")
        print(f"  Trend Strength: {metrics.trend_strength:.3f}")
        print(f"  Volatility: {metrics.volatility:.3f}")
        print(f"  Momentum: {metrics.momentum:.3f}")
        print(f"  Strategy: {detector.get_recommended_strategy()}")
