#!/usr/bin/env python3
"""
Shared Ensemble Predictor - Combines XGBoost, ARIMA, and Trend models.
Supports both Bitcoin and BNB with configurable parameters.
"""

import os
import sys
import warnings
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

from .utils import setup_logging, calculate_percent_change, ModelNotReadyError, InsufficientDataError


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models:
    - XGBoost: Primary ML model trained on engineered features
    - ARIMA: Statistical time series model
    - Trend: Simple linear extrapolation
    
    Supports both Bitcoin and BNB with configurable parameters.
    """
    
    def __init__(self, 
                 symbol: str,
                 model_dir: str,
                 feature_engineer_module: Any,
                 lookback_minutes: int = 30,
                 prediction_horizon: int = 15,
                 min_data_points: int = 100):
        """
        Initialize the ensemble predictor.
        
        Args:
            symbol: Cryptocurrency symbol (BTC, BNB)
            model_dir: Directory containing model files
            feature_engineer_module: Module with compute_features function
            lookback_minutes: Minutes to look back for trend calculation
            prediction_horizon: Minutes ahead to predict
            min_data_points: Minimum data points for simple predictions
        """
        self.symbol = symbol.upper()
        self.model_dir = model_dir
        self.feature_engineer = feature_engineer_module
        self.lookback_minutes = lookback_minutes
        self.prediction_horizon = prediction_horizon
        self.min_data_points = min_data_points
        
        # Model paths
        self.model_path = os.path.join(model_dir, f"xgb_{prediction_horizon}min_latest.pkl")
        
        # Initialize models
        self.xgb_model = None
        self.feature_names = None
        
        # Setup logging
        self.logger = setup_logging(f"{self.symbol}-Predictor")
        
        # Load XGBoost model if available
        self.load_xgb()
    
    def load_xgb(self) -> None:
        """Load the XGBoost model from disk if it exists."""
        try:
            if os.path.exists(self.model_path):
                self.xgb_model, self.feature_names = joblib.load(self.model_path)
                self.logger.info(f"XGBoost model loaded from {self.model_path}")
                self.logger.info(f"Model features: {len(self.feature_names) if self.feature_names else 0}")
            else:
                self.logger.warning(f"No XGBoost model found at {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {e}")
            self.xgb_model = None
    
    def get_current_trend(self, lookback_minutes: Optional[int] = None) -> float:
        """
        Calculate current price trend using linear regression.
        
        Args:
            lookback_minutes: Override default lookback period
            
        Returns:
            Trend as percentage change per minute
        """
        try:
            lookback = lookback_minutes or self.lookback_minutes
            
            # Load price data
            df = self.feature_engineer.load_price_data(limit=2000)
            if df is None or len(df) < lookback:
                return 0.0
            
            # Get recent prices
            recent = df['price'].iloc[-lookback:]
            x = np.arange(len(recent))
            
            # Linear regression slope
            slope = np.polyfit(x, recent.values, 1)[0]
            current_price = recent.iloc[-1]
            
            if current_price > 0:
                return float(slope / current_price)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trend: {e}")
            return 0.0
    
    def arima_predict(self, steps: Optional[int] = None) -> Optional[float]:
        """
        Predict using ARIMA model.
        
        Args:
            steps: Number of minutes ahead to predict
            
        Returns:
            Predicted percentage change or None if prediction fails
        """
        try:
            steps = steps or self.prediction_horizon
            
            # Load price data
            df = self.feature_engineer.load_price_data(limit=500)
            if df is None or len(df) < 50:
                self.logger.debug("Insufficient data for ARIMA prediction")
                return None
            
            series = df['price'].copy()
            
            # Set frequency if not set
            if series.index.inferred_freq is None:
                series.index = pd.date_range(
                    start=series.index[0], 
                    periods=len(series), 
                    freq='1min'
                )
            
            # Fit ARIMA model
            model = ARIMA(series, order=(5, 1, 0))
            fitted = model.fit()
            
            # Forecast
            forecast = fitted.forecast(steps=steps)
            last_price = series.iloc[-1]
            
            if last_price > 0:
                pct_change = (forecast.iloc[-1] - last_price) / last_price
                return float(pct_change)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"ARIMA prediction failed: {e}")
            return None
    
    def trend_predict(self, lookback: int = 10) -> Optional[float]:
        """
        Predict using simple trend extrapolation.
        
        Args:
            lookback: Number of minutes to look back
            
        Returns:
            Predicted percentage change or None if prediction fails
        """
        try:
            # Load price data
            df = self.feature_engineer.load_price_data(limit=200)
            if df is None or len(df) < lookback:
                return None
            
            prices = df['price'].iloc[-lookback:].values
            x = np.arange(len(prices))
            
            # Linear regression
            slope = np.polyfit(x, prices, 1)[0]
            last_price = prices[-1]
            steps = self.prediction_horizon
            
            if last_price > 0:
                predicted_price = last_price + slope * steps
                pct_change = (predicted_price - last_price) / last_price
                return float(pct_change)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Trend prediction failed: {e}")
            return None
    
    def xgb_predict(self) -> Optional[float]:
        """
        Predict using XGBoost model.
        
        Returns:
            Predicted percentage change or None if model not ready
        """
        if self.xgb_model is None:
            self.logger.debug("XGBoost model not loaded")
            return None
        
        try:
            X, _, _ = self.feature_engineer.compute_features()
            if X is None or len(X) == 0:
                self.logger.debug("No features available for XGBoost prediction")
                return None
            
            # Use most recent features
            latest_features = X[-1].reshape(1, -1)
            pct_change = self.xgb_model.predict(latest_features)[0]
            return float(pct_change)
            
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            return None
    
    def predict(self) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Generate ensemble prediction by combining all models.
        
        Returns:
            Tuple of (best_prediction_pct_change, info_dict) or (None, None)
            
        Raises:
            ModelNotReadyError: If no model can make a prediction
            InsufficientDataError: If there's not enough data
        """
        try:
            # Get current trend for model selection
            trend = self.get_current_trend()
            
            # Get predictions from all models
            predictions = {
                'xgb': self.xgb_predict(),
                'arima': self.arima_predict(),
                'trend': self.trend_predict()
            }
            
            # Filter out None predictions
            preds = {k: v for k, v in predictions.items() if v is not None}
            
            if not preds:
                self.logger.warning("No predictions available from any model")
                return None, None
            
            # Select model whose prediction is closest to current trend
            best_name = min(preds.items(), key=lambda x: abs(x[1] - trend))[0]
            best_value = preds[best_name]
            
            self.logger.debug(
                f"Ensemble prediction - Chosen: {best_name}, Value: {best_value:.4f}, "
                f"Trend: {trend:.4f}"
            )
            
            return best_value, {
                'all': preds,
                'trend': trend,
                'chosen': best_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return None, None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state.
        
        Returns:
            Dictionary with model information
        """
        model_exists = os.path.exists(self.model_path)
        last_trained = None
        
        if model_exists:
            mtime = os.path.getmtime(self.model_path)
            last_trained = datetime.fromtimestamp(mtime).isoformat()
        
        # Get data info from feature engineer
        data_info = self.feature_engineer.get_data_info() if hasattr(self.feature_engineer, 'get_data_info') else {}
        
        return {
            'symbol': self.symbol,
            'model_type': 'XGBoost Regressor',
            'loaded': self.xgb_model is not None,
            'model_exists': model_exists,
            'last_trained': last_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'prediction_horizon': self.prediction_horizon,
            'lookback_minutes': self.lookback_minutes,
            **data_info
        }
