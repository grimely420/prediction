#!/usr/bin/env python3
"""
Advanced ML-based prediction module with ensemble support and regime detection.
"""

import os
import sys
import joblib
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging


class MLPredictor:
    """Advanced machine learning predictor with ensemble and regime support."""
    
    def __init__(self, symbol: str, model_dir: str, feature_engineer, 
                 prediction_horizon: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the ML predictor.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'BNB')
            model_dir: Directory containing trained models
            feature_engineer: Module with compute_features function
            prediction_horizon: Prediction horizon in minutes
            logger: Optional logger instance
        """
        self.symbol = symbol
        self.model_dir = model_dir
        self.feature_engineer = feature_engineer
        self.prediction_horizon = prediction_horizon
        self.logger = logger or setup_logging(f"{symbol}-Predictor")
        
        # Model storage
        self.model = None
        self.advanced_ensemble = None
        self.feature_names = None
        self.scaler = None
        self.last_loaded = None
        self.use_advanced = False
        
        # Regime detection
        self.regime_detector = None
        self.use_regime = False
        
    def load_latest_model(self) -> bool:
        """
        Load the latest trained model (advanced or basic).
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # First try to load advanced ensemble model
            advanced_paths = [
                os.path.join(self.model_dir, f"ensemble_{self.symbol}_latest.pkl"),
                os.path.join(self.model_dir, f"advanced_{self.symbol}_latest.pkl"),
                os.path.join(self.model_dir, f"advanced_trainer_{self.symbol}.pkl")
            ]
            
            for path in advanced_paths:
                if os.path.exists(path):
                    self.logger.info(f"Loading advanced model from {path}")
                    model_data = joblib.load(path)
                    
                    # Handle different model formats
                    if hasattr(model_data, 'predict'):  # EnsembleModel
                        self.advanced_ensemble = model_data
                        self.use_advanced = True
                        self.use_regime = False
                        self.last_loaded = "advanced_ensemble"
                        self.logger.info("Loaded advanced ensemble model")
                        return True
                    elif isinstance(model_data, dict):
                        if 'ensemble' in model_data:
                            self.advanced_ensemble = model_data['ensemble']
                            self.use_advanced = True
                            if 'regime_detector' in model_data:
                                self.regime_detector = model_data['regime_detector']
                                self.use_regime = True
                            self.last_loaded = "advanced_trainer"
                            self.logger.info("Loaded advanced trainer model with ensemble")
                            return True
                        else:
                            # Basic dict format
                            self.model = model_data.get('model')
                            self.scaler = model_data.get('scaler')
                            self.feature_names = model_data.get('feature_names')
                            self.last_loaded = "basic_dict"
                            self.logger.info("Loaded basic model (dict format)")
                            return True
            
            # Fall back to basic XGBoost model
            basic_path = os.path.join(self.model_dir, f"xgb_{self.prediction_horizon}min_latest.pkl")
            if os.path.exists(basic_path):
                self.model, self.feature_names = joblib.load(basic_path)
                self.last_loaded = "basic_xgboost"
                self.logger.info(f"Loaded basic XGBoost model from {basic_path}")
                return True
            
            self.logger.warning(f"No model found in {self.model_dir}")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_current_features(self) -> Optional[np.ndarray]:
        """
        Get current features for prediction.
        
        Returns:
            Feature array or None if insufficient data
        """
        try:
            # Get latest features using the feature engineer
            X, y, feature_names = self.feature_engineer.compute_features(
                future_minutes=self.prediction_horizon
            )
            
            if X is None or len(X) == 0:
                self.logger.warning("No features available")
                return None
            
            # Return the most recent feature row
            return X[-1:].reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Failed to get features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, current_price: float, recent_ohlc: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Make a prediction using the ML model.
        
        Args:
            current_price: Current price of the cryptocurrency
            recent_ohlc: Optional recent OHLC data for regime detection
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        try:
            # Ensure model is loaded
            if self.model is None and self.advanced_ensemble is None:
                if not self.load_latest_model():
                    self.logger.warning("Using fallback prediction - no model available")
                    return self.fallback_prediction(current_price)
            
            # Get current features
            features = self.get_current_features()
            if features is None:
                self.logger.warning("Using fallback prediction - no features available")
                return self.fallback_prediction(current_price)
            
            # Make prediction based on model type
            if self.use_advanced and self.advanced_ensemble is not None:
                # Use advanced ensemble
                if hasattr(self.advanced_ensemble, 'predict'):
                    # EnsembleModel instance
                    result = self.advanced_ensemble.predict(features, return_individual=False)
                    predicted_change_pct = result['prediction'][0] if isinstance(result['prediction'], np.ndarray) else result['prediction']
                    uncertainty = result.get('uncertainty', [0])[0] if isinstance(result.get('uncertainty'), np.ndarray) else result.get('uncertainty', 0)
                    confidence = result.get('confidence', [0.5])[0] if isinstance(result.get('confidence'), np.ndarray) else result.get('confidence', 0.5)
                    model_used = "advanced_ensemble"
                    
                    # Add ensemble metadata
                    ensemble_meta = {
                        'ensemble_weights': result.get('weights', {}),
                        'uncertainty': uncertainty,
                        'confidence_score': confidence
                    }
                else:
                    # Fallback to basic prediction
                    predicted_change_pct = self._basic_predict(features)
                    uncertainty = None
                    confidence = None
                    model_used = "advanced_basic"
                    ensemble_meta = {}
            else:
                # Use basic model
                predicted_change_pct = self._basic_predict(features)
                uncertainty = None
                confidence = None
                model_used = "xgboost"
                ensemble_meta = {}
            
            # Apply regime adjustment if available
            if self.use_regime and self.regime_detector is not None and recent_ohlc is not None:
                try:
                    regime_metrics = self.regime_detector.detect_regime(
                        recent_ohlc['close'],
                        recent_ohlc.get('high'),
                        recent_ohlc.get('low')
                    )
                    
                    # Adjust prediction based on regime
                    adjusted_change_pct = self.regime_detector.get_prediction_adjustment(predicted_change_pct)
                    
                    regime_meta = {
                        'regime': regime_metrics.regime.value,
                        'trend_strength': regime_metrics.trend_strength,
                        'volatility': regime_metrics.volatility,
                        'momentum': regime_metrics.momentum,
                        'support': regime_metrics.support_level,
                        'resistance': regime_metrics.resistance_level,
                        'strategy': self.regime_detector.get_recommended_strategy(),
                        'raw_prediction': predicted_change_pct,
                        'regime_adjustment': adjusted_change_pct / (predicted_change_pct + 1e-10) - 1
                    }
                    
                    predicted_change_pct = adjusted_change_pct
                    model_used = f"{model_used}_regime_adjusted"
                    
                except Exception as e:
                    self.logger.warning(f"Regime adjustment failed: {e}")
                    regime_meta = {}
            else:
                regime_meta = {}
            
            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_change_pct)
            
            # Calculate confidence interval
            confidence_interval = self.calculate_confidence(uncertainty, predicted_change_pct)
            
            # Build result
            result = {
                'success': True,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'change_percent': float(predicted_change_pct * 100),
                'model_used': model_used,
                'confidence_interval': confidence_interval,
                'prediction_horizon': self.prediction_horizon,
                'model_type': self.last_loaded or 'unknown'
            }
            
            # Add metadata
            if ensemble_meta:
                result['ensemble'] = ensemble_meta
            if regime_meta:
                result['regime'] = regime_meta
            
            self.logger.info(
                f"Prediction: ${predicted_price:.2f} "
                f"(Current: ${current_price:.2f}, "
                f"Change: {predicted_change_pct*100:+.2f}%, "
                f"Model: {model_used})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self.fallback_prediction(current_price)
    
    def _basic_predict(self, features: np.ndarray) -> float:
        """Make prediction with basic model."""
        if self.model is None:
            raise RuntimeError("No model available for prediction")
        
        # Scale features if scaler available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make prediction
        if hasattr(self.model, 'predict'):
            return float(self.model.predict(features)[0])
        else:
            # Assume it's already a prediction function
            return float(self.model(features)[0])
    
    def calculate_confidence(self, uncertainty: Optional[float] = None, 
                           predicted_change: float = 0) -> Dict[str, float]:
        """
        Calculate confidence interval for prediction.
        
        Args:
            uncertainty: Model uncertainty (std dev)
            predicted_change: Predicted percentage change
            
        Returns:
            Dictionary with confidence bounds
        """
        if uncertainty is not None and uncertainty > 0:
            # Use model uncertainty
            confidence_level = 0.68  # 1 standard deviation
            lower = -uncertainty * 100
            upper = uncertainty * 100
        else:
            # Estimate based on typical volatility
            base_volatility = 0.01  # 1%
            
            # Increase uncertainty for larger predicted changes
            magnitude_factor = min(abs(predicted_change) * 10, 1.0)
            volatility_estimate = base_volatility * (1 + magnitude_factor)
            
            confidence_level = 0.68
            lower = -volatility_estimate * 100
            upper = volatility_estimate * 100
        
        return {
            'lower_percent': lower,
            'upper_percent': upper,
            'confidence_level': confidence_level,
            'estimated_accuracy': max(0, 100 - (upper - lower))
        }
    
    def fallback_prediction(self, current_price: float) -> Dict[str, Any]:
        """
        Fallback prediction when ML model is unavailable.
        
        Args:
            current_price: Current price
            
        Returns:
            Dictionary with fallback prediction
        """
        # Simple momentum-based prediction
        change_pct = 0.0005  # 0.05% upward bias
        predicted_price = current_price * (1 + change_pct)
        
        return {
            'success': True,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_percent': change_pct * 100,
            'model_used': 'fallback_momentum',
            'confidence_interval': {
                'lower_percent': -0.5,
                'upper_percent': 0.5,
                'confidence_level': 0.5,
                'estimated_accuracy': 50
            },
            'prediction_horizon': self.prediction_horizon,
            'fallback': True,
            'note': 'ML model not available - using momentum heuristic'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        info = {
            'symbol': self.symbol,
            'prediction_horizon': self.prediction_horizon,
            'model_dir': self.model_dir,
            'last_loaded': self.last_loaded,
            'use_advanced': self.use_advanced,
            'use_regime': self.use_regime,
            'has_model': self.model is not None or self.advanced_ensemble is not None
        }
        
        if self.advanced_ensemble and hasattr(self.advanced_ensemble, 'performance'):
            info['ensemble_models'] = list(self.advanced_ensemble.performance.keys())
            info['model_weights'] = {k: v.weight for k, v in self.advanced_ensemble.performance.items()}
        
        return info


def get_predictor(symbol: str, model_dir: str, feature_engineer, 
                 prediction_horizon: int) -> MLPredictor:
    """
    Factory function to get a predictor instance.
    
    Args:
        symbol: Cryptocurrency symbol
        model_dir: Model directory
        feature_engineer: Feature engineering module
        prediction_horizon: Prediction horizon in minutes
        
    Returns:
        MLPredictor instance
    """
    return MLPredictor(
        symbol=symbol,
        model_dir=model_dir,
        feature_engineer=feature_engineer,
        prediction_horizon=prediction_horizon
    )


# Convenience function for getting advanced predictor
def get_advanced_predictor(symbol: str, model_dir: str, feature_engineer,
                          prediction_horizon: int, use_regime: bool = True) -> MLPredictor:
    """
    Get predictor configured for advanced models.
    
    Args:
        symbol: Cryptocurrency symbol
        model_dir: Model directory
        feature_engineer: Feature engineering module
        prediction_horizon: Prediction horizon in minutes
        use_regime: Whether to use regime detection
        
    Returns:
        Configured MLPredictor
    """
    predictor = MLPredictor(
        symbol=symbol,
        model_dir=model_dir,
        feature_engineer=feature_engineer,
        prediction_horizon=prediction_horizon
    )
    
    # Try to load advanced model
    predictor.load_latest_model()
    
    return predictor
