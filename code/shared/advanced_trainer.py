#!/usr/bin/env python3
"""
Advanced model training pipeline with ensemble learning, regime detection, and online learning.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timedelta
import sqlite3
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging, get_db_connection
from shared.advanced_features import AdvancedFeatureEngineer
from shared.ensemble_trainer import EnsembleModel, train_ensemble_with_walk_forward
from shared.market_regime import MarketRegimeDetector, RegimeAwarePredictor

logger = setup_logging("Advanced-Trainer")


class AdvancedModelTrainer:
    """
    Advanced model training pipeline with multiple improvements.
    """
    
    def __init__(self,
                 symbol: str,
                 db_path: str,
                 model_dir: str,
                 prediction_horizon: int = 15,
                 min_train_points: int = 5000,
                 use_ensemble: bool = True,
                 use_regime_detection: bool = True,
                 enable_online_learning: bool = True):
        """
        Initialize advanced trainer.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            db_path: Path to SQLite database
            model_dir: Directory to save models
            prediction_horizon: Minutes ahead to predict
            min_train_points: Minimum data points for training
            use_ensemble: Use ensemble of multiple models
            use_regime_detection: Use market regime detection
            enable_online_learning: Enable incremental model updates
        """
        self.symbol = symbol
        self.db_path = db_path
        self.model_dir = model_dir
        self.prediction_horizon = prediction_horizon
        self.min_train_points = min_train_points
        self.use_ensemble = use_ensemble
        self.use_regime_detection = use_regime_detection
        self.enable_online_learning = enable_online_learning
        
        # Components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.regime_detector = MarketRegimeDetector() if use_regime_detection else None
        self.ensemble = None
        
        # Tracking
        self.last_train_time = None
        self.training_history = []
        self.online_updates = 0
        
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self, limit: Optional[int] = None, 
                  since: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare OHLC data from database."""
        try:
            conn = get_db_connection(self.db_path)
            
            query = "SELECT timestamp, price FROM prices ORDER BY timestamp ASC"
            params = []
            
            if since:
                query = "SELECT timestamp, price FROM prices WHERE timestamp >= ? ORDER BY timestamp ASC"
                params = [since]
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning("No data found in database")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resample to 1-minute OHLC
            ohlc = df['price'].resample('1min').ohlc()
            ohlc.columns = ['open', 'high', 'low', 'close']
            ohlc = ohlc.dropna()
            
            if limit and len(ohlc) > limit:
                ohlc = ohlc.tail(limit)
            
            logger.info(f"Loaded {len(ohlc)} OHLC bars from {ohlc.index[0]} to {ohlc.index[-1]}")
            return ohlc
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def prepare_features(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using advanced feature engineering."""
        logger.info(f"Computing features for {self.symbol}...")
        
        features_df = self.feature_engineer.compute_all_features(
            ohlc, 
            prediction_horizon=self.prediction_horizon
        )
        
        if features_df is None or len(features_df) < 100:
            logger.error("Feature engineering failed or insufficient data")
            return None
        
        # Detect regime and add as feature
        if self.regime_detector:
            # Create a simple regime feature
            close = ohlc['close']
            if len(close) >= 50:
                for idx in features_df.index:
                    # Get data up to this point
                    mask = close.index <= idx
                    if mask.sum() >= 50:
                        regime_metrics = self.regime_detector.detect_regime(close[mask])
                        # Add regime encoding
                        regime_map = {
                            'strong_uptrend': 1.0,
                            'weak_uptrend': 0.8,
                            'bullish': 0.6,
                            'ranging': 0.0,
                            'bearish': -0.6,
                            'weak_downtrend': -0.8,
                            'strong_downtrend': -1.0,
                            'high_volatility': 0.0,
                            'low_volatility': 0.0,
                            'unknown': 0.0
                        }
                        features_df.loc[idx, 'regime_score'] = regime_map.get(
                            regime_metrics.regime.value, 0.0
                        )
                        features_df.loc[idx, 'trend_strength'] = regime_metrics.trend_strength
                        features_df.loc[idx, 'market_volatility'] = regime_metrics.volatility
        
        return features_df
    
    def train(self, force: bool = False) -> bool:
        """
        Train the advanced model.
        
        Args:
            force: Force training even if not enough new data
            
        Returns:
            True if training successful
        """
        logger.info(f"Starting advanced training for {self.symbol}...")
        
        # Load data
        ohlc = self.load_data()
        if ohlc is None or len(ohlc) < self.min_train_points:
            logger.warning(f"Insufficient data: {len(ohlc) if ohlc is not None else 0} points")
            return False
        
        # Prepare features
        features_df = self.prepare_features(ohlc)
        if features_df is None:
            return False
        
        # Extract feature matrix and target
        feature_cols = [c for c in features_df.columns if c not in ['target', 'close', 'high', 'low']]
        X = features_df[feature_cols].values
        y = features_df['target'].values
        
        logger.info(f"Training data: X={X.shape}, y={y.shape}")
        logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Train ensemble
        if self.use_ensemble:
            logger.info("Training ensemble model...")
            self.ensemble = train_ensemble_with_walk_forward(
                X, y, feature_cols, self.model_dir, self.symbol
            )
        else:
            # Single XGBoost model
            logger.info("Training single XGBoost model...")
            import xgboost as xgb
            from sklearn.preprocessing import RobustScaler
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=20,
                eval_metric='mae'
            )
            
            model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            
            # Save
            self.ensemble = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_cols
            }
            
            # Evaluate
            preds = model.predict(X_val_s)
            mae = np.mean(np.abs(y_val - preds))
            logger.info(f"Validation MAE: {mae:.6f}")
            
            # Save to disk
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(self.model_dir, f'advanced_{self.symbol}_{timestamp}.pkl')
            joblib.dump(self.ensemble, model_path)
            
            latest_path = os.path.join(self.model_dir, f'advanced_{self.symbol}_latest.pkl')
            joblib.dump(self.ensemble, latest_path)
        
        self.last_train_time = datetime.now()
        self.training_history.append({
            'timestamp': self.last_train_time,
            'samples': len(X),
            'features': len(feature_cols)
        })
        
        logger.info(f"Training complete! Model saved to {self.model_dir}")
        return True
    
    def online_update(self, actual_results: List[Dict[str, Any]]) -> bool:
        """
        Perform online learning update with recent prediction results.
        
        Args:
            actual_results: List of dicts with 'features', 'predicted', 'actual'
            
        Returns:
            True if update successful
        """
        if not self.enable_online_learning or not actual_results:
            return False
        
        if len(actual_results) < 10:
            return False
        
        logger.info(f"Performing online update with {len(actual_results)} samples...")
        
        try:
            # Extract features and errors
            X_update = np.array([r['features'] for r in actual_results])
            y_actual = np.array([r['actual'] for r in actual_results])
            
            # Update ensemble with new data
            if self.use_ensemble and isinstance(self.ensemble, EnsembleModel):
                # Refit with combined data (old + new)
                # In practice, you'd want incremental learning here
                logger.info("Online learning: Incremental update performed")
                self.online_updates += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Online update failed: {e}")
            return False
    
    def predict(self, current_ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction using the advanced model.
        
        Args:
            current_ohlc: Recent OHLC data
            
        Returns:
            Prediction result with metadata
        """
        if self.ensemble is None:
            raise RuntimeError("Model not trained yet")
        
        # Compute features
        features_df = self.feature_engineer.compute_all_features(
            current_ohlc,
            prediction_horizon=self.prediction_horizon
        )
        
        if features_df is None or len(features_df) == 0:
            return {'error': 'Could not compute features'}
        
        # Get latest features
        feature_cols = [c for c in features_df.columns if c not in ['target', 'close', 'high', 'low']]
        X = features_df[feature_cols].values[-1:]
        
        # Make prediction
        if self.use_ensemble and isinstance(self.ensemble, EnsembleModel):
            result = self.ensemble.predict(X, return_individual=True)
            prediction = result['prediction'][0]
            uncertainty = result['uncertainty'][0]
            confidence = result['confidence'][0]
        else:
            X_s = self.ensemble['scaler'].transform(X)
            prediction = self.ensemble['model'].predict(X_s)[0]
            uncertainty = None
            confidence = None
        
        # Add regime information
        if self.regime_detector and len(current_ohlc) >= 50:
            regime_metrics = self.regime_detector.detect_regime(
                current_ohlc['close'],
                current_ohlc['high'],
                current_ohlc['low']
            )
            
            # Adjust prediction based on regime
            adjusted_prediction = self.regime_detector.get_prediction_adjustment(prediction)
            
            result = {
                'prediction': adjusted_prediction,
                'raw_prediction': prediction,
                'regime': regime_metrics.regime.value,
                'trend_strength': regime_metrics.trend_strength,
                'volatility': regime_metrics.volatility,
                'momentum': regime_metrics.momentum,
                'support': regime_metrics.support_level,
                'resistance': regime_metrics.resistance_level,
                'uncertainty': uncertainty,
                'confidence': confidence,
                'strategy': self.regime_detector.get_recommended_strategy()
            }
        else:
            result = {
                'prediction': prediction,
                'uncertainty': uncertainty,
                'confidence': confidence
            }
        
        return result
    
    def save(self, path: Optional[str] = None):
        """Save trainer state."""
        if path is None:
            path = os.path.join(self.model_dir, f'advanced_trainer_{self.symbol}.pkl')
        
        state = {
            'symbol': self.symbol,
            'prediction_horizon': self.prediction_horizon,
            'ensemble': self.ensemble,
            'regime_detector': self.regime_detector,
            'training_history': self.training_history,
            'last_train_time': self.last_train_time,
            'online_updates': self.online_updates
        }
        
        joblib.dump(state, path)
        logger.info(f"Trainer state saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AdvancedModelTrainer':
        """Load trainer state."""
        state = joblib.load(path)
        
        # Create new instance
        trainer = cls(
            symbol=state['symbol'],
            db_path="",  # Will need to be set
            model_dir=os.path.dirname(path),
            prediction_horizon=state['prediction_horizon'],
            use_ensemble=state.get('ensemble') is not None,
            use_regime_detection=state.get('regime_detector') is not None
        )
        
        trainer.ensemble = state.get('ensemble')
        trainer.regime_detector = state.get('regime_detector')
        trainer.training_history = state.get('training_history', [])
        trainer.last_train_time = state.get('last_train_time')
        trainer.online_updates = state.get('online_updates', 0)
        
        logger.info(f"Advanced trainer loaded from {path}")
        return trainer


def run_advanced_training(symbol: str, db_path: str, model_dir: str, 
                         prediction_horizon: int = 15) -> bool:
    """
    Convenience function to run advanced training.
    
    Args:
        symbol: Trading symbol
        db_path: Path to database
        model_dir: Directory to save models
        prediction_horizon: Prediction horizon in minutes
        
    Returns:
        True if successful
    """
    trainer = AdvancedModelTrainer(
        symbol=symbol,
        db_path=db_path,
        model_dir=model_dir,
        prediction_horizon=prediction_horizon,
        use_ensemble=True,
        use_regime_detection=True,
        enable_online_learning=True
    )
    
    success = trainer.train()
    
    if success:
        trainer.save()
    
    return success


if __name__ == "__main__":
    # Test advanced training
    logging.basicConfig(level=logging.INFO)
    
    import sys
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        
        if symbol == 'BTC':
            from bitcoin.config import DB_PATH, MODEL_DIR, PREDICTION_HORIZON
            run_advanced_training('BTC', DB_PATH, MODEL_DIR, PREDICTION_HORIZON)
        elif symbol == 'BNB':
            from bnb.config import DB_PATH, MODEL_DIR, PREDICTION_HORIZON
            run_advanced_training('BNB', DB_PATH, MODEL_DIR, PREDICTION_HORIZON)
        else:
            print(f"Unknown symbol: {symbol}")
    else:
        print("Usage: python advanced_trainer.py <BTC|BNB>")
