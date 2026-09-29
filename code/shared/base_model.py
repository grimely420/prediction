#!/usr/bin/env python3
"""
Base model training class for cryptocurrency prediction.
"""

import os
import sys
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from typing import Any
from datetime import datetime

# Add parent directory for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging, get_db_connection


class BaseModelTrainer:
    """Base class for model training."""
    
    def __init__(self,
                 symbol: str,
                 db_path: str,
                 model_dir: str,
                 feature_engineer: Any,
                 min_train_points: int = 5000,
                 retrain_every_n_points: int = 10000,
                 prediction_horizon: int = 15):
        """
        Initialize the model trainer.
        
        Args:
            symbol: Cryptocurrency symbol
            db_path: Path to database
            model_dir: Directory to save models
            feature_engineer: Module with feature engineering
            min_train_points: Minimum points required for training
            retrain_every_n_points: Retrain after N new points
            prediction_horizon: Minutes ahead to predict
        """
        self.symbol = symbol
        self.db_path = db_path
        self.model_dir = model_dir
        self.feature_engineer = feature_engineer
        self.min_train_points = min_train_points
        self.retrain_every_n_points = retrain_every_n_points
        self.prediction_horizon = prediction_horizon
        
        # Setup logging
        self.logger = setup_logging(f"{symbol}-Model")
        
        # Track last training
        self.last_train_file = os.path.join(model_dir, ".last_train_count")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
    
    def get_price_count(self) -> int:
        """Get total number of price records."""
        try:
            conn = get_db_connection(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Failed to get price count: {e}")
            return 0
    
    def should_retrain(self) -> tuple:
        """Check if retraining is needed."""
        current_count = self.get_price_count()
        
        if not os.path.exists(self.last_train_file):
            return True, current_count
        
        try:
            with open(self.last_train_file, 'r') as f:
                last_count = int(f.read().strip())
            return (current_count - last_count) >= self.retrain_every_n_points, current_count
        except Exception:
            return True, current_count
    
    def save_last_train_count(self, count: int) -> None:
        """Save last training count."""
        with open(self.last_train_file, 'w') as f:
            f.write(str(count))
    
    def train_model(self) -> bool:
        """Train the XGBoost model."""
        try:
            self.logger.info(f"Training {self.symbol} XGBoost model...")
            
            # Get features
            X, y, feature_names = self.feature_engineer.compute_features()
            
            if X is None or len(X) < self.min_train_points:
                self.logger.warning(f"Insufficient data for training: {len(X) if X is not None else 0} points")
                return False
            
            # Time series cross validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    early_stopping_rounds=20,
                    eval_metric='mae'
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                mae = mean_absolute_error(y_val, preds)
                scores.append(mae)
            
            avg_mae = np.mean(scores)
            self.logger.info(f"Validation MAE: {avg_mae:.5f} ({avg_mae*100:.3f}%)")
            
            # Train final model on all data
            final_model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            final_model.fit(X, y)
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(self.model_dir, f"xgb_{self.prediction_horizon}min_{timestamp}.pkl")
            joblib.dump((final_model, feature_names), model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            # Update latest symlink
            latest_path = os.path.join(self.model_dir, f"xgb_{self.prediction_horizon}min_latest.pkl")
            joblib.dump((final_model, feature_names), latest_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def run(self) -> None:
        """Run the training process."""
        need_retrain, new_count = self.should_retrain()
        
        if need_retrain:
            if new_count < self.min_train_points:
                self.logger.info(f"Need {self.min_train_points} points for initial training. Have {new_count}.")
            else:
                success = self.train_model()
                if success:
                    self.save_last_train_count(new_count)
                    self.logger.info("Training completed successfully")
        else:
            self.logger.debug("No retrain needed")


if __name__ == "__main__":
    # This should be overridden by subclass
    pass
