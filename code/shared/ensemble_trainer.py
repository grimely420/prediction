#!/usr/bin/env python3
"""
Advanced ensemble learning with multiple algorithms and adaptive weighting.
"""

import os
import sys
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# ML libraries
try:
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging

logger = setup_logging("Ensemble-Trainer")


@dataclass
class ModelPerformance:
    """Track model performance for adaptive weighting."""
    name: str
    mae: float
    rmse: float
    r2: float
    recent_mae: float = 0.0
    weight: float = 1.0
    predictions_count: int = 0
    
    def update_weight(self, new_mae: float, alpha: float = 0.3):
        """Update weight based on recent performance using exponential decay."""
        self.recent_mae = alpha * new_mae + (1 - alpha) * self.recent_mae if self.recent_mae > 0 else new_mae
        # Lower MAE = higher weight
        self.weight = 1.0 / (1.0 + self.recent_mae)
        self.predictions_count += 1


class EnsembleModel:
    """
    Advanced ensemble model with multiple algorithms and adaptive weighting.
    """
    
    def __init__(self, 
                 use_xgb: bool = True,
                 use_lightgbm: bool = False,
                 use_catboost: bool = False,
                 use_sklearn_gb: bool = True,
                 use_random_forest: bool = True,
                 use_linear: bool = True,
                 adaptive_weighting: bool = True,
                 lookback_window: int = 50):
        """
        Initialize ensemble with selected models.
        
        Args:
            use_xgb: Use XGBoost
            use_lightgbm: Use LightGBM (if available)
            use_catboost: Use CatBoost (if available)
            use_sklearn_gb: Use sklearn GradientBoosting
            use_random_forest: Use Random Forest
            use_linear: Use linear models (Ridge, ElasticNet)
            adaptive_weighting: Enable adaptive ensemble weighting
            lookback_window: Window for adaptive weighting
        """
        self.models = {}
        self.scalers = {}
        self.performance = {}
        self.adaptive_weighting = adaptive_weighting
        self.lookback_window = lookback_window
        self.feature_names = None
        self.is_fitted = False
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for ensemble training")
        
        # Initialize models
        if use_xgb:
            self.models['xgboost'] = None  # Will be created during fit
        
        if use_sklearn_gb:
            self.models['sklearn_gb'] = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        
        if use_random_forest:
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        
        if use_linear:
            self.models['ridge'] = Ridge(alpha=1.0, random_state=42)
            self.models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        # Try optional libraries
        if use_lightgbm:
            try:
                import lightgbm as lgb
                self.models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                )
                logger.info("LightGBM available")
            except ImportError:
                logger.info("LightGBM not available")
        
        if use_catboost:
            try:
                from catboost import CatBoostRegressor
                self.models['catboost'] = CatBoostRegressor(
                    iterations=200,
                    depth=6,
                    learning_rate=0.05,
                    verbose=False,
                    random_seed=42
                )
                logger.info("CatBoost available")
            except ImportError:
                logger.info("CatBoost not available")
        
        logger.info(f"Ensemble initialized with {len(self.models)} models: {list(self.models.keys())}")
    
    def _create_xgb_model(self, X: np.ndarray, y: np.ndarray, 
                          validation_data: Optional[Tuple] = None) -> Any:
        """Create and train XGBoost model with early stopping."""
        # Calculate appropriate hyperparameters based on data size
        n_samples = len(X)
        
        # Scale hyperparameters based on data size
        n_estimators = min(500, max(100, n_samples // 50))
        max_depth = min(8, max(4, int(np.log2(n_samples) / 2)))
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=20 if validation_data else None,
            eval_metric='mae'
        )
        
        if validation_data:
            model.fit(X, y, eval_set=[validation_data], verbose=False)
        else:
            model.fit(X, y)
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            validation_split: float = 0.2) -> 'EnsembleModel':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Names of features
            validation_split: Fraction for validation
            
        Returns:
            self for method chaining
        """
        self.feature_names = feature_names
        n_samples = len(X)
        
        # Split for validation (time series split - last portion)
        split_idx = int(n_samples * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training ensemble on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        # Fit and evaluate each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Create scaler for this model
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                self.scalers[name] = scaler
                
                # Special handling for XGBoost
                if name == 'xgboost':
                    self.models[name] = self._create_xgb_model(
                        X_train_scaled, y_train, 
                        validation_data=(X_val_scaled, y_val)
                    )
                else:
                    self.models[name].fit(X_train_scaled, y_train)
                
                # Evaluate
                preds = self.models[name].predict(X_val_scaled)
                mae = mean_absolute_error(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                r2 = r2_score(y_val, preds)
                
                self.performance[name] = ModelPerformance(
                    name=name, mae=mae, rmse=rmse, r2=r2
                )
                
                logger.info(f"  {name}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                self.models[name] = None
        
        # Remove failed models
        self.models = {k: v for k, v in self.models.items() if v is not None}
        self.performance = {k: v for k, v in self.performance.items()}
        
        logger.info(f"Successfully trained {len(self.models)} models")
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray, return_individual: bool = False) -> Dict[str, Any]:
        """
        Make ensemble prediction.
        
        Args:
            X: Feature matrix
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dictionary with ensemble prediction and metadata
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted yet")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                X_scaled = self.scalers[name].transform(X)
                preds = model.predict(X_scaled)
                predictions[name] = preds
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        # Calculate ensemble prediction with adaptive weighting
        if self.adaptive_weighting and self.performance:
            weights = np.array([self.performance[name].weight for name in predictions.keys()])
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.ones(len(predictions)) / len(predictions)
        
        # Weighted average
        pred_matrix = np.column_stack(list(predictions.values()))
        ensemble_pred = np.average(pred_matrix, axis=1, weights=weights)
        
        # Calculate prediction uncertainty (std dev of predictions)
        pred_std = np.std(pred_matrix, axis=1)
        
        result = {
            'prediction': ensemble_pred,
            'uncertainty': pred_std,
            'weights': {name: w for name, w in zip(predictions.keys(), weights)},
            'confidence': self._calculate_confidence(ensemble_pred, pred_std)
        }
        
        if return_individual:
            result['individual_predictions'] = predictions
        
        return result
    
    def _calculate_confidence(self, prediction: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        """Calculate confidence score based on prediction and uncertainty."""
        # Higher uncertainty = lower confidence
        # Normalize uncertainty to [0, 1] range
        max_uncertainty = np.percentile(uncertainty, 95) if len(uncertainty) > 0 else 1.0
        normalized_uncertainty = np.clip(uncertainty / (max_uncertainty + 1e-10), 0, 1)
        confidence = 1.0 - normalized_uncertainty
        return confidence
    
    def update_performance(self, name: str, actual: float, predicted: float):
        """Update model performance after observing actual result."""
        if name in self.performance:
            error = abs(actual - predicted)
            self.performance[name].update_weight(error)
            logger.debug(f"Updated {name} weight to {self.performance[name].weight:.4f}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance across models."""
        if not self.is_fitted or not self.feature_names:
            return {}
        
        importance_scores = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    for feat_name, importance in zip(self.feature_names, model.feature_importances_):
                        importance_scores[feat_name] = importance_scores.get(feat_name, 0) + importance
            except:
                pass
        
        # Average across models
        if importance_scores:
            for key in importance_scores:
                importance_scores[key] /= len(self.models)
        
        # Sort by importance
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str):
        """Save ensemble to disk."""
        ensemble_data = {
            'models': self.models,
            'scalers': self.scalers,
            'performance': self.performance,
            'feature_names': self.feature_names,
            'adaptive_weighting': self.adaptive_weighting,
            'lookback_window': self.lookback_window,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(ensemble_data, path)
        logger.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load ensemble from disk."""
        ensemble_data = joblib.load(path)
        
        # Create new instance
        instance = cls(
            adaptive_weighting=ensemble_data.get('adaptive_weighting', True),
            lookback_window=ensemble_data.get('lookback_window', 50)
        )
        
        instance.models = ensemble_data['models']
        instance.scalers = ensemble_data['scalers']
        instance.performance = ensemble_data['performance']
        instance.feature_names = ensemble_data.get('feature_names')
        instance.is_fitted = ensemble_data.get('is_fitted', False)
        
        logger.info(f"Ensemble loaded from {path} (fitted: {instance.is_fitted})")
        return instance


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    Simulates real trading by training on past data and testing on future data.
    """
    
    def __init__(self, train_size: int = 1000, test_size: int = 100, step_size: int = 50):
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Number of samples for training
            test_size: Number of samples for testing
            step_size: How many samples to move forward each iteration
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.results = []
    
    def validate(self, X: np.ndarray, y: np.ndarray, 
                 model_factory: callable) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            X: Feature matrix
            y: Target values
            model_factory: Function that returns a new model instance
            
        Returns:
            Validation results summary
        """
        n_samples = len(X)
        
        # Determine number of folds
        available_for_walking = n_samples - self.train_size - self.test_size
        n_folds = max(1, available_for_walking // self.step_size + 1)
        
        logger.info(f"Walk-forward validation: {n_folds} folds")
        
        fold_results = []
        
        for fold in range(n_folds):
            # Define train/test split for this fold
            train_start = fold * self.step_size
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end >= n_samples:
                break
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Train model
            model = model_factory()
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                
                fold_results.append({
                    'fold': fold,
                    'train_range': (train_start, train_end),
                    'test_range': (test_start, test_end),
                    'mae': mae,
                    'rmse': rmse,
                    'preds_mean': np.mean(preds),
                    'actual_mean': np.mean(y_test)
                })
                
                logger.info(f"Fold {fold+1}/{n_folds}: MAE={mae:.6f}")
                
            except Exception as e:
                logger.error(f"Fold {fold} failed: {e}")
        
        # Aggregate results
        if fold_results:
            maes = [r['mae'] for r in fold_results]
            rmses = [r['rmse'] for r in fold_results]
            
            summary = {
                'n_folds': len(fold_results),
                'mean_mae': np.mean(maes),
                'std_mae': np.std(maes),
                'mean_rmse': np.mean(rmses),
                'std_rmse': np.std(rmses),
                'stability': np.std(maes) / (np.mean(maes) + 1e-10),  # Coefficient of variation
                'folds': fold_results
            }
            
            logger.info(f"Walk-forward validation complete:")
            logger.info(f"  Mean MAE: {summary['mean_mae']:.6f} ± {summary['std_mae']:.6f}")
            logger.info(f"  Stability (CV): {summary['stability']:.4f}")
            
            return summary
        else:
            return {'error': 'No successful folds'}


def train_ensemble_with_walk_forward(X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str],
                                     model_dir: str,
                                     symbol: str) -> EnsembleModel:
    """
    Train ensemble with walk-forward validation.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: Feature names
        model_dir: Directory to save model
        symbol: Symbol name
        
    Returns:
        Trained ensemble model
    """
    logger.info(f"Training ensemble for {symbol} with {len(X)} samples, {len(feature_names)} features")
    
    # First, perform walk-forward validation
    validator = WalkForwardValidator(train_size=min(2000, len(X)//3), 
                                     test_size=200, 
                                     step_size=100)
    
    def create_model():
        return EnsembleModel(use_xgb=True, use_sklearn_gb=True, 
                           use_random_forest=True, use_linear=True)
    
    # Run validation (optional, can be skipped for faster training)
    if len(X) > 3000:
        logger.info("Running walk-forward validation...")
        val_results = validator.validate(X, y, create_model)
        
        # If model is unstable, adjust hyperparameters
        if val_results.get('stability', 0) > 0.5:
            logger.warning("Model shows high variance, consider collecting more data or simplifying features")
    
    # Train final ensemble on all data
    logger.info("Training final ensemble...")
    ensemble = EnsembleModel(
        use_xgb=True,
        use_sklearn_gb=True,
        use_random_forest=True,
        use_linear=True,
        adaptive_weighting=True
    )
    
    ensemble.fit(X, y, feature_names=feature_names, validation_split=0.2)
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'ensemble_{symbol}_{timestamp}.pkl')
    ensemble.save(model_path)
    
    # Also save as latest
    latest_path = os.path.join(model_dir, f'ensemble_{symbol}_latest.pkl')
    ensemble.save(latest_path)
    
    # Print feature importance
    importance = ensemble.get_feature_importance()
    if importance:
        logger.info("Top 10 most important features:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble training
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 2000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    # Create target with some pattern
    y = np.sin(np.arange(n_samples) * 0.1) * 0.01 + np.random.randn(n_samples) * 0.005
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train ensemble
    ensemble = EnsembleModel(use_xgb=True, use_sklearn_gb=True, 
                            use_random_forest=True, use_linear=True)
    ensemble.fit(X, y, feature_names=feature_names)
    
    # Make predictions
    X_test = X[-10:]
    result = ensemble.predict(X_test, return_individual=True)
    
    print(f"\nPredictions on last 10 samples:")
    print(f"  Ensemble mean: {result['prediction']}")
    print(f"  Uncertainty: {result['uncertainty']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Model weights: {result['weights']}")
