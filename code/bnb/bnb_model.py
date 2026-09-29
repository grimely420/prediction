#!/usr/bin/env python3
"""
BNB model training script.
"""

import os
import sys

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.base_model import BaseModelTrainer
from bnb.config import DB_PATH, MODEL_DIR, MIN_TRAIN_POINTS, RETRAIN_EVERY_N_POINTS, SYMBOL, PREDICTION_HORIZON
from bnb import engine as bnb_engine


class BNBModelTrainer(BaseModelTrainer):
    """BNB-specific model trainer."""
    
    def __init__(self):
        super().__init__(
            symbol=SYMBOL,
            db_path=DB_PATH,
            model_dir=MODEL_DIR,
            feature_engineer=bnb_engine,
            min_train_points=MIN_TRAIN_POINTS,
            retrain_every_n_points=RETRAIN_EVERY_N_POINTS,
            prediction_horizon=PREDICTION_HORIZON
        )


def main():
    """Main entry point."""
    trainer = BNBModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
