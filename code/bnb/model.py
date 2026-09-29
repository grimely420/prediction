#!/usr/bin/env python3
"""
BNB model training entry point.
Delegates to bnb_model for backward compatibility with auto_retrain scripts.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bnb.bnb_model import BNBModelTrainer, main

if __name__ == "__main__":
    main()
