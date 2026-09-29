#!/usr/bin/env python3
"""
HYPE model training entry point.
Delegates to hype_model for backward compatibility with auto_retrain scripts.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from hype.hype_model import HYPEModelTrainer, main

if __name__ == "__main__":
    main()
