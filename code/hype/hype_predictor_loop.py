#!/usr/bin/env python3
"""
HYPE prediction loop entry point.

Generates 5, 10, and 15-minute price predictions and validates them
against future prices stored in the SQLite database.
"""

import os
import sys

# Add the prediction module root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.predictor_loop import PredictorLoop


def main():
    PredictorLoop('hype').run()


if __name__ == '__main__':
    main()
