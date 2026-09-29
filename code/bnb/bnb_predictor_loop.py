#!/usr/bin/env python3
"""
BNB prediction loop entry point.

Generates 5, 10, and 15-minute price predictions and validates them
against future prices stored in the SQLite database.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.predictor_loop import PredictorLoop


def main():
    PredictorLoop('bnb').run()


if __name__ == '__main__':
    main()
