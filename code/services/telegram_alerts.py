#!/usr/bin/env python3
"""Telegram alerter service entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.alerts import main

if __name__ == "__main__":
    main()
