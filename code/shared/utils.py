#!/usr/bin/env python3
"""
Shared utility functions for prediction systems.
"""

import os
import sqlite3
import logging
import logging.handlers
from datetime import datetime, timezone
from typing import Optional, Any

# Constants - use relative log directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging for a service.
    
    Args:
        name: Logger name
        log_file: Optional specific log file path
        
    Returns:
        Configured logger instance
    """
    if log_file is None:
        log_file = os.path.join(LOG_DIR, f"{name.lower().replace(' ', '_')}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler (rotates at 20 MB, keeps 5 generations)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=20 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a database connection with proper settings.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_table_schema(db_path: str, table_name: str = "predictions") -> None:
    """
    Ensure all required columns exist in the specified table.
    
    Args:
        db_path: Path to database
        table_name: Name of the table to check
    """
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Get existing columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns for predictions table
        if table_name == "predictions":
            if 'model_used' not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN model_used TEXT")
            if 'checked' not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN checked INTEGER DEFAULT 0")
            if 'actual_price' not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN actual_price REAL")
            if 'error' not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN error REAL")
            if 'is_correct' not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN is_correct INTEGER")
        
        # Add missing columns for prices table
        elif table_name == "prices":
            if 'source' not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN source TEXT")
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to ensure table schema: {e}")


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def calculate_percent_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (e.g., 0.05 = 5%)
    """
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def format_currency(amount: float) -> str:
    """Format a number as currency string."""
    return f"${amount:,.2f}"


class ModelNotReadyError(Exception):
    """Raised when the model is not ready to make predictions."""
    pass


class InsufficientDataError(Exception):
    """Raised when there is insufficient data for the requested operation."""
    pass
