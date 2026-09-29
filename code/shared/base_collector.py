#!/usr/bin/env python3
"""
Base collector class for cryptocurrency price collection.
"""

import os
import time
import sqlite3
import signal
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass

from .utils import setup_logging, get_db_connection, ensure_table_schema, get_timestamp


@dataclass
class APIEndpoint:
    """Represents a price API endpoint configuration."""
    name: str
    url: str
    parser: Callable
    timeout: int = 10
    weight: int = 1  # Lower weight = higher priority
    cooldown: int = 1  # Minimum seconds between calls


class BaseCollector(ABC):
    """
    Base class for cryptocurrency price collectors.
    Implements multi-API fallback, rate limiting, circuit breaker pattern.
    """
    
    def __init__(self, 
                 symbol: str, 
                 db_path: str, 
                 collection_interval: int = 5,
                 name: Optional[str] = None):
        """
        Initialize the collector.
        
        Args:
            symbol: Cryptocurrency symbol (BTC, BNB)
            db_path: Path to SQLite database
            collection_interval: Seconds between collections
            name: Service name for logging
        """
        self.symbol = symbol.upper()
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.name = name or f"{self.symbol}-Collector"
        self.running = True
        
        # Setup logging
        self.logger = setup_logging(self.name)
        
        # Statistics tracking
        self.stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'api_stats': {},
            'start_time': datetime.now(timezone.utc).isoformat()
        }
        
        # Initialize APIs (to be defined by subclass)
        self.apis: List[APIEndpoint] = []
        self._setup_apis()
        
        # Initialize API failure tracking
        for api in self.apis:
            self.stats['api_stats'][api.name] = {
                'failures': 0,
                'successes': 0,
                'last_failure': None,
                'circuit_open': False,
                'last_call': 0
            }
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize database
        self._init_database()
    
    @abstractmethod
    def _setup_apis(self) -> None:
        """
        Setup API endpoints for this cryptocurrency.
        Must be implemented by subclass.
        """
        pass
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            self.logger.info("Received shutdown signal, stopping collector...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            conn = get_db_connection(self.db_path)
            cursor = conn.cursor()
            
            # Create prices table
            cursor.execute('''CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prices(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_id ON prices(id)')
            
            # Create predictions table
            cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                error REAL,
                checked INTEGER DEFAULT 0,
                model_used TEXT,
                is_correct INTEGER,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')
            
            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            conn.close()
    
    def _respect_rate_limit(self, api: APIEndpoint) -> bool:
        """
        Check if we need to respect API rate limits.
        
        Args:
            api: API endpoint to check
            
        Returns:
            True if we can call the API, False if we should wait
        """
        stats = self.stats['api_stats'][api.name]
        time_since_last = time.time() - stats['last_call']
        
        if time_since_last < api.cooldown:
            return False
        
        stats['last_call'] = time.time()
        return True
    
    def get_price(self) -> Tuple[Optional[float], Optional[str]]:
        """
        Fetch current price from available APIs with circuit breaker pattern.
        
        Returns:
            Tuple of (price, source_name) or (None, None) if all APIs fail
        """
        # Sort APIs by weight (lower weight = higher priority)
        sorted_apis = sorted(self.apis, key=lambda x: x.weight)
        
        for api in sorted_apis:
            # Check rate limit
            if not self._respect_rate_limit(api):
                continue
            
            # Check circuit breaker
            stats = self.stats['api_stats'][api.name]
            if stats['circuit_open']:
                # Check if circuit should be closed (after 60 seconds)
                if stats['last_failure'] and (time.time() - stats['last_failure']) > 60:
                    stats['circuit_open'] = False
                    self.logger.info(f"Circuit closed for {api.name}, retrying...")
                else:
                    continue
            
            try:
                self.logger.debug(f"Attempting to fetch price from {api.name}")
                response = requests.get(api.url, timeout=api.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    price = api.parser(data)
                    
                    if price and isinstance(price, (int, float)) and price > 0:
                        # Success - update stats
                        stats['successes'] += 1
                        stats['failures'] = 0
                        self.logger.debug(f"Successfully got price from {api.name}: {self.symbol} ${price:.2f}")
                        return price, api.name
                    else:
                        self.logger.warning(f"Invalid price from {api.name}: {price}")
                else:
                    self.logger.warning(f"HTTP {response.status_code} from {api.name}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout from {api.name}")
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Connection error from {api.name}")
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Parse error from {api.name}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error from {api.name}: {e}")
            
            # Record failure
            stats['failures'] += 1
            stats['last_failure'] = time.time()
            
            # Open circuit if too many failures
            if stats['failures'] >= 5:
                stats['circuit_open'] = True
                self.logger.warning(f"Circuit opened for {api.name} due to {stats['failures']} failures")
        
        return None, None
    
    def save_price(self, price: float, source: str) -> bool:
        """
        Save price to database.
        
        Args:
            price: Current cryptocurrency price
            source: Source API name
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            conn = get_db_connection(self.db_path)
            cursor = conn.cursor()
            timestamp = get_timestamp()
            cursor.execute(
                "INSERT INTO prices (timestamp, price, source) VALUES (?, ?, ?)",
                (timestamp, price, source)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Failed to save price to database: {e}")
            return False
    
    def log_stats(self) -> None:
        """Log current statistics."""
        if self.stats['total_collections'] > 0:
            success_rate = (self.stats['successful_collections'] / self.stats['total_collections']) * 100
            self.logger.info(
                f"Stats: {self.stats['successful_collections']}/{self.stats['total_collections']} "
                f"({success_rate:.1f}% success rate)"
            )
    
    def run(self) -> None:
        """Main collection loop."""
        self.logger.info(f"{self.symbol} Collector started")
        self.logger.info(f"Collection interval: {self.collection_interval} seconds")
        self.logger.info(f"Database: {self.db_path}")
        
        while self.running:
            try:
                # Fetch price
                price, source = self.get_price()
                self.stats['total_collections'] += 1
                
                if price and source:
                    # Save to database
                    if self.save_price(price, source):
                        self.stats['successful_collections'] += 1
                        self.logger.info(
                            f"{self.symbol}: ${price:,.2f} (source: {source}, "
                            f"total: {self.stats['successful_collections']})"
                        )
                    else:
                        self.stats['failed_collections'] += 1
                        self.logger.error(f"Failed to save price: ${price:.2f}")
                else:
                    self.stats['failed_collections'] += 1
                    self.logger.warning("No price available from any API")
                
                # Log stats periodically
                if self.stats['total_collections'] % 100 == 0:
                    self.log_stats()
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(self.collection_interval)
        
        self.logger.info(f"Collector stopped. Final stats: {self.stats}")
