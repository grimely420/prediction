#!/usr/bin/env python3
"""
Comprehensive health monitoring system for cryptocurrency prediction services.
"""

import os
import sys
import sqlite3
import requests
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import threading

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared.utils import setup_logging

logger = setup_logging("Health-Monitor")


class ComponentHealth:
    """Represents the health status of a system component."""
    
    def __init__(self, name: str, healthy: bool, message: str = "", 
                 last_check: Optional[datetime] = None, metrics: Optional[Dict] = None):
        self.name = name
        self.healthy = healthy
        self.message = message
        self.last_check = last_check or datetime.now(timezone.utc)
        self.metrics = metrics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'healthy': self.healthy,
            'message': self.message,
            'last_check': self.last_check.isoformat(),
            'metrics': self.metrics
        }


class HealthMonitor:
    """Monitors the health of all prediction system components."""
    
    def __init__(self):
        self.components = {}
        self.btc_api_url = "http://localhost:5001"
        self.bnb_api_url = "http://localhost:5002"
        self.btc_db_path = os.path.join(BASE_DIR, "bitcoin", "bitcoin_prices.db")
        self.bnb_db_path = os.path.join(BASE_DIR, "bnb", "bnb_prices.db")
        self.running = False
        self.monitor_thread = None
    
    def check_btc_api(self) -> ComponentHealth:
        """Check Bitcoin API health."""
        try:
            response = requests.get(f"{self.btc_api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return ComponentHealth(
                    name="BTC-API",
                    healthy=True,
                    message=f"Running: {data.get('service', 'Bitcoin')}",
                    metrics={'response_time_ms': response.elapsed.total_seconds() * 1000}
                )
            else:
                return ComponentHealth(
                    name="BTC-API",
                    healthy=False,
                    message=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return ComponentHealth(
                name="BTC-API",
                healthy=False,
                message=f"Connection failed: {str(e)}"
            )
    
    def check_bnb_api(self) -> ComponentHealth:
        """Check BNB API health."""
        try:
            response = requests.get(f"{self.bnb_api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return ComponentHealth(
                    name="BNB-API",
                    healthy=True,
                    message=f"Running: {data.get('service', 'BNB')}",
                    metrics={'response_time_ms': response.elapsed.total_seconds() * 1000}
                )
            else:
                return ComponentHealth(
                    name="BNB-API",
                    healthy=False,
                    message=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return ComponentHealth(
                name="BNB-API",
                healthy=False,
                message=f"Connection failed: {str(e)}"
            )
    
    def check_btc_collector(self) -> ComponentHealth:
        """Check Bitcoin data collection."""
        try:
            conn = sqlite3.connect(self.btc_db_path)
            cursor = conn.cursor()
            
            # Get last price timestamp
            cursor.execute("SELECT timestamp FROM prices ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            if row:
                last_ts = datetime.fromisoformat(row[0])
                now = datetime.now(timezone.utc)
                time_diff = (now - last_ts).total_seconds()
                
                # Healthy if last data point is within 60 seconds
                if time_diff < 60:
                    return ComponentHealth(
                        name="BTC-Collector",
                        healthy=True,
                        message=f"Last data: {time_diff:.1f}s ago",
                        metrics={'last_data_seconds_ago': time_diff}
                    )
                else:
                    return ComponentHealth(
                        name="BTC-Collector",
                        healthy=False,
                        message=f"Stale data: {time_diff:.1f}s ago",
                        metrics={'last_data_seconds_ago': time_diff}
                    )
            else:
                return ComponentHealth(
                    name="BTC-Collector",
                    healthy=False,
                    message="No data in database"
                )
        except Exception as e:
            return ComponentHealth(
                name="BTC-Collector",
                healthy=False,
                message=f"Database error: {str(e)}"
            )
    
    def check_bnb_collector(self) -> ComponentHealth:
        """Check BNB data collection."""
        try:
            conn = sqlite3.connect(self.bnb_db_path)
            cursor = conn.cursor()
            
            # Get last price timestamp
            cursor.execute("SELECT timestamp FROM prices ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            if row:
                last_ts = datetime.fromisoformat(row[0])
                now = datetime.now(timezone.utc)
                time_diff = (now - last_ts).total_seconds()
                
                if time_diff < 60:
                    return ComponentHealth(
                        name="BNB-Collector",
                        healthy=True,
                        message=f"Last data: {time_diff:.1f}s ago",
                        metrics={'last_data_seconds_ago': time_diff}
                    )
                else:
                    return ComponentHealth(
                        name="BNB-Collector",
                        healthy=False,
                        message=f"Stale data: {time_diff:.1f}s ago",
                        metrics={'last_data_seconds_ago': time_diff}
                    )
            else:
                return ComponentHealth(
                    name="BNB-Collector",
                    healthy=False,
                    message="No data in database"
                )
        except Exception as e:
            return ComponentHealth(
                name="BNB-Collector",
                healthy=False,
                message=f"Database error: {str(e)}"
            )
    
    def check_prediction_accuracy(self, symbol: str, db_path: str) -> ComponentHealth:
        """Check prediction accuracy for a symbol."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get recent validated predictions
            cursor.execute("""
                SELECT COUNT(*), AVG(ABS(error)/predicted_price*100)
                FROM predictions 
                WHERE checked=1 AND actual_price IS NOT NULL 
                AND prediction_time > ?
            """, ((datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),))
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                count, avg_error_pct = row
                # Healthy if average error is less than 5%
                healthy = avg_error_pct < 5.0 if avg_error_pct else True
                
                return ComponentHealth(
                    name=f"{symbol}-Accuracy",
                    healthy=healthy,
                    message=f"{count} predictions, avg error: {avg_error_pct:.2f}%",
                    metrics={
                        'predictions_24h': count,
                        'avg_error_percent': avg_error_pct if avg_error_pct else 0
                    }
                )
            else:
                return ComponentHealth(
                    name=f"{symbol}-Accuracy",
                    healthy=True,  # Not enough data yet
                    message="No validated predictions in last 24h",
                    metrics={'predictions_24h': 0}
                )
        except Exception as e:
            return ComponentHealth(
                name=f"{symbol}-Accuracy",
                healthy=False,
                message=f"Accuracy check failed: {str(e)}"
            )
    
    def check_models(self) -> ComponentHealth:
        """Check if ML models exist and are recent."""
        btc_model_dir = os.path.join(BASE_DIR, "bitcoin", "models")
        bnb_model_dir = os.path.join(BASE_DIR, "bnb", "models")
        
        issues = []
        metrics = {}
        
        # Check BTC models
        btc_latest = os.path.join(btc_model_dir, "xgb_15min_latest.pkl")
        if os.path.exists(btc_latest):
            mtime = os.path.getmtime(btc_latest)
            age_hours = (time.time() - mtime) / 3600
            metrics['btc_model_age_hours'] = round(age_hours, 2)
            if age_hours > 48:  # Alert if model is older than 48 hours
                issues.append(f"BTC model is {age_hours:.1f}h old")
        else:
            issues.append("BTC model not found")
            metrics['btc_model_age_hours'] = None
        
        # Check BNB models
        bnb_latest = os.path.join(bnb_model_dir, "xgb_5min_latest.pkl")
        if os.path.exists(bnb_latest):
            mtime = os.path.getmtime(bnb_latest)
            age_hours = (time.time() - mtime) / 3600
            metrics['bnb_model_age_hours'] = round(age_hours, 2)
            if age_hours > 48:
                issues.append(f"BNB model is {age_hours:.1f}h old")
        else:
            issues.append("BNB model not found")
            metrics['bnb_model_age_hours'] = None
        
        if issues:
            return ComponentHealth(
                name="ML-Models",
                healthy=False,
                message="; ".join(issues),
                metrics=metrics
            )
        else:
            return ComponentHealth(
                name="ML-Models",
                healthy=True,
                message="Models up to date",
                metrics=metrics
            )
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check on all components."""
        checks = [
            self.check_btc_api(),
            self.check_bnb_api(),
            self.check_btc_collector(),
            self.check_bnb_collector(),
            self.check_prediction_accuracy("BTC", self.btc_db_path),
            self.check_prediction_accuracy("BNB", self.bnb_db_path),
            self.check_models()
        ]
        
        self.components = {c.name: c for c in checks}
        
        overall_healthy = all(c.healthy for c in checks)
        
        health_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'components': {name: comp.to_dict() for name, comp in self.components.items()}
        }
        
        # Log health status
        if overall_healthy:
            logger.info("Health check: All components healthy")
        else:
            unhealthy_components = [name for name, comp in self.components.items() if not comp.healthy]
            logger.warning(f"Health check failed for: {', '.join(unhealthy_components)}")
        
        return health_report
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring."""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    self.run_health_check()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Monitor loop error: {e}")
                    time.sleep(30)  # Retry after shorter interval
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Health monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.components:
            return self.run_health_check()
        
        overall_healthy = all(c.healthy for c in self.components.values())
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'components': {name: comp.to_dict() for name, comp in self.components.items()}
        }


def main():
    """Run health monitor as standalone service."""
    monitor = HealthMonitor()
    
    # Run initial health check
    print(json.dumps(monitor.run_health_check(), indent=2))
    
    # Start continuous monitoring
    monitor.start_monitoring(interval_seconds=60)
    
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\nHealth monitor stopped")


if __name__ == "__main__":
    main()
