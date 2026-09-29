#!/usr/bin/env python3
"""Generic multi-source price collector using shared config and data store."""

import os
import time
import signal
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from .coin_config import CoinConfig
from .data_store import DataStore
from .utils import setup_logging

logger = setup_logging("Collector")


class PriceCollector:
    """Collect prices from multiple sources for a single coin."""

    def __init__(self, coin_cfg: CoinConfig, data_store: DataStore):
        self.cfg = coin_cfg
        self.symbol = coin_cfg.symbol
        self.data_store = data_store
        self.running = True
        self.logger = setup_logging(f"{self.symbol}-Collector")
        self.stats = {
            'total': 0,
            'success': 0,
            'fail': 0,
            'api_stats': {},
        }
        self._setup_signals()
        self._init_api_stats()

    def _setup_signals(self) -> None:
        def handler(signum, frame):
            self.running = False
            self.logger.info("Shutdown signal received")
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _init_api_stats(self) -> None:
        for src in self.cfg.price_sources:
            self.stats['api_stats'][src['name']] = {
                'successes': 0,
                'failures': 0,
                'last_call': 0,
                'circuit_open': False,
                'last_failure': 0,
            }

    def _can_call(self, src: Dict[str, Any]) -> bool:
        stats = self.stats['api_stats'][src['name']]
        cooldown = src.get('cooldown', 1)
        if time.time() - stats['last_call'] < cooldown:
            return False
        if stats['circuit_open']:
            if time.time() - stats['last_failure'] > 60:
                stats['circuit_open'] = False
            else:
                return False
        return True

    def _record_failure(self, name: str) -> None:
        stats = self.stats['api_stats'][name]
        stats['failures'] += 1
        stats['last_failure'] = time.time()
        if stats['failures'] >= 5:
            stats['circuit_open'] = True
            self.logger.warning(f"Circuit opened for {name}")

    def _record_success(self, name: str) -> None:
        stats = self.stats['api_stats'][name]
        stats['successes'] += 1
        stats['failures'] = 0
        stats['last_call'] = time.time()

    def get_price(self) -> Tuple[Optional[float], Optional[str]]:
        sources = sorted(self.cfg.price_sources, key=lambda s: s.get('weight', 99))
        for src in sources:
            if not self._can_call(src):
                continue
            try:
                self.logger.debug(f"Trying {src['name']}")
                resp = requests.get(
                    src['url'],
                    headers=src.get('headers'),
                    timeout=src.get('timeout', 10),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    price = src['parser'](data)
                    if self.cfg.min_price <= price <= self.cfg.max_price:
                        self._record_success(src['name'])
                        return float(price), src['name']
                    self.logger.warning(f"Price out of range from {src['name']}: {price}")
                else:
                    self.logger.warning(f"HTTP {resp.status_code} from {src['name']}")
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout from {src['name']}")
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Parse error from {src['name']}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error from {src['name']}: {e}")
            self._record_failure(src['name'])
        return None, None

    def run(self) -> None:
        self.logger.info(f"{self.symbol} collector started (interval={self.cfg.collection_interval}s)")
        consecutive_failures = 0
        while self.running:
            try:
                price, source = self.get_price()
                self.stats['total'] += 1
                if price and source:
                    self.data_store.save_price(price, source)
                    self.stats['success'] += 1
                    consecutive_failures = 0
                    if self.stats['success'] % 12 == 0:
                        self.logger.info(f"{self.symbol}: ${price:,.4f} (source: {source}, total: {self.stats['success']})")
                else:
                    self.stats['fail'] += 1
                    consecutive_failures += 1
                    self.logger.warning(f"No price from any source (consecutive: {consecutive_failures})")
                    if consecutive_failures >= 12:
                        backoff = min(60, 5 * (2 ** (consecutive_failures // 12)))
                        self.logger.warning(f"Backing off for {backoff}s")
                        time.sleep(backoff)
                        consecutive_failures = 0
            except Exception as e:
                self.logger.error(f"Collector error: {e}")
                consecutive_failures += 1
            if self.running:
                time.sleep(self.cfg.collection_interval)
        self.logger.info(f"Collector stopped. Stats: {self.stats}")


def main() -> None:
    import sys
    coin_id = sys.argv[1] if len(sys.argv) > 1 else 'btc'
    from .coin_config import get_coin_config
    cfg = get_coin_config(coin_id)
    ds = DataStore(cfg.db_path, cfg.symbol)
    collector = PriceCollector(cfg, ds)
    collector.run()


if __name__ == '__main__':
    main()
