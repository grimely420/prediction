#!/usr/bin/env python3
"""Hyperliquid collector: book/funding/OI snapshots, trade tape, and 1m candles.

    python -m shared.hl_collector [COIN] [--coin-dir hype] [--interval 5]
"""

import argparse
import signal
import time

from .hyperliquid import HLClient, HLStore, hl_db_path, snapshot_row
from .utils import setup_logging


class HLCollector:
    def __init__(self, coin: str, coin_dir: str, interval: int):
        self.coin = coin
        self.interval = interval
        self.client = HLClient()
        self.store = HLStore(hl_db_path(coin_dir))
        self.logger = setup_logging(f"{coin}-HL-Collector")
        self.running = True
        self.failures = 0
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        self.running = False
        self.logger.info("Shutdown signal received")

    def _sync_candles(self):
        """Pull the last couple of hours of 1m candles (idempotent upsert)."""
        now_ms = int(time.time() * 1000)
        last = self.store.last_candle_ms() or (now_ms - 3 * 3600_000)
        rows = self.client.candles(self.coin, max(last - 120_000, now_ms - 3 * 3600_000), now_ms)
        self.store.save_candles(rows)

    def _sync_funding(self):
        now_ms = int(time.time() * 1000)
        last = self.store.last_funding_ms() or (now_ms - 48 * 3600_000)
        rows = self.client.funding_history(self.coin, last + 1, now_ms)
        if rows:
            self.store.save_funding(rows)

    def run(self):
        self.logger.info(f"{self.coin} Hyperliquid collector started (interval {self.interval}s)")
        n, last_candle, last_funding = 0, 0, 0
        while self.running:
            t0 = time.time()
            try:
                ctx = self.client.asset_ctx(self.coin)
                book = self.client.l2_book(self.coin)
                row = snapshot_row(ctx, book)
                self.store.save_snapshot(row)
                new_trades = self.store.save_trades(self.client.recent_trades(self.coin))
                self.failures = 0
                n += 1
                if n % 60 == 0:
                    self.logger.info(
                        f"{self.coin}: mark {row['mark_px']:.4f} funding {row['funding']:+.7f} "
                        f"OI {row['open_interest']:,.0f} spread {row.get('spread_bps', 0):.2f}bps "
                        f"imb5 {row.get('imbalance_5', 0):+.3f} (+{new_trades} trades, {n} snaps)")
                if t0 - last_candle > 60:
                    self._sync_candles()
                    last_candle = t0
                if t0 - last_funding > 900:
                    self._sync_funding()
                    last_funding = t0
            except Exception as e:
                self.failures += 1
                self.logger.warning(f"{self.coin} collect error ({self.failures}): {e}")
                if self.failures >= 5:
                    time.sleep(min(60, 5 * self.failures))
            time.sleep(max(0.5, self.interval - (time.time() - t0)))
        self.logger.info(f"{self.coin} Hyperliquid collector stopped")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("coin", nargs="?", default="HYPE")
    p.add_argument("--coin-dir", default=None)
    p.add_argument("--interval", type=int, default=5)
    a = p.parse_args()
    HLCollector(a.coin.upper(), a.coin_dir or a.coin.lower(), a.interval).run()


if __name__ == "__main__":
    main()
