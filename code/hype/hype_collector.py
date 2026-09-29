#!/usr/bin/env python3
"""HYPE price collector entry point."""

from shared.coin_config import get_coin_config
from shared.data_store import DataStore
from shared.collector import PriceCollector


def main():
    cfg = get_coin_config('hype')
    ds = DataStore(cfg.db_path, cfg.symbol)
    collector = PriceCollector(cfg, ds)
    collector.run()


if __name__ == '__main__':
    main()
