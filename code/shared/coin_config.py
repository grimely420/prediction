#!/usr/bin/env python3
"""
Central configuration for all supported prediction coins.
"""

import os
import base64
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Predictions logged before this (UTC) came from a pipeline that trained and
# predicted on stale June data, so they say nothing about the current models.
# Accuracy statistics exclude them.
STATS_SINCE = '2026-09-08T03:20:00+00:00'


def _parse_binance_price(data: dict) -> float:
    return float(data.get('price', 0))


def _parse_coinbase_price(data: dict) -> float:
    return float(data['data']['amount'])


def _parse_kraken_xxbtzusd(data: dict) -> float:
    return float(data['result']['XXBTZUSD']['c'][0])


def _parse_kraken_bnbusd(data: dict) -> float:
    return float(data['result']['BNBUSD']['c'][0])


def _parse_coingecko_hype(data: dict) -> float:
    return float(data['hyperliquid']['usd'])


def _parse_kraken_hypeusd(data: dict) -> float:
    return float(data['result']['HYPEUSD']['c'][0])


def _parse_mexc_price(data: dict) -> float:
    return float(data['price'])


def _parse_kraken_spot(pair_key: str) -> Callable[[dict], float]:
    """Factory: Kraken spot ticker parser for a given internal pair key."""
    def parse(data: dict) -> float:
        return float(data['result'][pair_key]['c'][0])
    return parse


def _generic_sources(symbol: str, pf_symbol: str, kraken_pair: str,
                     kraken_key: str, binance_sym: str, coinbase_pair: str,
                     gecko_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Standard source set for a Kalshi 15m coin: CFB RTI primary + spot fallbacks."""
    sources = [
        {
            'name': f'CFBenchmarks-{symbol}-RTI',
            'url': f'https://futures.kraken.com/derivatives/api/v3/tickers?symbol={pf_symbol}',
            'parser': _parse_kraken_futures_rti,
            'timeout': 6,
            'cooldown': 2,
            'weight': -2,
        },
        {
            'name': 'Kraken',
            'url': f'https://api.kraken.com/0/public/Ticker?pair={kraken_pair}',
            'parser': _parse_kraken_spot(kraken_key),
            'timeout': 10,
            'cooldown': 6,
            'weight': 1,
        },
        {
            'name': 'Coinbase',
            'url': f'https://api.coinbase.com/v2/prices/{coinbase_pair}/spot',
            'parser': _parse_coinbase_price,
            'timeout': 10,
            'cooldown': 6,
            'weight': 2,
        },
        {
            'name': 'Binance',
            'url': f'https://api.binance.com/api/v3/ticker/price?symbol={binance_sym}',
            'parser': _parse_binance_price,
            'timeout': 10,
            'cooldown': 6,
            'weight': 3,
        },
    ]
    if gecko_id:
        sources.append({
            'name': 'CoinGecko',
            'url': f'https://api.coingecko.com/api/v3/simple/price?ids={gecko_id}&vs_currencies=usd',
            'parser': lambda d, _g=gecko_id: float(d[_g]['usd']),
            'timeout': 15,
            'cooldown': 8,
            'weight': 4,
        })
    return sources


def _parse_kraken_futures_rti(data: dict) -> float:
    """Parse real-time CF Benchmarks index price (BRTI/RTI) from Kraken Futures.
    Kalshi settles against the 60-second TWAP of this exact index.
    """
    tickers = data.get('tickers', [])
    if not tickers:
        raise ValueError("Kraken Futures response missing tickers")
    t = tickers[0]
    for key in ('indexPrice', 'markPrice', 'last'):
        if t.get(key) is not None:
            val = float(t[key])
            if val > 0:
                return val
    raise ValueError(f"Unable to locate price in Kraken Futures response: {t}")


def _parse_cfbenchmarks_index(data: dict) -> float:
    """Parse current price from a CF Benchmarks REST index response.

    Response envelope: {"serverTime": "...", "payload": {...}}
    The payload may hold the current value under several known keys.
    """
    payload = data.get('payload') if isinstance(data, dict) else None
    if payload is None:
        raise ValueError("CF Benchmarks response missing payload")
    # /api/v1/indices returns a list in some contexts; take first item if so
    if isinstance(payload, list):
        payload = payload[0] if payload else None
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected CF Benchmarks payload: {data}")
    for key in ('value', 'price', 'last', 'lastPrice', 'last_price', 'indexValue', 'currentValue'):
        if key in payload and payload[key] is not None:
            return float(payload[key])
    raise ValueError(f"Unable to locate price in CF Benchmarks payload: {payload}")




def _alpaca_headers() -> Optional[Dict[str, str]]:
    """Build Alpaca API auth headers from environment variables."""
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    if api_key and secret_key:
        return {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    return None


def _parse_alpaca_quote(data: dict, pair: str) -> float:
    q = data.get('quotes', {}).get(pair)
    if q is None:
        raise ValueError(f"Alpaca quote missing for {pair}")
    ask = float(q.get('ap', 0) or 0)
    bid = float(q.get('bp', 0) or 0)
    if ask > 0 and bid > 0:
        return (ask + bid) / 2
    if ask > 0:
        return ask
    if bid > 0:
        return bid
    raise ValueError(f"Alpaca bid/ask missing for {pair}")


def _parse_alpaca_btc(data: dict) -> float:
    return _parse_alpaca_quote(data, "BTC/USD")


def _parse_alpaca_bnb(data: dict) -> float:
    return _parse_alpaca_quote(data, "BNB/USD")


def _cfbenchmarks_headers() -> Optional[Dict[str, str]]:
    """Build CF Benchmarks Basic Auth header from environment variables."""
    user = os.environ.get('CF_API_USERNAME') or os.environ.get('CFBENCHMARKS_API_USERNAME')
    password = os.environ.get('CF_API_PASSWORD') or os.environ.get('CFBENCHMARKS_API_PASSWORD')
    if user and password:
        credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}
    return None


@dataclass
class CoinConfig:
    """Configuration for a single coin."""
    symbol: str
    display_name: str
    db_name: str
    price_sources: List[Dict[str, Any]] = field(default_factory=list)
    min_price: float = 0.0
    max_price: float = 1e12
    collection_interval: int = 5
    prediction_horizons: List[int] = field(default_factory=lambda: [5, 10, 15])
    min_train_bars: int = 120
    retrain_min_bars: int = 360
    train_lookback_hours: int = 24 * 60
    prediction_threshold_pct: float = 1.0
    api_port: int = 5000
    # Merge Hyperliquid market data (volume, funding, book) into features.
    hl_features: bool = False
    hl_micro: bool = False
    # Round-trip cost used by the fee-aware classifier (percent).
    fee_pct: float = 0.05

    @property
    def db_path(self) -> str:
        return os.path.join(BASE_DIR, self.db_name, f"{self.db_name}_prices.db")

    @property
    def model_dir(self) -> str:
        return os.path.join(BASE_DIR, self.db_name, "models")

    def ensure_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


def _btc_sources() -> List[Dict[str, Any]]:
    sources = [
        {
            'name': 'CFBenchmarks-BRTI',
            'url': 'https://futures.kraken.com/derivatives/api/v3/tickers?symbol=PF_XBTUSD',
            'parser': _parse_kraken_futures_rti,
            'timeout': 6,
            'cooldown': 2,
            'weight': -2,
        },
        {
            'name': 'Alpaca',
            'url': 'https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes?symbols=BTC%2FUSD',
            'parser': _parse_alpaca_btc,
            'timeout': 10,
            'cooldown': 1,
            'weight': -1,
            'headers': _alpaca_headers(),
        },
        {
            'name': 'Coinbase',
            'url': 'https://api.coinbase.com/v2/prices/BTC-USD/spot',
            'parser': _parse_coinbase_price,
            'timeout': 10,
            'cooldown': 6,
            'weight': 1,
        },
        {
            'name': 'Kraken',
            'url': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
            'parser': _parse_kraken_xxbtzusd,
            'timeout': 10,
            'cooldown': 6,
            'weight': 2,
        },
        {
            'name': 'Binance',
            'url': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
            'parser': _parse_binance_price,
            'timeout': 10,
            'cooldown': 6,
            'weight': 5,
        },
    ]
    return sources


def _bnb_sources() -> List[Dict[str, Any]]:
    sources = [
        {
            'name': 'CFBenchmarks-BNB-RTI',
            'url': 'https://futures.kraken.com/derivatives/api/v3/tickers?symbol=PF_BNBUSD',
            'parser': _parse_kraken_futures_rti,
            'timeout': 6,
            'cooldown': 2,
            'weight': -2,
        },
        {
            'name': 'Alpaca',
            'url': 'https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes?symbols=BNB%2FUSD',
            'parser': _parse_alpaca_bnb,
            'timeout': 10,
            'cooldown': 1,
            'weight': -1,
            'headers': _alpaca_headers(),
        },
        {
            'name': 'Binance',
            'url': 'https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT',
            'parser': _parse_binance_price,
            'timeout': 10,
            'cooldown': 6,
            'weight': 1,
        },
        {
            'name': 'Coinbase',
            'url': 'https://api.coinbase.com/v2/prices/BNB-USD/spot',
            'parser': _parse_coinbase_price,
            'timeout': 10,
            'cooldown': 6,
            'weight': 2,
        },
        {
            'name': 'Kraken',
            'url': 'https://api.kraken.com/0/public/Ticker?pair=BNBUSD',
            'parser': _parse_kraken_bnbusd,
            'timeout': 10,
            'cooldown': 6,
            'weight': 3,
        },
    ]
    return sources


def _hype_sources() -> List[Dict[str, Any]]:
    sources = [
        {
            'name': 'CFBenchmarks-HYPE-RTI',
            'url': 'https://futures.kraken.com/derivatives/api/v3/tickers?symbol=PF_HYPEUSD',
            'parser': _parse_kraken_futures_rti,
            'timeout': 6,
            'cooldown': 2,
            'weight': -2,
        },
        {
            'name': 'Kraken',
            'url': 'https://api.kraken.com/0/public/Ticker?pair=HYPEUSD',
            'parser': _parse_kraken_hypeusd,
            'timeout': 10,
            'cooldown': 6,
            'weight': 1,
        },
        {
            'name': 'CoinGecko',
            'url': 'https://api.coingecko.com/api/v3/simple/price?ids=hyperliquid&vs_currencies=usd',
            'parser': _parse_coingecko_hype,
            'timeout': 15,
            'cooldown': 8,
            'weight': 2,
        },
        {
            'name': 'MEXC',
            'url': 'https://api.mexc.com/api/v3/ticker/price?symbol=HYPEUSDT',
            'parser': _parse_mexc_price,
            'timeout': 10,
            'cooldown': 8,
            'weight': 3,
        },
    ]
    return sources


COINS: Dict[str, CoinConfig] = {
    'btc': CoinConfig(
        symbol='BTC',
        display_name='Bitcoin',
        db_name='bitcoin',
        price_sources=_btc_sources(),
        min_price=1000.0,
        max_price=1_000_000.0,
        collection_interval=2,
        prediction_horizons=[5, 10, 15],
        min_train_bars=120,
        retrain_min_bars=60,
        prediction_threshold_pct=1.0,
    ),
    'bnb': CoinConfig(
        symbol='BNB',
        display_name='BNB',
        db_name='bnb',
        price_sources=_bnb_sources(),
        min_price=10.0,
        max_price=100_000.0,
        collection_interval=2,
        prediction_horizons=[5, 10, 15],
        min_train_bars=120,
        retrain_min_bars=60,
        prediction_threshold_pct=1.0,
    ),
    'hype': CoinConfig(
        symbol='HYPE',
        display_name='HYPE',
        db_name='hype',
        price_sources=_hype_sources(),
        min_price=0.01,
        max_price=1_000_000.0,
        collection_interval=2,
        prediction_horizons=[5, 10, 15],
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=2.0,
        hl_features=True,
        hl_micro=True,
    ),
    'eth': CoinConfig(
        symbol='ETH',
        display_name='Ethereum',
        db_name='eth',
        price_sources=_generic_sources(
            'ETH', 'PF_ETHUSD', 'ETHUSD', 'XETHZUSD', 'ETHUSDT', 'ETH-USD'),
        min_price=100.0,
        max_price=100_000.0,
        collection_interval=2,
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=1.0,
    ),
    'sol': CoinConfig(
        symbol='SOL',
        display_name='Solana',
        db_name='sol',
        price_sources=_generic_sources(
            'SOL', 'PF_SOLUSD', 'SOLUSD', 'SOLUSD', 'SOLUSDT', 'SOL-USD'),
        min_price=1.0,
        max_price=10_000.0,
        collection_interval=2,
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=1.0,
    ),
    'xrp': CoinConfig(
        symbol='XRP',
        display_name='XRP',
        db_name='xrp',
        price_sources=_generic_sources(
            'XRP', 'PF_XRPUSD', 'XRPUSD', 'XXRPZUSD', 'XRPUSDT', 'XRP-USD'),
        min_price=0.05,
        max_price=1_000.0,
        collection_interval=2,
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=1.5,
    ),
    'doge': CoinConfig(
        symbol='DOGE',
        display_name='Dogecoin',
        db_name='doge',
        price_sources=_generic_sources(
            'DOGE', 'PF_DOGEUSD', 'DOGEUSD', 'XDGUSD', 'DOGEUSDT', 'DOGE-USD'),
        min_price=0.005,
        max_price=100.0,
        collection_interval=2,
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=1.5,
    ),
    'near': CoinConfig(
        symbol='NEAR',
        display_name='NEAR Protocol',
        db_name='near',
        price_sources=_generic_sources(
            'NEAR', 'PF_NEARUSD', 'NEARUSD', 'NEARUSD', 'NEARUSDT', 'NEAR-USD'),
        min_price=0.1,
        max_price=1_000.0,
        collection_interval=2,
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=1.5,
    ),
    'zec': CoinConfig(
        symbol='ZEC',
        display_name='Zcash',
        db_name='zec',
        price_sources=_generic_sources(
            'ZEC', 'PF_ZECUSD', 'ZECUSD', 'XZECZUSD', 'ZECUSDT', 'ZEC-USD'),
        min_price=10.0,
        max_price=100_000.0,
        collection_interval=2,
        min_train_bars=60,
        retrain_min_bars=30,
        prediction_threshold_pct=1.5,
    ),
}


def get_coin_config(coin_id: str) -> CoinConfig:
    """Return configuration for a coin id (case-insensitive)."""
    key = coin_id.lower()
    if key not in COINS:
        raise ValueError(f"Unknown coin: {coin_id}. Supported: {list(COINS.keys())}")
    cfg = COINS[key]
    cfg.ensure_dirs()
    return cfg


def list_coins() -> List[str]:
    return list(COINS.keys())
