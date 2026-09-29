#!/usr/bin/env python3
"""
Feature engineering shared by all coins and prediction horizons.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, List, Dict, Any

from .utils import setup_logging

logger = setup_logging("FeatureEngine")


def to_dataframe(records: List[Dict[str, Any]],
                 single_source: bool = False) -> Optional[pd.DataFrame]:
    """Convert raw price records into a continuous DataFrame.
    Prioritizes official CF Benchmarks and primary feeds without dropping
    entire months of historical data from other sources.
    """
    if not records:
        return None
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='ISO8601')
    if 'source' in df.columns:
        source_rank = {
            'Alpaca': 3,
            'Coinbase': 4,
            'Kraken': 5,
            'Binance': 6,
            'MEXC': 7,
            'CoinGecko': 8
        }
        df['rank'] = df['source'].map(
            lambda s: 1 if str(s).startswith('CFBenchmarks') else source_rank.get(s, 10))
        df = df.sort_values(['timestamp', 'rank']).drop_duplicates(subset=['timestamp'], keep='first')
        df = df.drop(columns=['rank'])
    else:
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    df = df.set_index('timestamp')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    return df


def resample_ohlc(df: pd.DataFrame, freq: str = '1min') -> Optional[pd.DataFrame]:
    """Resample tick price data into OHLC bars."""
    if df is None or len(df) < 2:
        return None
    try:
        ohlc = df.resample(freq, closed='left', label='left').agg({
            'price': ['first', 'max', 'min', 'last']
        })
        ohlc.columns = ['open', 'high', 'low', 'close']
        return ohlc.dropna()
    except Exception as e:
        logger.error(f"resample error: {e}")
        return None


class FeatureEngine:
    """Compute a stable feature set for any coin / horizon."""

    def __init__(self, data_store, symbol: str):
        self.data_store = data_store
        self.symbol = symbol
        # Optional callable(bars) -> DataFrame aligned to bars.index with
        # extra feature columns (e.g. Hyperliquid market data). None = off.
        self.extras_fn = None
        # Optional callable(bars) -> DataFrame of the leader asset's 1m bars
        # (BTC). Used for cross-asset lead-lag features on BNB/HYPE.
        self.cross_bars_fn = None
        self._bars_cache: Optional[pd.DataFrame] = None
        self._bars_cache_time: float = 0.0
        self._live_bars_count: int = 0
        self._news_cache: Optional[pd.DataFrame] = None
        self._news_cache_time: float = 0.0
        # Base-feature frame cache: the heavy rolling-indicator computation is
        # identical across horizons (only the target column differs), so it is
        # shared. Keyed on the bar set identity incl. the latest close.
        self._base_feats_cache: Optional[Tuple[Any, pd.DataFrame]] = None
        # External extras (Hyperliquid) are minute-aggregated; recompute at most
        # once per new minute bar rather than on every request.
        self._extras_cache_key: Any = None
        self._extras_cache: Optional[pd.DataFrame] = None

    def _base_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        key = (bars.index[0], bars.index[-1], len(bars), float(bars['close'].iloc[-1]))
        if self._base_feats_cache is not None and self._base_feats_cache[0] == key:
            return self._base_feats_cache[1]
        feats = self._compute_base_features(bars)
        self._base_feats_cache = (key, feats)
        return feats

    def _compute_base_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        close = bars['close']
        high = bars['high']
        low = bars['low']
        returns = close.pct_change()

        feats = pd.DataFrame(index=bars.index)
        feats['return_1'] = returns

        for w in [3, 5, 10, 15]:
            feats[f'return_{w}'] = close.pct_change(w)

        for w in [3, 5, 10]:
            feats[f'momentum_{w}'] = close - close.shift(w)

        for w in [5, 10, 20]:
            feats[f'volatility_{w}'] = returns.rolling(w, min_periods=3).std()

        for w in [5, 10, 20]:
            ma = close.rolling(w, min_periods=3).mean()
            feats[f'ma_dist_{w}'] = (close - ma) / (ma + 1e-10)

        feats['hl_range_pct'] = (high - low) / (close + 1e-10)
        feats['price_position'] = (close - low) / (high - low + 1e-10)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
        rs = gain / (loss + 1e-10)
        feats['rsi_14'] = 100 - 100 / (1 + rs)

        for w in [30, 60]:
            feats[f'return_{w}'] = close.pct_change(w)
            feats[f'volatility_{w}'] = returns.rolling(w, min_periods=10).std()
        feats['vol_ratio_5_60'] = feats['volatility_5'] / (feats['volatility_60'] + 1e-10)

        # EMA Trend Alignment (Fast & Medium term)
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        feats['ema_diff_9_21'] = (ema_9 - ema_21) / (close + 1e-10)
        feats['ema_diff_21_50'] = (ema_21 - ema_50) / (close + 1e-10)
        feats['price_above_ema50'] = (close > ema_50).astype(float)

        # MACD (Moving Average Convergence Divergence) & Acceleration
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = (ema_12 - ema_26) / (close + 1e-10)
        macd_signal = macd_line.ewm(span=9).mean()
        macd_hist = macd_line - macd_signal
        feats['macd_line'] = macd_line
        feats['macd_hist'] = macd_hist
        feats['macd_hist_slope'] = macd_hist - macd_hist.shift(2)

        # Bollinger Bands (%B and Bandwidth squeeze)
        bb_ma20 = close.rolling(20, min_periods=5).mean()
        bb_std20 = close.rolling(20, min_periods=5).std()
        bb_upper = bb_ma20 + 2.0 * bb_std20
        bb_lower = bb_ma20 - 2.0 * bb_std20
        feats['bb_pct_b'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        feats['bb_bandwidth'] = (bb_upper - bb_lower) / (close + 1e-10)

        # Donchian 15-Minute Support/Resistance Channel
        high_15 = high.rolling(15, min_periods=5).max()
        low_15 = low.rolling(15, min_periods=5).min()
        feats['donchian_pos_15'] = (close - low_15) / (high_15 - low_15 + 1e-10)
        feats['breakout_high_15'] = (close >= high_15.shift(1)).astype(float)
        feats['breakout_low_15'] = (close <= low_15.shift(1)).astype(float)

        # Stochastic Oscillator %K and %D (14-period)
        low_14 = low.rolling(14, min_periods=5).min()
        high_14 = high.rolling(14, min_periods=5).max()
        stoch_k = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        feats['stoch_k'] = stoch_k
        feats['stoch_d'] = stoch_k.rolling(3, min_periods=1).mean()

        # Trend regime: ADX-14, directional index spread, normalized ATR
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=bars.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=bars.index)
        tr = pd.concat([(high - low),
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1.0 / 14, min_periods=5).mean()
        feats['atr_pct'] = atr_14 / (close + 1e-10)
        plus_di = 100 * plus_dm.ewm(alpha=1.0 / 14, min_periods=5).mean() / (atr_14 + 1e-10)
        minus_di = 100 * minus_dm.ewm(alpha=1.0 / 14, min_periods=5).mean() / (atr_14 + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        feats['adx_14'] = dx.ewm(alpha=1.0 / 14, min_periods=5).mean()
        feats['di_diff'] = (plus_di - minus_di) / 100.0

        # In-round return: how far price has already travelled since round open.
        # Strongly predictive for Kalshi contract settlement.
        round_id_series = pd.Series(bars.index.floor('15min'), index=bars.index)
        round_open_px = bars.groupby(round_id_series)['open'].transform('first')
        feats['dist_to_strike'] = np.log(close / (round_open_px + 1e-10))
        feats['mins_remaining'] = (14 - (bars.index.minute % 15)).astype(float)
        feats['mins_elapsed'] = (bars.index.minute % 15).astype(float)

        for lag in [1, 5, 15]:
            feats[f'autocorr_{lag}'] = returns.rolling(60, min_periods=30).corr(returns.shift(lag))

        # Volatility regime flags using only past data (no future leak)
        vol_60 = feats['volatility_60']
        vol_high = vol_60.rolling(1000, min_periods=200).quantile(0.75)
        vol_low = vol_60.rolling(1000, min_periods=200).quantile(0.25)
        feats['regime_low'] = (vol_60 < vol_low).astype(float)
        feats['regime_high'] = (vol_60 > vol_high).astype(float)

        # Time cyclicals & Kalshi 15-minute round alignment
        hour = bars.index.hour + bars.index.minute / 60.0
        feats['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        feats['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        dow = bars.index.dayofweek
        feats['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        feats['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # 15-minute Kalshi round cycle phase (0 to 14 minutes)
        min_in_15 = bars.index.minute % 15
        feats['round_15m_phase'] = min_in_15 / 15.0
        feats['round_15m_sin'] = np.sin(2 * np.pi * min_in_15 / 15.0)
        feats['round_15m_cos'] = np.cos(2 * np.pi * min_in_15 / 15.0)

        # News features joined by bar timestamp from the logged news series.
        # Previously every bar in the frame got the *live* value — a constant
        # the model could never learn from. merge_asof attaches each bar to the
        # most recent logged sample within 30 min; older bars (pre-logging)
        # get neutral zeros.
        news = self._news_series()
        if news is not None and len(news) > 0:
            try:
                feats = feats.reset_index()
                ts_col = feats.columns[0]
                feats = feats.rename(columns={ts_col: 'timestamp'})
                feats = feats.sort_values('timestamp')
                feats = pd.merge_asof(
                    feats, news, on='timestamp',
                    direction='backward', tolerance=pd.Timedelta('30min'))
                feats = feats.set_index('timestamp').sort_index()
                for c in ('news_sentiment', 'news_global', 'news_velocity'):
                    if c not in feats.columns:
                        feats[c] = 0.0
                feats[['news_sentiment', 'news_global', 'news_velocity']] = \
                    feats[['news_sentiment', 'news_global', 'news_velocity']].fillna(0.0)
            except Exception as e:
                logger.warning(f"news join failed: {e}")
                feats['news_sentiment'] = feats.get('news_sentiment', 0.0)
                feats['news_global'] = feats.get('news_global', 0.0)
                feats['news_velocity'] = feats.get('news_velocity', 0.0)
        else:
            feats['news_sentiment'] = 0.0
            feats['news_global'] = 0.0
            feats['news_velocity'] = 0.0

        return feats

    def _build_features(self, bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
        feats = self._base_features(bars).copy()
        close = bars['close']
        feats['target'] = (close.shift(-horizon) - close) / (close + 1e-10)
        return feats

    def _news_series(self) -> Optional[pd.DataFrame]:
        """Logged news features as a timestamped frame (cached ~60s)."""
        now = time.time()
        if self._news_cache is not None and (now - self._news_cache_time) < 60.0:
            return self._news_cache
        try:
            rows = self.data_store.get_news_features()
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='ISO8601')
            df = df.rename(columns={
                'compound': 'news_sentiment',
                'global_sentiment': 'news_global',
                'velocity_15m': 'news_velocity',
            })[['timestamp', 'news_sentiment', 'news_global', 'news_velocity']]
            df = df.sort_values('timestamp')
            self._news_cache = df
            self._news_cache_time = now
            return df
        except Exception as e:
            logger.warning(f"news series load failed: {e}")
            return None

    def build_bars(self, since: Optional[str] = None, until: Optional[str] = None,
                   limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetch ticks in a window and resample to 1-minute OHLC bars."""
        # Short cache for concurrent prediction requests without specific time range
        if since is None and until is None and (time.time() - self._bars_cache_time < 4.0) and self._bars_cache is not None:
            return self._bars_cache.copy()

        records = self.data_store.get_prices(limit=limit, since=since, until=until)
        bars = resample_ohlc(to_dataframe(records), freq='1min')
        if since is None and until is None and bars is not None:
            self._bars_cache = bars.copy()
            self._bars_cache_time = time.time()
        return bars

    def _join_extras(self, feats: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        """Join external feature blocks (Hyperliquid extras, BTC cross-asset).

        Missing external values are zero-filled rather than row-dropped so that
        periods without coverage (e.g. before the HL micro collector existed)
        still contribute training rows for the base features. Zero is a neutral
        value for imbalance/flow/basis style features.
        """
        if self.extras_fn is not None:
            try:
                # extras are minute-resampled upstream; one reload per new bar
                if self._extras_cache_key != bars.index[-1]:
                    extra = self.extras_fn(bars)
                    self._extras_cache = extra.fillna(0.0) \
                        if extra is not None and not extra.empty else None
                    self._extras_cache_key = bars.index[-1]
                if self._extras_cache is not None:
                    feats = feats.join(
                        self._extras_cache.reindex(feats.index).fillna(0.0))
            except Exception as e:
                logger.warning(f"extras loader failed: {e}")

        if self.cross_bars_fn is not None and self.symbol != 'BTC':
            try:
                xb = self.cross_bars_fn(bars)
                if xb is not None and not xb.empty:
                    xc = xb['close']
                    xr = xc.pct_change()
                    x = pd.DataFrame(index=xb.index)
                    x['btc_ret_1'] = xr
                    x['btc_ret_3'] = xc.pct_change(3)
                    x['btc_ret_5'] = xc.pct_change(5)
                    x['btc_ret_15'] = xc.pct_change(15)
                    x['btc_mom_5'] = (xc - xc.shift(5)) / (xc.shift(5) + 1e-10)
                    x['btc_vol_10'] = xr.rolling(10, min_periods=3).std()
                    x = x.reindex(feats.index).ffill(limit=5).fillna(0.0)
                    feats = feats.join(x)
            except Exception as e:
                logger.warning(f"cross-asset features failed: {e}")
        return feats

    def features_from_bars(self, bars: pd.DataFrame, horizon: int):
        """Return (feature_frame, feature_cols) with NaN feature rows dropped."""
        feats = self._build_features(bars, horizon)
        feats = self._join_extras(feats, bars)
        feature_cols = [c for c in feats.columns if c != 'target']
        return feats.dropna(subset=feature_cols), feature_cols

    def contract_frame_from_bars(self, bars: pd.DataFrame,
                                 for_training: bool = True):
        """Build the Kalshi 15-minute contract dataset.

        For every 1-minute bar inside a quarter-hour round (:00/:15/:30/:45):
          features = market features + dist_to_strike + mins_remaining
          target   = 1 if the round settles strictly above the round-open strike

        This is the exact question a Kalshi 15-minute contract asks, learned
        directly instead of approximated by a rolling-horizon regression.
        """
        feats = self._base_features(bars).copy()
        feats = self._join_extras(feats, bars)

        round_series = pd.Series(bars.index.floor('15min'), index=bars.index)
        grouped = bars.groupby(round_series)
        round_open = grouped['open'].transform('first')
        settle = grouped['close'].transform('last')
        feats['contract_target'] = (settle > round_open).astype(float)

        feature_cols = [c for c in feats.columns if c != 'contract_target']
        feats = feats.dropna(subset=feature_cols)
        if for_training:
            # Drop the first and last rounds in the window: the first may be
            # truncated (no true open) and the last has no settlement yet.
            rs = round_series.loc[feats.index]
            feats = feats[(rs > rs.iloc[0]) & (rs < rs.iloc[-1])]
        return feats, feature_cols

    def get_contract_training_data(self, lookback_hours: int = 1440,
                                   until: Optional[str] = None):
        """(X, y_contract, feature_names) for the strike-aware contract model."""
        end = datetime.fromisoformat(until) if until else datetime.now(timezone.utc)
        since = (end - timedelta(hours=lookback_hours)).isoformat()
        bars = self.build_bars(since=since, until=until)
        if bars is None or len(bars) < 120:
            return None, None, None
        feats, feature_cols = self.contract_frame_from_bars(bars, for_training=True)
        if len(feats) < 200:
            return None, None, None
        return feats[feature_cols].values, feats['contract_target'].values, feature_cols

    def get_current_contract_features(self, limit: int = 4000):
        """Feature vector for the live in-progress round (last bar)."""
        bars = self.build_bars(limit=limit)
        if bars is None or len(bars) < 60:
            return None, None
        self._live_bars_count = len(bars)
        feats, feature_cols = self.contract_frame_from_bars(bars, for_training=False)
        if len(feats) == 0:
            return None, None
        # At inference, sharpen the two highest-signal features with exact
        # wall-clock values: bar-level mins_remaining is quantized to whole
        # minutes (up to 60s stale at round edges).
        now = datetime.now(timezone.utc)
        secs_in_round = (now.minute % 15) * 60 + now.second
        if 'mins_remaining' in feats.columns:
            feats.iloc[-1, feats.columns.get_loc('mins_remaining')] = (900 - secs_in_round) / 60.0
        if 'mins_elapsed' in feats.columns:
            feats.iloc[-1, feats.columns.get_loc('mins_elapsed')] = secs_in_round / 60.0
        if 'round_15m_phase' in feats.columns:
            feats.iloc[-1, feats.columns.get_loc('round_15m_phase')] = secs_in_round / 900.0
        if 'round_15m_sin' in feats.columns:
            feats.iloc[-1, feats.columns.get_loc('round_15m_sin')] = np.sin(2 * np.pi * secs_in_round / 900.0)
        if 'round_15m_cos' in feats.columns:
            feats.iloc[-1, feats.columns.get_loc('round_15m_cos')] = np.cos(2 * np.pi * secs_in_round / 900.0)
        return feats.iloc[-1][feature_cols].values, feature_cols

    def get_training_data(self, horizon: int, lookback_hours: int = 168,
                          until: Optional[str] = None):
        end = datetime.fromisoformat(until) if until else datetime.now(timezone.utc)
        since = (end - timedelta(hours=lookback_hours)).isoformat()
        bars = self.build_bars(since=since, until=until)
        if bars is None or len(bars) < max(20, horizon) + 5:
            return None, None, None

        feats, feature_cols = self.features_from_bars(bars, horizon)
        train = feats.dropna(subset=['target'])
        if len(train) < 30:
            return None, None, None

        X = train[feature_cols].values
        y = train['target'].values
        return X, y, feature_cols

    def get_current_features(self, horizon: int, limit: int = 4000):
        bars = self.build_bars(limit=limit)
        if bars is None or len(bars) < max(20, horizon) + 5:
            return None, None

        feats, feature_cols = self.features_from_bars(bars, horizon)
        if len(feats) == 0:
            return None, None
        return feats.iloc[-1][feature_cols].values, feature_cols
