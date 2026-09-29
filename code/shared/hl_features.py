#!/usr/bin/env python3
"""Merge Hyperliquid data onto 1-minute bars as extra model features.

Two tiers:
  - candle/funding features: available for the whole backfilled history
  - microstructure features (spread, imbalance, OI, trade-flow): only exist
    once the live collector has been running; enabled via ``micro=True``.

All features at bar index t use only data known by the end of minute t:
  - hl candles cover the same [t, t+1) window as the bar
  - funding is forward-filled from the last funding event <= t
  - snapshots/trades are last-observation/aggregated within minute t
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from .hyperliquid import HLStore, hl_db_path
from .utils import setup_logging

logger = setup_logging("HLFeatures")


def hl_extras(bars: pd.DataFrame, store: HLStore, micro: bool = False) -> Optional[pd.DataFrame]:
    """Return a DataFrame aligned to ``bars.index`` with hl_* columns, or None."""
    since = bars.index[0].isoformat()
    extra = pd.DataFrame(index=bars.index)

    candles = store.candles_frame(since=since)
    if not candles.empty:
        c = candles.reindex(bars.index)
        close = bars["close"]
        extra["hl_volume"] = c["v"]
        extra["hl_n_trades"] = c["n"]
        extra["hl_basis_bps"] = (c["c"] - close) / close * 1e4
        extra["hl_range"] = (c["h"] - c["l"]) / (c["c"] + 1e-10)
        extra["hl_body"] = (c["c"] - c["o"]) / (c["o"] + 1e-10)
        vol = c["v"].astype(float)
        extra["hl_vol_chg"] = vol.pct_change()
        extra["hl_vol_rel_20"] = vol / (vol.rolling(20, min_periods=5).mean() + 1e-10)
        extra["hl_vol_rel_240"] = vol / (vol.rolling(240, min_periods=60).mean() + 1e-10)
        extra["hl_nt_chg"] = c["n"].pct_change()
        r = c["c"].pct_change()
        extra["hl_volu_vol_20"] = (r * vol).rolling(20, min_periods=5).std()
        extra["hl_range_volume"] = extra["hl_range"] * vol
        extra["hl_candle_trend_5"] = (c["c"] - c["c"].rolling(5, min_periods=3).mean()) / (c["c"] + 1e-10)
        extra["hl_volume_z_60"] = (vol - vol.rolling(60, min_periods=20).mean()) / (vol.rolling(60, min_periods=20).std() + 1e-10)
        extra["hl_volume_z_240"] = (vol - vol.rolling(240, min_periods=60).mean()) / (vol.rolling(240, min_periods=60).std() + 1e-10)

    funding = store.funding_frame(since=since)
    if not funding.empty:
        f = funding.reindex(funding.index.union(bars.index)).ffill().reindex(bars.index)
        extra["hl_funding"] = f["funding_rate"]
        extra["hl_premium"] = f["premium"]
        fr = f["funding_rate"].astype(float)
        pm = f["premium"].astype(float)
        extra["hl_funding_premium_diff"] = fr - pm
        extra["hl_funding_ma_24h"] = fr.rolling(1440, min_periods=360).mean()
        extra["hl_funding_z_24h"] = (fr - fr.rolling(1440, min_periods=360).mean()) / (fr.rolling(1440, min_periods=360).std() + 1e-10)
        extra["hl_premium_z_24h"] = (pm - pm.rolling(1440, min_periods=360).mean()) / (pm.rolling(1440, min_periods=360).std() + 1e-10)

    if micro:
        snaps = store.snapshots_frame(since=since)
        if not snaps.empty:
            s = snaps.resample("1min").last()
            s = s.reindex(s.index.union(bars.index)).ffill().reindex(bars.index)
            extra["hl_spread_bps"] = s["spread_bps"]
            extra["hl_imb_1"] = s["imbalance_1"]
            extra["hl_imb_5"] = s["imbalance_5"]
            extra["hl_imb_20"] = s["imbalance_20"]
            extra["hl_oi"] = s["open_interest"]
            extra["hl_oi_chg"] = s["open_interest"].pct_change(5)
            extra["hl_mark_basis_bps"] = (s["mark_px"] - bars["close"]) / bars["close"] * 1e4

        trades = store.trades_frame(since=since)
        if not trades.empty:
            trades = trades.copy()
            trades["signed_sz"] = np.where(trades["side"] == "B", trades["sz"], -trades["sz"])
            g = trades.resample("1min").agg(
                signed=("signed_sz", "sum"), total=("sz", "sum"))
            g = g.reindex(bars.index)
            extra["hl_trade_imb"] = g["signed"] / (g["total"] + 1e-10)

    # Drop inf / pathological numerics that break XGBoost training.
    extra = extra.replace([np.inf, -np.inf], np.nan)
    extra = extra.clip(lower=-1e6, upper=1e6)
    if extra.isna().all(axis=None) or extra.empty:
        return None
    return extra


def attach_extras(fe, coin_dir: str, micro: bool = False) -> bool:
    """Attach a Hyperliquid extras loader to a FeatureEngine if data exists."""
    path = hl_db_path(coin_dir)
    if not os.path.exists(path):
        return False
    store = HLStore(path)
    if (store.status()["hl_candles_1m"]["count"] or 0) == 0:
        return False
    fe.extras_fn = lambda bars: hl_extras(bars, store, micro=micro)
    return True
