#!/usr/bin/env python3
"""Walk-forward backtest of the XGBoost return model against naive baselines.

Usage:
    python -m shared.backtest btc [--days 30] [--train-hours 336] [--step-hours 24]

For every step the model is trained only on bars strictly before the test
window (same code path as production: ModelManager.fit_model), then scored on
the next ``step_hours`` of bars. Reports MAE, direction accuracy, how often the
model beats "no change", and a simple threshold-trading simulation.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .coin_config import get_coin_config
from .data_store import DataStore
from .feature_engine import FeatureEngine
from .model_manager import ModelManager


def _score(y_true: np.ndarray, y_pred: np.ndarray, fee_pct: float, thresholds):
    out = {
        'n': int(len(y_true)),
        'mae_pct': float(np.mean(np.abs(y_true - y_pred)) * 100),
        'naive_mae_pct': float(np.mean(np.abs(y_true)) * 100),
        'dir_acc_pct': float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100),
        'beat_naive_pct': float(np.mean(np.abs(y_true - y_pred) < np.abs(y_true)) * 100),
        'corr': float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else float('nan'),
    }
    # Trade only when |predicted move| exceeds threshold; earn realised move minus fees.
    for th in thresholds:
        mask = np.abs(y_pred) > th / 100
        if mask.sum() == 0:
            out[f'trade_{th}'] = {'trades': 0}
            continue
        pnl = np.sign(y_pred[mask]) * y_true[mask] * 100 - fee_pct
        out[f'trade_{th}'] = {
            'trades': int(mask.sum()),
            'hit_rate_pct': float(np.mean(pnl > 0) * 100),
            'avg_pnl_pct': float(np.mean(pnl)),
            'total_pnl_pct': float(np.sum(pnl)),
        }
    return out


def run(coin_id: str, days: int, train_hours: int, step_hours: int, horizons,
        fee_pct: float, thresholds, hl: bool = False, micro: bool = False):
    cfg = get_coin_config(coin_id)
    ds = DataStore(cfg.db_path, cfg.symbol)
    fe = FeatureEngine(ds, cfg.symbol)
    if hl:
        from .hl_features import attach_extras
        if attach_extras(fe, cfg.db_name, micro=micro):
            print(f"[{cfg.symbol}] Hyperliquid features enabled (micro={micro})", flush=True)
        else:
            print(f"[{cfg.symbol}] WARNING: no Hyperliquid data found, running price-only", flush=True)

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    since = (start - timedelta(hours=train_hours)).isoformat()
    print(f"[{cfg.symbol}] loading ticks since {since[:19]} ...", flush=True)
    bars = fe.build_bars(since=since)
    if bars is None:
        print("no data"); return
    print(f"[{cfg.symbol}] {len(bars):,} one-minute bars, "
          f"{bars.index[0]:%Y-%m-%d %H:%M} -> {bars.index[-1]:%Y-%m-%d %H:%M}\n", flush=True)

    results = {}
    for h in horizons:
        feats, cols = fe.features_from_bars(bars, h)
        feats = feats.dropna(subset=['target'])
        y_true_all, y_pred_all, folds = [], [], 0
        t = start
        while t < end:
            t_next = min(t + timedelta(hours=step_hours), end)
            train = feats[(feats.index >= t - timedelta(hours=train_hours)) &
                          (feats.index < t - timedelta(minutes=h))]  # no target leakage
            test = feats[(feats.index >= t) & (feats.index < t_next)]
            if len(train) >= 200 and len(test) > 0:
                model, _ = ModelManager.fit_model(train[cols].values, train['target'].values)
                y_pred_all.append(model.predict(test[cols].values))
                y_true_all.append(test['target'].values)
                folds += 1
            t = t_next
        if not folds:
            print(f"  {h}m: insufficient data"); continue
        y_true = np.concatenate(y_true_all)
        y_pred = np.clip(np.concatenate(y_pred_all), -0.05, 0.05)
        res = _score(y_true, y_pred, fee_pct, thresholds)
        res['folds'] = folds
        results[h] = res

        print(f"=== {cfg.symbol} {h}m horizon  ({folds} folds, {res['n']:,} out-of-sample bars) ===")
        print(f"  MAE            model {res['mae_pct']:.4f}%   naive(no-change) {res['naive_mae_pct']:.4f}%")
        print(f"  direction acc  {res['dir_acc_pct']:.2f}%   (50% = coin flip)")
        print(f"  beats naive    {res['beat_naive_pct']:.2f}%   corr(pred,actual) {res['corr']:+.4f}")
        for th in thresholds:
            tr = res[f'trade_{th}']
            if tr['trades']:
                print(f"  trade if |pred|>{th}%: {tr['trades']:>5} trades  hit {tr['hit_rate_pct']:.1f}%  "
                      f"avg {tr['avg_pnl_pct']:+.4f}%  total {tr['total_pnl_pct']:+.2f}%  (fee {fee_pct}%/trade)")
            else:
                print(f"  trade if |pred|>{th}%:     0 trades")
        print(flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('coin', nargs='?', default='btc')
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--train-hours', type=int, default=24 * 14)
    p.add_argument('--step-hours', type=int, default=24)
    p.add_argument('--horizons', default='5,10,15')
    p.add_argument('--fee-pct', type=float, default=0.05, help='round-trip cost per trade, in percent')
    p.add_argument('--thresholds', default='0.05,0.1,0.2')
    p.add_argument('--hl', action='store_true', help='add Hyperliquid volume/funding features')
    p.add_argument('--micro', action='store_true', help='also add book/trade-flow features (needs live collector data)')
    a = p.parse_args()
    run(a.coin, a.days, a.train_hours, a.step_hours,
        [int(x) for x in a.horizons.split(',')], a.fee_pct,
        [float(x) for x in a.thresholds.split(',')], hl=a.hl, micro=a.micro)


if __name__ == '__main__':
    main()
