#!/usr/bin/env python3
"""Research experiments on the collected tick database.

    python -m shared.experiments [--days 28]

1. Longer horizons (60m / 240m), price-only features.
2. Cross-asset features: does BTC lead BNB / HYPE (and vice-versa)?
3. Volatility forecasting: predict realised vol over the next hour and compare
   to the "vol persists" baseline.

Everything is walk-forward (train strictly before each test window).
"""

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .coin_config import get_coin_config
from .data_store import DataStore
from .feature_engine import FeatureEngine
from .model_manager import ModelManager

COINS = ('btc', 'bnb', 'hype')


def load_bars(days: int, train_hours: int):
    since = (datetime.now(timezone.utc) - timedelta(days=days, hours=train_hours)).isoformat()
    bars = {}
    for c in COINS:
        cfg = get_coin_config(c)
        fe = FeatureEngine(DataStore(cfg.db_path, cfg.symbol), cfg.symbol)
        b = fe.build_bars(since=since)
        bars[c] = b
        print(f"  {c}: {len(b):,} bars {b.index[0]:%m-%d %H:%M} -> {b.index[-1]:%m-%d %H:%M}")
    idx = bars['btc'].index
    for c in COINS[1:]:
        idx = idx.intersection(bars[c].index)
    return {c: b.loc[idx] for c, b in bars.items()}, fe


def walk_forward(feats: pd.DataFrame, cols, target: str, days: int,
                 train_hours: int, step_hours: int, horizon_min: int):
    end = feats.index[-1]
    start = end - timedelta(days=days)
    yt, yp, base = [], [], []
    t = start
    while t < end:
        t_next = min(t + timedelta(hours=step_hours), end)
        train = feats[(feats.index >= t - timedelta(hours=train_hours)) &
                      (feats.index < t - timedelta(minutes=horizon_min))]
        test = feats[(feats.index >= t) & (feats.index < t_next)]
        if len(train) >= 300 and len(test) > 0:
            m, _ = ModelManager.fit_model(train[cols].values, train[target].values)
            yp.append(m.predict(test[cols].values))
            yt.append(test[target].values)
            if 'baseline' in test.columns:
                base.append(test['baseline'].values)
        t = t_next
    if not yt:
        return None
    out = np.concatenate(yt), np.concatenate(yp)
    return out + ((np.concatenate(base),) if base else ())


def report_direction(name, y, p, fee=0.05, thresholds=(0.1, 0.25, 0.5)):
    dir_acc = np.mean(np.sign(p) == np.sign(y)) * 100
    corr = np.corrcoef(y, p)[0, 1]
    line = f"  {name:34s} n={len(y):>6,} dir={dir_acc:5.1f}%  corr={corr:+.3f}"
    for th in thresholds:
        m = np.abs(p) > th / 100
        if m.sum() >= 10:
            pnl = np.sign(p[m]) * y[m] * 100 - fee
            line += f" | >{th}%: {m.sum():>4} tr, hit {np.mean(pnl > 0) * 100:4.1f}%, avg {pnl.mean():+.3f}%"
    print(line)


def exp_long_horizons(bars, fe, days, train_hours, step_hours):
    print("\n=== 1. Longer horizons, price-only features ===")
    for h in (60, 240):
        for c in COINS:
            feats, cols = fe.features_from_bars(bars[c], h)
            feats = feats.dropna(subset=['target'])
            r = walk_forward(feats, cols, 'target', days, train_hours, step_hours, h)
            if r:
                report_direction(f"{c.upper()} {h}m", r[0], r[1])


def cross_feats(bars, src: str, prefix: str) -> pd.DataFrame:
    close = bars[src]['close']
    ret = close.pct_change()
    out = pd.DataFrame(index=close.index)
    for w in (1, 5, 15, 60):
        out[f'{prefix}_ret_{w}'] = close.pct_change(w)
    for w in (15, 60):
        out[f'{prefix}_vol_{w}'] = ret.rolling(w, min_periods=5).std()
    return out


def exp_cross_asset(bars, fe, days, train_hours, step_hours):
    print("\n=== 2. Cross-asset features (walk-forward, own-features vs own+other coins) ===")
    for h in (15, 60):
        for tgt in COINS:
            own, cols = fe.features_from_bars(bars[tgt], h)
            others = [c for c in COINS if c != tgt]
            extra = pd.concat([cross_feats(bars, o, o) for o in others], axis=1)
            # relative-strength features: target's move minus the others'
            for o in others:
                extra[f'rel_ret_5_{o}'] = bars[tgt]['close'].pct_change(5) - bars[o]['close'].pct_change(5)
                extra[f'rel_ret_15_{o}'] = bars[tgt]['close'].pct_change(15) - bars[o]['close'].pct_change(15)
            full = own.join(extra, how='inner').dropna()
            all_cols = cols + list(extra.columns)
            r_own = walk_forward(full, cols, 'target', days, train_hours, step_hours, h)
            r_all = walk_forward(full, all_cols, 'target', days, train_hours, step_hours, h)
            if r_own and r_all:
                report_direction(f"{tgt.upper()} {h}m  own only", r_own[0], r_own[1])
                report_direction(f"{tgt.upper()} {h}m  + {'/'.join(others)}", r_all[0], r_all[1])


def exp_volatility(bars, fe, days, train_hours, step_hours):
    print("\n=== 3. Volatility forecasting: realised vol of next 60 min (std of 1-min returns, %) ===")
    print("    baseline = trailing 60-min vol (persistence)")
    h = 60
    for c in COINS:
        feats, cols = fe.features_from_bars(bars[c], 5)
        ret = bars[c]['close'].pct_change()
        fwd_vol = ret[::-1].rolling(h, min_periods=h).std()[::-1].shift(-1) * 100
        feats = feats.drop(columns=['target']).copy()
        feats['target'] = fwd_vol.reindex(feats.index)
        feats['baseline'] = (ret.rolling(h, min_periods=30).std() * 100).reindex(feats.index)
        feats = feats.dropna(subset=['target', 'baseline'])
        r = walk_forward(feats, cols, 'target', days, train_hours, step_hours, h)
        if not r:
            continue
        y, p, b = r
        mae_m, mae_b = np.mean(np.abs(y - p)), np.mean(np.abs(y - b))
        corr_m, corr_b = np.corrcoef(y, p)[0, 1], np.corrcoef(y, b)[0, 1]
        # Ranking skill: does the top-quintile forecast actually pick the high-vol hours?
        q = np.quantile(p, [0.2, 0.8])
        lo, hi = y[p <= q[0]].mean(), y[p >= q[1]].mean()
        qb = np.quantile(b, [0.2, 0.8])
        lo_b, hi_b = y[b <= qb[0]].mean(), y[b >= qb[1]].mean()
        print(f"  {c.upper():5s} n={len(y):,}  MAE model {mae_m:.4f} vs baseline {mae_b:.4f} "
              f"({(1 - mae_m / mae_b) * 100:+.1f}%)  corr model {corr_m:.3f} vs baseline {corr_b:.3f}")
        print(f"        realised vol when forecast is bottom-20% / top-20%:  model {lo:.4f} / {hi:.4f}  "
              f"(x{hi / lo:.2f})   baseline {lo_b:.4f} / {hi_b:.4f} (x{hi_b / lo_b:.2f})")


def exp_streak_conditioning(bars, days):
    print("\n=== 4. The bot's streak strategy, conditioned on volatility regime ===")
    print("    After N consecutive down 1-min closes: avg next-15m return (BUY-the-dip payoff), by trailing-60m vol tercile")
    for c in COINS:
        close = bars[c]['close']
        cutoff = close.index[-1] - timedelta(days=days)
        ret = close.pct_change()
        down = (ret < 0).astype(int)
        streak = down.groupby((down != down.shift()).cumsum()).cumsum() * down
        fwd15 = (close.shift(-15) / close - 1) * 100
        vol60 = ret.rolling(60, min_periods=30).std()
        df = pd.DataFrame({'streak': streak, 'fwd15': fwd15, 'vol': vol60}).loc[cutoff:].dropna()
        df['vol_tercile'] = pd.qcut(df['vol'], 3, labels=['low', 'mid', 'high'])
        for n in (3, 5):
            sub = df[df.streak >= n]
            g = sub.groupby('vol_tercile', observed=True)['fwd15'].agg(['mean', 'count'])
            parts = "  ".join(f"{k}: {v['mean']:+.3f}% (n={int(v['count'])})" for k, v in g.iterrows())
            print(f"  {c.upper():5s} streak>={n}: all {sub.fwd15.mean():+.3f}% (n={len(sub)}) | {parts}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=28)
    p.add_argument('--train-hours', type=int, default=24 * 14)
    p.add_argument('--step-hours', type=int, default=72)
    p.add_argument('--only', default='1,2,3,4')
    a = p.parse_args()
    print("loading bars ...")
    bars, fe = load_bars(a.days, a.train_hours)
    only = a.only.split(',')
    if '1' in only: exp_long_horizons(bars, fe, a.days, a.train_hours, a.step_hours)
    if '2' in only: exp_cross_asset(bars, fe, a.days, a.train_hours, a.step_hours)
    if '3' in only: exp_volatility(bars, fe, a.days, a.train_hours, a.step_hours)
    if '4' in only: exp_streak_conditioning(bars, a.days)


if __name__ == '__main__':
    main()
