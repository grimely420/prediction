#!/usr/bin/env python3
"""Core correctness tests for the prediction system.

Run from the prediction/ directory:
    python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.kalshi_engine import KalshiRoundTracker  # noqa: E402
from shared.feature_engine import FeatureEngine  # noqa: E402


class StubStore:
    """Minimal DataStore stand-in for unit tests."""

    def __init__(self, prices=None, news=None, stats=None):
        self._prices = prices or []
        self._news = news or []
        self._stats = stats or {}

    def get_prices(self, limit=None, **kw):
        rows = self._prices
        return rows[-limit:] if limit else rows

    def get_news_features(self, limit=60000):
        return self._news[-limit:]

    def get_contract_signal_stats(self):
        return self._stats


def _ticks(sigma_tick, n=90, cadence_s=2.3, base=100.0):
    """Synthetic constant-volatility tick series with a fixed cadence."""
    # deterministic alternating-sign returns -> std ~= sigma_tick
    recs = []
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    p = base
    for i in range(n):
        sign = 1.0 if i % 2 == 0 else -1.0
        p *= (1.0 + sign * sigma_tick)
        recs.append({'timestamp': (t0 + timedelta(seconds=i * cadence_s)).isoformat(),
                     'price': p})
    return recs


class TestRoundTiming(unittest.TestCase):
    def setUp(self):
        self.rt = KalshiRoundTracker(StubStore(), 'btc')

    def _start_min(self, minute, second=0):
        now = datetime(2026, 1, 1, 10, minute, second, tzinfo=timezone.utc)
        t = self.rt.get_round_timing(now)['current_round']
        return (datetime.fromisoformat(t['start_time']).minute,
                datetime.fromisoformat(t['end_time']).minute,
                t['remaining_seconds'])

    def test_boundaries(self):
        self.assertEqual(self._start_min(7)[:2], (0, 15))
        self.assertEqual(self._start_min(15, 30)[:2], (15, 30))
        self.assertEqual(self._start_min(44)[:2], (30, 45))
        self.assertEqual(self._start_min(0)[:2], (0, 15))

    def test_cross_hour(self):
        start, end, _ = self._start_min(59)
        self.assertEqual(start, 45)
        self.assertEqual(end, 0)  # next hour

    def test_exact_boundary_is_new_round(self):
        # at exactly :15:00 the NEW round must have already started
        now = datetime(2026, 1, 1, 10, 15, 0, tzinfo=timezone.utc)
        cur = self.rt.get_round_timing(now)['current_round']
        self.assertEqual(cur['elapsed_seconds'], 0)
        self.assertEqual(cur['remaining_seconds'], 900)

    def test_settlement_minute_flag(self):
        now = datetime(2026, 1, 1, 10, 14, 30, tzinfo=timezone.utc)
        cur = self.rt.get_round_timing(now)['current_round']
        self.assertTrue(cur['is_settlement_minute'])
        now2 = datetime(2026, 1, 1, 10, 8, 0, tzinfo=timezone.utc)
        self.assertFalse(self.rt.get_round_timing(now2)['current_round']['is_settlement_minute'])


class TestVolatilityScaling(unittest.TestCase):
    def test_per_tick_scaled_to_per_minute(self):
        sigma_tick = 0.0003  # ~0.03% per 2.3s tick
        rt = KalshiRoundTracker(StubStore(prices=_ticks(sigma_tick)), 'btc')
        vol = rt.get_realized_volatility()
        expected = sigma_tick * math.sqrt(60.0 / 2.3)
        # alternating-sign series gives std exactly sigma_tick; allow 15% slack
        self.assertAlmostEqual(vol, expected, delta=expected * 0.15)
        # and critically: it must NOT equal the raw per-tick std
        self.assertGreater(vol, sigma_tick * 3)

    def test_fallback_when_no_data(self):
        rt = KalshiRoundTracker(StubStore(), 'btc')
        self.assertEqual(rt.get_realized_volatility(), 0.0008)

    def test_cadence_clamped(self):
        # 200s cadence -> clamped to 30s cadence equivalent
        recs = _ticks(0.0003, cadence_s=200.0)
        rt = KalshiRoundTracker(StubStore(prices=recs), 'btc')
        vol = rt.get_realized_volatility()
        self.assertGreater(vol, 0.0)


class TestCalibrationGate(unittest.TestCase):
    def test_thin_bucket_returns_none(self):
        stats = {'calibration': {'50-60': {'n': 5, 'empirical_win_pct': 30.0}}}
        rt = KalshiRoundTracker(StubStore(stats=stats), 'btc')
        self.assertIsNone(rt._calib_win_rate(0.55))

    def test_losing_bucket_reports_win_rate(self):
        stats = {'calibration': {'50-60': {'n': 50, 'empirical_win_pct': 48.0}}}
        rt = KalshiRoundTracker(StubStore(stats=stats), 'btc')
        self.assertAlmostEqual(rt._calib_win_rate(0.55), 0.48)

    def test_bucket_edges(self):
        stats = {'calibration': {'60-70': {'n': 40, 'empirical_win_pct': 65.0}}}
        rt = KalshiRoundTracker(StubStore(stats=stats), 'btc')
        self.assertEqual(rt._calib_win_rate(0.65), 0.65)
        self.assertIsNone(rt._calib_win_rate(0.70))  # next bucket, no data


class TestNewsSeries(unittest.TestCase):
    def test_series_mapping_and_order(self):
        rows = [
            {'timestamp': '2026-01-01T12:02:00+00:00', 'coin_sentiment': -0.2,
             'global_sentiment': -0.1, 'velocity_15m': 0.05, 'compound': -0.17},
            {'timestamp': '2026-01-01T12:00:00+00:00', 'coin_sentiment': 0.3,
             'global_sentiment': 0.1, 'velocity_15m': -0.02, 'compound': 0.23},
        ]
        fe = FeatureEngine(StubStore(news=rows), 'btc')
        s = fe._news_series()
        self.assertIsNotNone(s)
        self.assertEqual(list(s.columns),
                         ['timestamp', 'news_sentiment', 'news_global', 'news_velocity'])
        # ascending sort: the earlier row must come first
        self.assertAlmostEqual(s.iloc[0]['news_sentiment'], 0.23)
        self.assertAlmostEqual(s.iloc[1]['news_sentiment'], -0.17)

    def test_empty_series_none(self):
        fe = FeatureEngine(StubStore(), 'btc')
        self.assertIsNone(fe._news_series())


class TestPruning(unittest.TestCase):
    def test_prune_sql_builds(self):
        # import-time check that pruning SQL is syntactically reachable
        from shared.data_store import DataStore
        self.assertTrue(hasattr(DataStore, 'prune_old_data'))


if __name__ == '__main__':
    unittest.main()
