#!/usr/bin/env python3
"""
Generic SQLite data manager for price and prediction records.
"""

import sqlite3
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any

from .utils import setup_logging

logger = setup_logging("DataStore")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataStore:
    """Manages a single coin's SQLite database."""

    def __init__(self, db_path: str, coin_symbol: str):
        self.db_path = db_path
        self.symbol = coin_symbol
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices(timestamp)
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_prices_id ON prices(id)
        ''')

        # Backward-compatible migration for prices table
        prices_cols = [col[1] for col in c.execute("PRAGMA table_info(prices)").fetchall()]
        if 'created_at' not in prices_cols:
            try:
                # SQLite ALTER TABLE does not accept CURRENT_TIMESTAMP as a default,
                # so add the column without a default. save_price() populates it.
                c.execute("ALTER TABLE prices ADD COLUMN created_at TEXT")
            except Exception as e:
                logger.warning(f"Could not add created_at to prices: {e}")

        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                horizon_min INTEGER NOT NULL,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                error REAL,
                error_pct REAL,
                checked INTEGER DEFAULT 0,
                is_correct INTEGER,
                model_used TEXT,
                model_version TEXT,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Backward-compatible schema migrations (must run before indexes)
        existing = [col[1] for col in c.execute("PRAGMA table_info(predictions)").fetchall()]
        for col, dtype in [
            ('horizon_min', 'INTEGER NOT NULL DEFAULT 5'),
            ('current_price', 'REAL'),
            ('error_pct', 'REAL'),
            ('model_version', 'TEXT'),
            ('created_at', 'TEXT'),
        ]:
            if col not in existing:
                try:
                    c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {dtype}")
                except Exception as e:
                    logger.warning(f"Could not add column {col}: {e}")

        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time DESC)
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_checked ON predictions(checked)
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions(horizon_min)
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_perf ON predictions(checked, horizon_min, prediction_time DESC, id DESC)
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                qty REAL,
                amount_usd REAL,
                signal_source TEXT,
                prediction_id INTEGER,
                pnl REAL,
                pnl_pct REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC)
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action)
        ''')

        # Kalshi contract signals: one logged call per round per phase,
        # resolved later against the true settlement TWAP.
        c.execute('''
            CREATE TABLE IF NOT EXISTS contract_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_start TEXT NOT NULL,
                round_end TEXT NOT NULL,
                phase TEXT NOT NULL,
                call_time TEXT NOT NULL,
                mins_elapsed REAL,
                mins_remaining REAL,
                strike_price REAL NOT NULL,
                current_price REAL NOT NULL,
                prob_yes REAL,
                bias TEXT,
                source TEXT,
                model_version TEXT,
                settle_price REAL,
                outcome INTEGER,
                is_correct INTEGER,
                resolved INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(round_start, phase)
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_resolved
            ON contract_signals(resolved, round_end)
        ''')

        # Time series of news-engine features, one row per predictor cycle.
        # Lets training frames join sentiment by timestamp instead of seeing
        # the live value smeared across all history.
        c.execute('''
            CREATE TABLE IF NOT EXISTS news_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                coin_sentiment REAL,
                global_sentiment REAL,
                velocity_15m REAL,
                compound REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_news_ts ON news_features(timestamp)
        ''')

        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def save_price(self, price: float, source: str = '') -> bool:
        try:
            conn = self._conn()
            c = conn.cursor()
            try:
                c.execute(
                    "INSERT INTO prices (timestamp, price, source, created_at) VALUES (?, ?, ?, ?)",
                    (now_iso(), price, source, now_iso())
                )
            except sqlite3.OperationalError as e:
                if "no column named created_at" in str(e):
                    c.execute(
                        "INSERT INTO prices (timestamp, price, source) VALUES (?, ?, ?)",
                        (now_iso(), price, source)
                    )
                else:
                    raise
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] save_price error: {e}")
            return False

    def log_news_features(self, feats: Dict[str, float]) -> bool:
        """Append one news-feature sample (called per predictor cycle)."""
        try:
            conn = self._conn()
            conn.execute(
                "INSERT INTO news_features (timestamp, coin_sentiment, "
                "global_sentiment, velocity_15m, compound) VALUES (?, ?, ?, ?, ?)",
                (now_iso(),
                 float(feats.get('news_coin_sentiment', 0.0)),
                 float(feats.get('news_global_sentiment', 0.0)),
                 float(feats.get('news_15m_velocity', 0.0)),
                 float(feats.get('news_compound_signal', 0.0))))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] log_news_features error: {e}")
            return False

    def get_news_features(self, limit: int = 60000) -> List[Dict[str, Any]]:
        """Return news-feature rows ascending by time (newest `limit` rows)."""
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM (SELECT timestamp, coin_sentiment, "
                "global_sentiment, velocity_15m, compound FROM news_features "
                "ORDER BY timestamp DESC LIMIT ?) ORDER BY timestamp ASC",
                (int(limit),)).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[{self.symbol}] get_news_features error: {e}")
            return []

    def get_prices(self, limit: Optional[int] = None,
                   since: Optional[str] = None,
                   until: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return prices in ascending time order.

        When ``limit`` is given the *newest* ``limit`` rows within the window
        are returned (still ascending), so callers always see recent data.
        """
        try:
            conn = self._conn()
            c = conn.cursor()
            query = "SELECT id, timestamp, price, source FROM prices WHERE 1=1"
            params = []
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            if until:
                query += " AND timestamp <= ?"
                params.append(until)
            if limit:
                query = (f"SELECT * FROM ({query} ORDER BY timestamp DESC LIMIT {int(limit)}) "
                         "ORDER BY timestamp ASC")
            else:
                query += " ORDER BY timestamp ASC"
            c.execute(query, params)
            rows = c.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[{self.symbol}] get_prices error: {e}")
            return []

    def count_prices(self) -> int:
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM prices")
            n = c.fetchone()[0]
            conn.close()
            return n
        except Exception as e:
            logger.error(f"[{self.symbol}] count_prices error: {e}")
            return 0

    def last_price(self) -> Optional[Tuple[float, str]]:
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT price, timestamp FROM prices ORDER BY id DESC LIMIT 1")
            row = c.fetchone()
            conn.close()
            if row:
                return float(row['price']), row['timestamp']
            return None
        except Exception as e:
            logger.error(f"[{self.symbol}] last_price error: {e}")
            return None

    def price_at_or_after(self, target_iso: str, window_seconds: int = 60) -> Optional[Tuple[float, str]]:
        """Return the price nearest to target time (after target, within window)."""
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                "SELECT price, timestamp FROM prices WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                (target_iso,)
            )
            row = c.fetchone()
            conn.close()
            if row:
                ts = row['timestamp']
                if abs(self._dt_diff_seconds(ts, target_iso)) <= window_seconds:
                    return float(row['price']), ts
            return None
        except Exception as e:
            logger.error(f"[{self.symbol}] price_at_or_after error: {e}")
            return None

    def price_at_or_before(self, target_iso: str, window_seconds: int = 60) -> Optional[Tuple[float, str]]:
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                "SELECT price, timestamp FROM prices WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                (target_iso,)
            )
            row = c.fetchone()
            conn.close()
            if row:
                ts = row['timestamp']
                if abs(self._dt_diff_seconds(ts, target_iso)) <= window_seconds:
                    return float(row['price']), ts
            return None
        except Exception as e:
            logger.error(f"[{self.symbol}] price_at_or_before error: {e}")
            return None

    def _dt_diff_seconds(self, ts_a: str, ts_b: str) -> float:
        try:
            a = datetime.fromisoformat(ts_a.replace('Z', '+00:00'))
            b = datetime.fromisoformat(ts_b.replace('Z', '+00:00'))
            return abs((a - b).total_seconds())
        except Exception:
            return 1e9

    def log_prediction(self, horizon_min: int, current_price: float,
                       predicted_price: float, model_used: str,
                       model_version: str = '') -> int:
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """INSERT INTO predictions
                   (prediction_time, horizon_min, current_price, predicted_price,
                    checked, model_used, model_version, created_at)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?)""",
                (now_iso(), horizon_min, current_price, predicted_price,
                 model_used, model_version, now_iso())
            )
            pred_id = c.lastrowid
            conn.commit()
            conn.close()
            return pred_id
        except Exception as e:
            logger.error(f"[{self.symbol}] log_prediction error: {e}")
            return -1

    def get_unvalidated_predictions(self, max_age_minutes: int = 120,
                                    limit: int = 500) -> List[Dict[str, Any]]:
        """Get predictions that are ready to be validated (target time has passed).

        Newest first, so a backlog of unresolvable rows can never starve
        validation of fresh predictions.
        """
        try:
            conn = self._conn()
            c = conn.cursor()
            cutoff = datetime.now(timezone.utc).isoformat()
            c.execute(
                """SELECT id, prediction_time, horizon_min, predicted_price,
                          current_price, model_used, model_version
                   FROM predictions
                   WHERE checked = 0
                   AND datetime(prediction_time, '+'||horizon_min||' minutes') <= datetime(?)
                   ORDER BY id DESC
                   LIMIT ?""",
                (cutoff, limit)
            )
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"[{self.symbol}] get_unvalidated_predictions error: {e}")
            return []

    def mark_unresolvable(self, pred_ids: List[int]) -> int:
        """Flag predictions whose outcome can no longer be determined (checked = -1)."""
        if not pred_ids:
            return 0
        try:
            conn = self._conn()
            c = conn.cursor()
            c.executemany("UPDATE predictions SET checked = -1 WHERE id = ? AND checked = 0",
                          [(i,) for i in pred_ids])
            n = c.rowcount
            conn.commit()
            conn.close()
            return n
        except Exception as e:
            logger.error(f"[{self.symbol}] mark_unresolvable error: {e}")
            return 0

    def update_prediction(self, pred_id: int, actual_price: float,
                          error: float, error_pct: float,
                          is_correct: int) -> bool:
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """UPDATE predictions
                   SET actual_price = ?, error = ?, error_pct = ?,
                       checked = 1, is_correct = ?
                   WHERE id = ?""",
                (actual_price, error, error_pct, is_correct, pred_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] update_prediction error: {e}")
            return False

    def get_prediction_stats(self, window: int = 100,
                             horizon: Optional[int] = None,
                             since: Optional[str] = None) -> Dict[str, Any]:
        """Accuracy of the most recent ``window`` validated predictions.

        ``direction_accuracy_pct`` (did we call up/down correctly) and
        ``beat_naive_pct`` (were we closer than "price won't change") are the
        metrics that matter for trading; ``accuracy_pct_1pct`` is kept for
        backwards compatibility but is nearly always ~100% at short horizons.
        """
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """SELECT
                      COUNT(*),
                      AVG(ABS(error_pct)),
                      AVG(error_pct),
                      MAX(ABS(error_pct)),
                      SUM(CASE WHEN ABS(error_pct) < 1.0 THEN 1 ELSE 0 END),
                      SUM(CASE WHEN (predicted_price - current_price) * (actual_price - current_price) > 0
                               THEN 1 ELSE 0 END),
                      SUM(CASE WHEN ABS(actual_price - predicted_price) < ABS(actual_price - current_price)
                               THEN 1 ELSE 0 END),
                      AVG(ABS(actual_price - current_price) / current_price * 100)
                   FROM (
                       SELECT error_pct, predicted_price, current_price, actual_price
                       FROM predictions
                       WHERE checked = 1 AND actual_price IS NOT NULL AND current_price > 0
                       AND horizon_min = COALESCE(?, horizon_min)
                       AND prediction_time >= COALESCE(?, prediction_time)
                       ORDER BY id DESC LIMIT ?
                   ) tmp""",
                (horizon, since, window)
            )
            row = c.fetchone()
            conn.close()
            n = row[0] or 0
            pct = lambda x: round(100 * (x or 0) / n, 2) if n else 0
            return {
                'total': n,
                'avg_abs_error_pct': round(row[1] or 0, 4) if n else None,
                'avg_error_pct': round(row[2] or 0, 4) if n else None,
                'max_abs_error_pct': round(row[3] or 0, 4) if n else None,
                'within_threshold_1pct': row[4] or 0,
                'accuracy_pct_1pct': pct(row[4]),
                'direction_accuracy_pct': pct(row[5]),
                'beat_naive_pct': pct(row[6]),
                'naive_abs_error_pct': round(row[7] or 0, 4) if n else None,
            }
        except Exception as e:
            logger.error(f"[{self.symbol}] get_prediction_stats error: {e}")
            return {'total': 0}

    def get_recent_predictions(self, limit: int = 50,
                               horizon: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            conn = self._conn()
            c = conn.cursor()
            query = """SELECT id, prediction_time, horizon_min, current_price,
                              predicted_price, actual_price, error_pct,
                              checked, is_correct, model_used
                       FROM predictions WHERE 1=1"""
            params = []
            if horizon:
                query += " AND horizon_min = ?"
                params.append(horizon)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            c.execute(query, params)
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"[{self.symbol}] get_recent_predictions error: {e}")
            return []

    def log_trade(self, action: str, price: float, qty: Optional[float] = None,
                  amount_usd: Optional[float] = None, signal_source: str = '',
                  prediction_id: Optional[int] = None,
                  pnl: Optional[float] = None, pnl_pct: Optional[float] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store a trade executed by the connected trading bot."""
        try:
            conn = self._conn()
            c = conn.cursor()
            meta_json = json.dumps(metadata) if metadata else None
            c.execute(
                """INSERT INTO trades
                   (timestamp, action, price, qty, amount_usd, signal_source,
                    prediction_id, pnl, pnl_pct, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now_iso(), action, price, qty, amount_usd, signal_source,
                 prediction_id, pnl, pnl_pct, meta_json, now_iso())
            )
            trade_id = c.lastrowid
            conn.commit()
            conn.close()
            return trade_id
        except Exception as e:
            logger.error(f"[{self.symbol}] log_trade error: {e}")
            return -1

    def log_contract_signal(self, round_start: str, round_end: str, phase: str,
                            mins_elapsed: float, mins_remaining: float,
                            strike_price: float, current_price: float,
                            prob_yes: float, bias: str, source: str,
                            model_version: str = '') -> bool:
        """Log a contract call; UNIQUE(round_start, phase) dedupes writers."""
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """INSERT OR IGNORE INTO contract_signals
                   (round_start, round_end, phase, call_time, mins_elapsed,
                    mins_remaining, strike_price, current_price, prob_yes,
                    bias, source, model_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (round_start, round_end, phase, now_iso(), mins_elapsed,
                 mins_remaining, strike_price, current_price, prob_yes,
                 bias, source, model_version, now_iso())
            )
            n = c.rowcount
            conn.commit()
            conn.close()
            return n > 0
        except Exception as e:
            logger.error(f"[{self.symbol}] log_contract_signal error: {e}")
            return False

    def get_unresolved_signals(self, settled_before_iso: str,
                               limit: int = 200) -> List[Dict[str, Any]]:
        """Signals whose round ended before the cutoff (TWAP window elapsed)."""
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """SELECT id, round_start, round_end, phase, strike_price,
                          current_price, prob_yes, bias
                   FROM contract_signals
                   WHERE resolved = 0 AND round_end <= ?
                   ORDER BY round_end ASC LIMIT ?""",
                (settled_before_iso, limit)
            )
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"[{self.symbol}] get_unresolved_signals error: {e}")
            return []

    def resolve_contract_signal(self, sig_id: int, settle_price: float,
                                outcome: int, is_correct: Optional[int]) -> bool:
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """UPDATE contract_signals
                   SET settle_price = ?, outcome = ?, is_correct = ?, resolved = 1
                   WHERE id = ?""",
                (settle_price, outcome, is_correct, sig_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] resolve_contract_signal error: {e}")
            return False

    def get_contract_signal_stats(self, window: int = 500) -> Dict[str, Any]:
        """Win rate + probability calibration of resolved contract signals."""
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute(
                """SELECT phase, prob_yes, bias, outcome, is_correct,
                          strike_price, settle_price, round_start
                   FROM contract_signals WHERE resolved = 1
                   ORDER BY id DESC LIMIT ?""",
                (window,)
            )
            rows = [dict(r) for r in c.fetchall()]
            c.execute(
                """SELECT phase, prob_yes, bias, mins_elapsed, strike_price,
                          current_price, round_start, call_time
                   FROM contract_signals WHERE resolved = 0
                   ORDER BY id DESC LIMIT 30"""
            )
            pending = [dict(r) for r in c.fetchall()]
            conn.close()
        except Exception as e:
            logger.error(f"[{self.symbol}] get_contract_signal_stats error: {e}")
            return {'total': 0}

        directional = [r for r in rows if r['bias'] in ('UP', 'DOWN')
                       and r['is_correct'] is not None]
        by_phase: Dict[str, Any] = {}
        for r in directional:
            ph = by_phase.setdefault(r['phase'], {'n': 0, 'wins': 0, 'probs': []})
            ph['n'] += 1
            ph['wins'] += int(r['is_correct'])
            ph['probs'].append(float(r['prob_yes'] or 0.5))

        phase_out = {}
        for ph, v in by_phase.items():
            phase_out[ph] = {
                'n': v['n'],
                'win_rate_pct': round(100 * v['wins'] / v['n'], 1),
                'avg_prob_yes': round(sum(v['probs']) / v['n'], 3),
            }

        # Calibration: side probability bucket vs empirical win rate
        buckets: Dict[str, Any] = {}
        for r in directional:
            side_p = float(r['prob_yes'] or 0.5)
            if r['bias'] == 'DOWN':
                side_p = 1.0 - side_p
            if side_p < 0.55:
                b = '50-55'
            elif side_p < 0.65:
                b = '55-65'
            elif side_p < 0.75:
                b = '65-75'
            elif side_p < 0.85:
                b = '75-85'
            else:
                b = '85-99'
            v = buckets.setdefault(b, {'n': 0, 'wins': 0, 'probs': []})
            v['n'] += 1
            v['wins'] += int(r['is_correct'])
            v['probs'].append(side_p)

        calibration = {}
        for b, v in buckets.items():
            calibration[b] = {
                'n': v['n'],
                'avg_side_prob': round(sum(v['probs']) / v['n'], 3),
                'empirical_win_pct': round(100 * v['wins'] / v['n'], 1),
            }

        n_dir = len(directional)
        wins = sum(int(r['is_correct']) for r in directional)
        return {
            'total_resolved': len(rows),
            'directional_signals': n_dir,
            'win_rate_pct': round(100 * wins / n_dir, 1) if n_dir else None,
            'by_phase': phase_out,
            'calibration': calibration,
            'pending_signals': len(pending),
            'recent': [
                {
                    'round_start': r['round_start'],
                    'phase': r['phase'],
                    'bias': r['bias'],
                    'prob_yes': r['prob_yes'],
                    'strike': r['strike_price'],
                    'settle': r['settle_price'],
                    'outcome': 'UP' if r['outcome'] == 1 else 'DOWN',
                    'correct': r['is_correct'],
                }
                for r in rows[:20]
            ],
        }

    def get_recent_trades(self, limit: int = 100, action: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent trades recorded by the trading bot."""
        try:
            conn = self._conn()
            c = conn.cursor()
            query = """SELECT id, timestamp, action, price, qty, amount_usd,
                              signal_source, prediction_id, pnl, pnl_pct, metadata
                       FROM trades WHERE 1=1"""
            params = []
            if action:
                query += " AND action = ?"
                params.append(action)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            c.execute(query, params)
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            for row in rows:
                try:
                    row['metadata'] = json.loads(row['metadata']) if row['metadata'] else {}
                except Exception:
                    row['metadata'] = {}
            return rows
        except Exception as e:
            logger.error(f"[{self.symbol}] get_recent_trades error: {e}")
            return []

    def trade_markouts(self, limit: int = 2000,
                       horizons_min: List[int] = (5, 15, 60)) -> List[Dict[str, Any]]:
        """For each trade, the signed % price move N minutes later (positive = good for the side)."""
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute("""SELECT id, timestamp, action, price, qty, amount_usd, prediction_id
                         FROM trades ORDER BY id DESC LIMIT ?""", (limit,))
            trades = [dict(r) for r in c.fetchall()]
            for t in trades:
                t0 = datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))
                sign = 1.0 if t['action'].upper() == 'BUY' else -1.0
                for h in horizons_min:
                    target = (t0 + timedelta(minutes=h)).isoformat()
                    c.execute("SELECT price, timestamp FROM prices WHERE timestamp >= ? "
                              "ORDER BY timestamp ASC LIMIT 1", (target,))
                    row = c.fetchone()
                    key = f'markout_{h}m_pct'
                    if row and self._dt_diff_seconds(row['timestamp'], target) <= 120:
                        t[key] = round(sign * (row['price'] - t['price']) / t['price'] * 100, 4)
                    else:
                        t[key] = None
            conn.close()
            return trades
        except Exception as e:
            logger.error(f"[{self.symbol}] trade_markouts error: {e}")
            return []

    def bulk_validate(self, window_seconds: int = 90, give_up_after_seconds: int = 7200) -> Dict[str, int]:
        """Resolve every pending prediction in one pass (used for backfills).

        Loads all price timestamps into memory and uses binary search rather
        than one query per prediction.
        """
        import bisect
        try:
            conn = self._conn()
            c = conn.cursor()
            c.execute("SELECT timestamp, price FROM prices ORDER BY timestamp ASC")
            ts_list, px_list = [], []
            for ts, px in c.fetchall():
                ts_list.append(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())
                px_list.append(px)
            c.execute("""SELECT id, prediction_time, horizon_min, current_price, predicted_price
                         FROM predictions
                         WHERE checked = 0
                         AND datetime(prediction_time, '+'||horizon_min||' minutes') <= datetime('now')""")
            pending = c.fetchall()
            now = datetime.now(timezone.utc).timestamp()
            updates, retire = [], []
            for pid, pt, h, cur, predicted in pending:
                if not cur or cur <= 0:
                    retire.append((pid,))
                    continue
                target = datetime.fromisoformat(pt.replace('Z', '+00:00')).timestamp() + h * 60
                i = bisect.bisect_left(ts_list, target)
                if i < len(ts_list) and ts_list[i] - target <= window_seconds:
                    actual = px_list[i]
                    err = actual - predicted
                    err_pct = err / cur * 100
                    updates.append((actual, err, err_pct, 1 if abs(err_pct) <= 1.0 else 0, pid))
                elif now - target > give_up_after_seconds:
                    retire.append((pid,))
            c.executemany("""UPDATE predictions SET actual_price=?, error=?, error_pct=?,
                             checked=1, is_correct=? WHERE id=?""", updates)
            c.executemany("UPDATE predictions SET checked=-1 WHERE id=?", retire)
            conn.commit()
            conn.close()
            return {'pending': len(pending), 'validated': len(updates), 'retired': len(retire)}
        except Exception as e:
            logger.error(f"[{self.symbol}] bulk_validate error: {e}")
            return {'pending': 0, 'validated': 0, 'retired': 0, 'error': str(e)}
