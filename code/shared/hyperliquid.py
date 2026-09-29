#!/usr/bin/env python3
"""Hyperliquid market-data client, storage, and derived features.

Data collected (per coin, default HYPE):
  hl_snapshots   every few seconds: mark/oracle/mid, funding, open interest,
                 premium, top-of-book and 5/20-level depth + imbalance
  hl_trades      the public trade tape (deduplicated by tid)
  hl_candles_1m  1-minute OHLCV with trade count (backfillable ~months)
  hl_funding     hourly funding rate history (backfillable)

CLI:
  python -m shared.hyperliquid backfill [--coin HYPE] [--days 90]
  python -m shared.hyperliquid status  [--coin HYPE]
"""

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from .coin_config import BASE_DIR
from .utils import setup_logging

logger = setup_logging("Hyperliquid")

INFO_URL = "https://api.hyperliquid.xyz/info"


def hl_db_path(coin_dir: str = "hype") -> str:
    return os.path.join(BASE_DIR, coin_dir, f"{coin_dir}_hl.db")


class HLClient:
    """Thin wrapper over the /info endpoint."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def _post(self, payload: Dict[str, Any]) -> Any:
        r = self.session.post(INFO_URL, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def asset_ctx(self, coin: str) -> Dict[str, Any]:
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        for u, ctx in zip(meta["universe"], ctxs):
            if u["name"] == coin:
                return ctx
        raise ValueError(f"{coin} not in Hyperliquid universe")

    def l2_book(self, coin: str, n_sig_figs: Optional[int] = None) -> Dict[str, Any]:
        payload = {"type": "l2Book", "coin": coin}
        if n_sig_figs:
            payload["nSigFigs"] = n_sig_figs
        return self._post(payload)

    def recent_trades(self, coin: str) -> List[Dict[str, Any]]:
        return self._post({"type": "recentTrades", "coin": coin})

    def candles(self, coin: str, start_ms: int, end_ms: int, interval: str = "1m") -> List[Dict[str, Any]]:
        return self._post({"type": "candleSnapshot",
                           "req": {"coin": coin, "interval": interval,
                                   "startTime": start_ms, "endTime": end_ms}})

    def funding_history(self, coin: str, start_ms: int, end_ms: Optional[int] = None) -> List[Dict[str, Any]]:
        payload = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
        if end_ms:
            payload["endTime"] = end_ms
        return self._post(payload)


class HLStore:
    """SQLite storage for Hyperliquid data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS hl_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mark_px REAL, oracle_px REAL, mid_px REAL,
                funding REAL, open_interest REAL, premium REAL,
                day_ntl_vlm REAL, day_base_vlm REAL,
                bid_px REAL, ask_px REAL, spread_bps REAL,
                bid_sz_1 REAL, ask_sz_1 REAL,
                bid_depth_5 REAL, ask_depth_5 REAL,
                bid_depth_20 REAL, ask_depth_20 REAL,
                imbalance_1 REAL, imbalance_5 REAL, imbalance_20 REAL
            );
            CREATE INDEX IF NOT EXISTS idx_hl_snap_ts ON hl_snapshots(timestamp);

            CREATE TABLE IF NOT EXISTS hl_trades (
                tid INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                time_ms INTEGER NOT NULL,
                side TEXT, px REAL, sz REAL
            );
            CREATE INDEX IF NOT EXISTS idx_hl_trades_ms ON hl_trades(time_ms);

            CREATE TABLE IF NOT EXISTS hl_candles_1m (
                t INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                o REAL, h REAL, l REAL, c REAL, v REAL, n INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_hl_candles_ts ON hl_candles_1m(timestamp);

            CREATE TABLE IF NOT EXISTS hl_funding (
                time_ms INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                funding_rate REAL, premium REAL
            );
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def _iso(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

    def save_snapshot(self, row: Dict[str, Any]) -> None:
        cols = list(row.keys())
        conn = self._conn()
        conn.execute(f"INSERT INTO hl_snapshots ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [row[k] for k in cols])
        conn.commit()
        conn.close()

    def save_trades(self, trades: List[Dict[str, Any]]) -> int:
        if not trades:
            return 0
        conn = self._conn()
        c = conn.cursor()
        c.executemany(
            "INSERT OR IGNORE INTO hl_trades (tid, timestamp, time_ms, side, px, sz) VALUES (?,?,?,?,?,?)",
            [(t["tid"], self._iso(t["time"]), t["time"], t["side"], float(t["px"]), float(t["sz"]))
             for t in trades])
        n = c.rowcount
        conn.commit()
        conn.close()
        return n

    def save_candles(self, candles: List[Dict[str, Any]]) -> int:
        if not candles:
            return 0
        conn = self._conn()
        c = conn.cursor()
        c.executemany(
            "INSERT OR REPLACE INTO hl_candles_1m (t, timestamp, o, h, l, c, v, n) VALUES (?,?,?,?,?,?,?,?)",
            [(k["t"], self._iso(k["t"]), float(k["o"]), float(k["h"]), float(k["l"]),
              float(k["c"]), float(k["v"]), int(k["n"])) for k in candles])
        n = c.rowcount
        conn.commit()
        conn.close()
        return n

    def save_funding(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        conn = self._conn()
        c = conn.cursor()
        c.executemany(
            "INSERT OR REPLACE INTO hl_funding (time_ms, timestamp, funding_rate, premium) VALUES (?,?,?,?)",
            [(r["time"], self._iso(r["time"]), float(r["fundingRate"]), float(r["premium"])) for r in rows])
        n = c.rowcount
        conn.commit()
        conn.close()
        return n

    def first_candle_ms(self) -> Optional[int]:
        conn = self._conn()
        r = conn.execute("SELECT MIN(t) FROM hl_candles_1m").fetchone()[0]
        conn.close()
        return r

    def last_candle_ms(self) -> Optional[int]:
        conn = self._conn()
        r = conn.execute("SELECT MAX(t) FROM hl_candles_1m").fetchone()[0]
        conn.close()
        return r

    def last_funding_ms(self) -> Optional[int]:
        conn = self._conn()
        r = conn.execute("SELECT MAX(time_ms) FROM hl_funding").fetchone()[0]
        conn.close()
        return r

    def status(self) -> Dict[str, Any]:
        conn = self._conn()
        out = {}
        for t, ts in (("hl_snapshots", "timestamp"), ("hl_trades", "timestamp"),
                      ("hl_candles_1m", "timestamp"), ("hl_funding", "timestamp")):
            r = conn.execute(f"SELECT COUNT(*), MIN({ts}), MAX({ts}) FROM {t}").fetchone()
            out[t] = {"count": r[0], "first": r[1], "last": r[2]}
        conn.close()
        return out

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        r = conn.execute("SELECT * FROM hl_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return dict(r) if r else None

    # ----- frames for feature building -----
    def candles_frame(self, since: Optional[str] = None) -> pd.DataFrame:
        conn = self._conn()
        q = "SELECT timestamp, o, h, l, c, v, n FROM hl_candles_1m"
        params = []
        if since:
            q += " WHERE timestamp >= ?"
            params.append(since)
        df = pd.read_sql(q + " ORDER BY t", conn, params=params)
        conn.close()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        return df.set_index("timestamp")

    def funding_frame(self, since: Optional[str] = None) -> pd.DataFrame:
        conn = self._conn()
        q = "SELECT timestamp, funding_rate, premium FROM hl_funding"
        params = []
        if since:
            q += " WHERE timestamp >= ?"
            params.append(since)
        df = pd.read_sql(q + " ORDER BY time_ms", conn, params=params)
        conn.close()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        return df.set_index("timestamp")

    def snapshots_frame(self, since: Optional[str] = None) -> pd.DataFrame:
        conn = self._conn()
        q = ("SELECT timestamp, mark_px, oracle_px, funding, open_interest, premium, spread_bps, "
             "imbalance_1, imbalance_5, imbalance_20 FROM hl_snapshots")
        params = []
        if since:
            q += " WHERE timestamp >= ?"
            params.append(since)
        df = pd.read_sql(q + " ORDER BY id", conn, params=params)
        conn.close()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        return df.set_index("timestamp")

    def trades_frame(self, since: Optional[str] = None) -> pd.DataFrame:
        conn = self._conn()
        q = "SELECT timestamp, side, px, sz FROM hl_trades"
        params = []
        if since:
            q += " WHERE timestamp >= ?"
            params.append(since)
        df = pd.read_sql(q + " ORDER BY time_ms", conn, params=params)
        conn.close()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        return df.set_index("timestamp")


def book_metrics(book: Dict[str, Any]) -> Dict[str, float]:
    bids, asks = book["levels"][0], book["levels"][1]
    if not bids or not asks:
        return {}
    b1, a1 = float(bids[0]["px"]), float(asks[0]["px"])
    mid = (b1 + a1) / 2
    bsz = np.array([float(x["sz"]) for x in bids])
    asz = np.array([float(x["sz"]) for x in asks])

    def imb(n):
        b, a = bsz[:n].sum(), asz[:n].sum()
        return (b - a) / (b + a) if (b + a) > 0 else 0.0

    return {
        "bid_px": b1, "ask_px": a1,
        "spread_bps": (a1 - b1) / mid * 1e4,
        "bid_sz_1": float(bsz[0]), "ask_sz_1": float(asz[0]),
        "bid_depth_5": float(bsz[:5].sum()), "ask_depth_5": float(asz[:5].sum()),
        "bid_depth_20": float(bsz.sum()), "ask_depth_20": float(asz.sum()),
        "imbalance_1": imb(1), "imbalance_5": imb(5), "imbalance_20": imb(20),
    }


def snapshot_row(ctx: Dict[str, Any], book: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mark_px": float(ctx["markPx"]), "oracle_px": float(ctx["oraclePx"]),
        "mid_px": float(ctx["midPx"]) if ctx.get("midPx") else None,
        "funding": float(ctx["funding"]), "open_interest": float(ctx["openInterest"]),
        "premium": float(ctx["premium"]) if ctx.get("premium") is not None else None,
        "day_ntl_vlm": float(ctx["dayNtlVlm"]), "day_base_vlm": float(ctx.get("dayBaseVlm", 0) or 0),
    }
    row.update(book_metrics(book))
    return row


# ----------------------------------------------------------------------------
# Backfill
# ----------------------------------------------------------------------------
def backfill(coin: str, store: HLStore, days: int, client: Optional[HLClient] = None) -> None:
    client = client or HLClient(timeout=30)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000
    step = 4_900 * 60_000  # ~5000 candles per request
    total = 0
    t = start_ms
    while t < now_ms:
        end = min(t + step, now_ms)
        try:
            rows = client.candles(coin, t, end)
        except Exception as e:
            logger.warning(f"candle backfill {coin} {t}: {e}; retrying")
            time.sleep(2)
            continue
        n = store.save_candles(rows)
        total += n
        logger.info(f"[{coin}] candles {HLStore._iso(t)[:16]} -> {HLStore._iso(end)[:16]}: {len(rows)} rows")
        t = end + 1
        time.sleep(0.3)
    logger.info(f"[{coin}] candle backfill done, {total} rows written")

    total = 0
    t = start_ms
    while t < now_ms:
        try:
            rows = client.funding_history(coin, t, now_ms)
        except Exception as e:
            logger.warning(f"funding backfill {coin} {t}: {e}; retrying")
            time.sleep(2)
            continue
        if not rows:
            break
        total += store.save_funding(rows)
        last = max(r["time"] for r in rows)
        if last <= t:
            break
        t = last + 1
        time.sleep(0.3)
    logger.info(f"[{coin}] funding backfill done, {total} rows written")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["backfill", "status"])
    p.add_argument("--coin", default="HYPE")
    p.add_argument("--coin-dir", default="hype")
    p.add_argument("--days", type=int, default=90)
    a = p.parse_args()
    store = HLStore(hl_db_path(a.coin_dir))
    if a.cmd == "backfill":
        backfill(a.coin, store, a.days)
    for k, v in store.status().items():
        print(f"{k:15s} {v['count']:>9,}  {str(v['first'])[:19]} -> {str(v['last'])[:19]}")


if __name__ == "__main__":
    main()
