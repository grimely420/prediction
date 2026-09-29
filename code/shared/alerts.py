#!/usr/bin/env python3
"""Read-only Telegram alerting for the prediction system.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment.
Sends alerts when services, prices, or predictions look stale/unhealthy.
"""

import os
import time
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import requests

from .utils import setup_logging

logger = setup_logging("Alerts")

BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

def _service_list() -> list:
    """All prediction services, derived from the configured coin set."""
    from .coin_config import list_coins
    svcs = ["prediction-api"]
    for c in list_coins():
        svcs.append(f"prediction-collector@{c}")
        svcs.append(f"prediction-predictor@{c}")
    svcs.append("prediction-hl-collector@hype")
    return svcs


SERVICES = _service_list()


def _load_env():
    """Load the project .env file into os.environ if present."""
    # alerts.py lives at project/prediction/shared/alerts.py; .env is at project/.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k not in os.environ:
                os.environ[k] = v


_load_env()


def get_telegram_credentials() -> Optional[tuple]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id
    return None


def send_telegram(message: str, disable_notification: bool = False) -> bool:
    creds = get_telegram_credentials()
    if not creds:
        logger.warning("Telegram credentials not set; cannot send alert")
        return False
    token, chat_id = creds
    try:
        r = requests.post(
            BASE_URL.format(token=token),
            data={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_notification": disable_notification,
            },
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


class SystemAlerter:
    """Poll the system and send state-change alerts to Telegram (read-only)."""

    def __init__(self, api_base: str = "http://127.0.0.1:5000", heartbeat_min: int = 240):
        self.api_base = os.environ.get("API_BASE", api_base)
        self.heartbeat_min = heartbeat_min
        self.last_state: Dict[str, Any] = {}
        self.last_heartbeat = 0
        self.last_service_check = 0

    def _api(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.api_base}{path}", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.debug(f"API {path} failed: {e}")
            return None

    @staticmethod
    def _services_status() -> Dict[str, bool]:
        status = {}
        for svc in SERVICES:
            try:
                out = subprocess.run(
                    ["systemctl", "is-active", svc],
                    capture_output=True, text=True, timeout=5,
                )
                status[svc] = out.stdout.strip() == "active"
            except Exception:
                status[svc] = False
        return status

    def _check_services(self) -> list:
        now = time.time()
        if now - self.last_service_check < 60:
            return []
        self.last_service_check = now
        status = self._services_status()
        alerts = []
        for svc, active in status.items():
            key = f"svc:{svc}"
            was = self.last_state.get(key)
            if active and was is False:
                alerts.append(f"✅ *{svc}* is back active")
            if not active and was is not False:
                alerts.append(f"🚨 *{svc}* is not active")
            self.last_state[key] = active
        return alerts

    def _check_feed(self, coin: str) -> list:
        data = self._api(f"/feed/{coin}")
        if data is None:
            key = f"feed:{coin}"
            fails = self.last_state.get(f"{key}:fails", 0) + 1
            self.last_state[f"{key}:fails"] = fails
            if fails >= 2 and self.last_state.get(key) != "down":
                self.last_state[key] = "down"
                return [f"⚠️ `/feed/{coin}` is unreachable"]
            return []
        self.last_state[f"feed:{coin}:fails"] = 0
        self.last_state[f"feed:{coin}"] = "ok"

        alerts = []
        price_ts = data.get("price_timestamp")
        if price_ts:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(price_ts)).total_seconds()
                key = f"stale:{coin}"
                stale = age > 600
                if stale and not self.last_state.get(key):
                    alerts.append(f"⏱️ {coin.upper()} price is {int(age)}s stale")
                self.last_state[key] = stale
            except Exception:
                pass

        # Alert on directional mismatch: strong positive signal with negative return or vice versa
        for h in [5, 10, 15]:
            epnl = data.get(f"expected_pnl_{h}")
            signal = data.get(f"signal_{h}")
            if signal is not None and epnl is not None:
                if signal > 0.4 and epnl < -0.2:
                    key = f"badsignal:{coin}:{h}"
                    if not self.last_state.get(key):
                        alerts.append(
                            f"⚠️ {coin.upper()} {h}m signal conflict: signal={signal:+.2f} "
                            f"but expected PnL={epnl:+.2f}%"
                        )
                    self.last_state[key] = True
                else:
                    self.last_state[f"badsignal:{coin}:{h}"] = False

        return alerts

    def _check_hl(self) -> list:
        data = self._api("/hl/hype")
        if data is None:
            key = "hl:reach"
            fails = self.last_state.get(f"{key}:fails", 0) + 1
            self.last_state[f"{key}:fails"] = fails
            if fails >= 2 and self.last_state.get(key) != "down":
                self.last_state[key] = "down"
                return ["⚠️ `/hl/hype` is unreachable"]
            return []
        self.last_state["hl:reach:fails"] = 0
        self.last_state["hl:reach"] = "ok"

        latest = data.get("latest_snapshot") or {}
        ts = latest.get("timestamp")
        alerts = []
        if ts:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
                key = "hl:stale"
                stale = age > 240
                if stale and not self.last_state.get(key):
                    alerts.append(f"⏱️ HYPE Hyperliquid snapshot is {int(age)}s stale")
                self.last_state[key] = stale
            except Exception:
                pass

        table = data.get("table_status", {}).get("hl_candles_1m", {})
        last_candle = table.get("last")
        if last_candle:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_candle)).total_seconds()
                key = "hl:candle_stale"
                stale = age > 300
                if stale and not self.last_state.get(key):
                    alerts.append(f"⏱️ HYPE 1m HL candle is {int(age)}s stale")
                self.last_state[key] = stale
            except Exception:
                pass
        return alerts

    def _heartbeat(self) -> list:
        now = time.time()
        if now - self.last_heartbeat < self.heartbeat_min * 60:
            return []
        self.last_heartbeat = now
        return [f"💓 Alerter heartbeat — {datetime.now(timezone.utc).isoformat()[:19]} UTC"]

    def run_once(self) -> list:
        alerts = []
        alerts.extend(self._check_services())
        from .coin_config import list_coins
        for coin in list_coins():
            alerts.extend(self._check_feed(coin))
        alerts.extend(self._check_hl())
        alerts.extend(self._heartbeat())

        for msg in alerts:
            send_telegram(msg)
        return alerts


def main():
    alerter = SystemAlerter()
    # Send a startup notice
    if get_telegram_credentials():
        send_telegram("🛰️ Prediction system alerter started (read-only).")
    else:
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not configured; alerts disabled")
    while True:
        try:
            alerter.run_once()
        except Exception as e:
            logger.error(f"Alerter loop error: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
