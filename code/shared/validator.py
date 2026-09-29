#!/usr/bin/env python3
"""Validate past predictions against actual prices."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from .coin_config import CoinConfig
from .data_store import DataStore
from .utils import setup_logging

logger = setup_logging("Validator")


class Validator:
    """Check old predictions and update them with actual outcomes."""

    def __init__(self, coin_cfg: CoinConfig, data_store: DataStore):
        self.cfg = coin_cfg
        self.symbol = coin_cfg.symbol
        self.data_store = data_store

    def _target_iso(self, prediction_time: str, horizon_min: int) -> str:
        try:
            dt = datetime.fromisoformat(prediction_time.replace('Z', '+00:00'))
        except Exception:
            dt = datetime.now(timezone.utc)
        target = dt + timedelta(minutes=horizon_min)
        return target.isoformat()

    # If no price exists within this long after the target time, the outcome
    # is lost for good (collector outage) and the row is retired.
    GIVE_UP_AFTER_SECONDS = 2 * 3600

    def validate(self, horizon: Optional[int] = None) -> int:
        """Validate pending predictions for the given horizon (or all)."""
        rows = self.data_store.get_unvalidated_predictions()
        validated = 0
        unresolvable = []
        now = datetime.now(timezone.utc)
        for row in rows:
            h = row['horizon_min']
            if horizon is not None and h != horizon:
                continue
            current_price = row['current_price']
            if current_price is None or current_price <= 0:
                unresolvable.append(row['id'])
                continue
            pred_time = row['prediction_time']
            target = self._target_iso(pred_time, h)
            actual = self.data_store.price_at_or_after(target, window_seconds=90)
            if actual is None:
                target_dt = datetime.fromisoformat(target)
                if (now - target_dt).total_seconds() > self.GIVE_UP_AFTER_SECONDS:
                    unresolvable.append(row['id'])
                continue
            actual_price, _ = actual
            current_price = row['current_price']
            predicted_price = row['predicted_price']
            error = actual_price - predicted_price
            if current_price and current_price != 0:
                error_pct = (error / current_price) * 100
            else:
                error_pct = 0.0
            is_correct = 1 if abs(error_pct) <= self.cfg.prediction_threshold_pct else 0
            self.data_store.update_prediction(
                pred_id=row['id'],
                actual_price=actual_price,
                error=error,
                error_pct=error_pct,
                is_correct=is_correct
            )
            validated += 1
            logger.debug(
                f"[{self.symbol}:{h}m] Validated pred {row['id']}: "
                f"predicted={predicted_price:.4f}, actual={actual_price:.4f}, "
                f"error_pct={error_pct:.4f}%"
            )
        if unresolvable:
            n = self.data_store.mark_unresolvable(unresolvable)
            logger.info(f"[{self.symbol}] Retired {n} unresolvable predictions")
        return validated
