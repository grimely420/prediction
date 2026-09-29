#!/usr/bin/env python3
"""Single-coin, single-horizon prediction endpoint."""

import os
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .coin_config import CoinConfig
from .data_store import DataStore
from .feature_engine import FeatureEngine
from .model_manager import ModelManager
from .news_engine import get_news_engine
from .kalshi_engine import KalshiRoundTracker
from .utils import setup_logging

logger = setup_logging("PredictorCore")


class Predictor:
    """Generate and log a price prediction for one coin and one horizon."""

    def __init__(self, coin_cfg: CoinConfig, data_store: DataStore,
                 feature_engine: FeatureEngine, model_manager: ModelManager):
        self.cfg = coin_cfg
        self.symbol = coin_cfg.symbol
        self.data_store = data_store
        self.feature_engine = feature_engine
        self.model_manager = model_manager
        # Short-lived in-memory caches: API/dashboard polls reuse the latest
        # prediction instead of re-computing and re-logging DB rows on every hit.
        self._pred_cache: Dict[int, Tuple[float, Dict[str, Any]]] = {}
        self._contract_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self.pred_cache_ttl = float(os.environ.get('PRED_CACHE_TTL', '10'))
        self._round_tracker: Optional[KalshiRoundTracker] = None

    def _kt(self) -> KalshiRoundTracker:
        if self._round_tracker is None:
            self._round_tracker = KalshiRoundTracker(self.data_store, self.symbol)
        return self._round_tracker

    # ------------------------------------------------------------------
    # Live contract-signal scoring (measure the real Kalshi edge)
    # ------------------------------------------------------------------

    def maybe_log_contract_signal(self) -> None:
        """Log one contract call per round per phase bucket (entry/mid/late).

        INSERT OR IGNORE on (round_start, phase) makes this idempotent across
        the 15s predictor loop cadence; each phase's first call wins.
        """
        try:
            timing = self._kt().get_round_timing()
            cur = timing['current_round']
            elapsed_min = cur['elapsed_minutes']
            phase = ('entry' if elapsed_min <= 6.5
                     else 'mid' if elapsed_min <= 13.0
                     else 'late')
            cp = self.get_contract_prob()
            if cp is None:
                return
            current_price = self._current_price()
            if current_price is None:
                return
            strike = self._kt().get_round_open_price(cur['start_time'])
            if strike is None or strike <= 0:
                return
            p = float(cp['p_up'])
            bias = 'UP' if p >= 0.54 else ('DOWN' if p <= 0.46 else 'NEUTRAL')
            logged = self.data_store.log_contract_signal(
                round_start=cur['start_time'],
                round_end=cur['end_time'],
                phase=phase,
                mins_elapsed=elapsed_min,
                mins_remaining=cur['remaining_minutes'],
                strike_price=strike,
                current_price=current_price,
                prob_yes=round(p, 4),
                bias=bias,
                source='contract_model',
                model_version=str(cp.get('model_version', '')),
            )
            if logged:
                logger.info(
                    f"[{self.symbol}] Contract signal {phase}: "
                    f"p_yes={p:.3f} bias={bias} strike={strike:.4f}"
                )
        except Exception as e:
            logger.debug(f"[{self.symbol}] contract signal log failed: {e}")

    def _settlement_twap(self, round_end_iso: str) -> Optional[float]:
        """Mean of ticks inside the round's final 60s — Kalshi settlement proxy."""
        try:
            from datetime import datetime, timedelta
            end = datetime.fromisoformat(round_end_iso.replace('Z', '+00:00'))
            start = (end - timedelta(seconds=60)).isoformat()
            ticks = self.data_store.get_prices(since=start, until=round_end_iso)
            prices = [float(r['price']) for r in ticks if r.get('price')]
            if prices:
                return sum(prices) / len(prices)
        except Exception as e:
            logger.debug(f"[{self.symbol}] settlement twap error: {e}")
        return None

    def resolve_due_signals(self) -> int:
        """Resolve logged signals whose round has fully settled."""
        try:
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(seconds=90)).isoformat()
            rows = self.data_store.get_unresolved_signals(cutoff)
            resolved = 0
            for r in rows:
                settle = self._settlement_twap(r['round_end'])
                if settle is None:
                    back = self.data_store.price_at_or_before(
                        r['round_end'], window_seconds=120)
                    if back:
                        settle = float(back[0])
                if settle is None:
                    continue
                strike = float(r['strike_price'])
                outcome = 1 if settle > strike else 0
                bias = r['bias']
                is_correct = None
                if bias in ('UP', 'DOWN'):
                    is_correct = int((bias == 'UP') == (outcome == 1))
                self.data_store.resolve_contract_signal(
                    r['id'], settle, outcome, is_correct)
                resolved += 1
                logger.info(
                    f"[{self.symbol}] Signal resolved: {r['phase']} "
                    f"bias={bias} strike={strike:.4f} settle={settle:.4f} "
                    f"outcome={'UP' if outcome else 'DOWN'} correct={is_correct}"
                )
            return resolved
        except Exception as e:
            logger.error(f"[{self.symbol}] resolve_due_signals failed: {e}")
            return 0

    def _current_price(self) -> Optional[float]:
        last = self.data_store.last_price()
        if last is not None:
            return float(last[0])
        return None

    def _trend_fallback(self) -> float:
        try:
            records = self.data_store.get_prices(limit=60)
            if not records or len(records) < 5:
                return 0.0
            prices = np.array([float(r['price']) for r in records])
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            if prices[-1]:
                return float(slope / prices[-1])
        except Exception as e:
            logger.debug(f"[{self.symbol}] trend fallback error: {e}")
        return 0.0

    def get_contract_prob(self) -> Optional[Dict[str, Any]]:
        """Calibrated P(current Kalshi 15m round settles above its strike).

        Uses the dedicated strike-aware contract model (dist_to_strike +
        mins_remaining features). Returns None when no model is available.
        """
        now_ts = time.time()
        if now_ts - self._contract_cache[0] < 5.0 and self._contract_cache[1] is not None:
            return self._contract_cache[1]

        info = self.model_manager.load_contract()
        if info is None:
            return None
        features, feature_names = self.feature_engine.get_current_contract_features()
        if features is None:
            return None
        fnames = info.get('feature_names') or feature_names
        if len(features) != len(fnames):
            logger.warning(f"[{self.symbol}] Contract feature mismatch at inference")
            return None
        try:
            X = np.array(features).reshape(1, -1)
            prob = info['model'].predict_proba(X)[0]
            le = info.get('le')
            class_to_prob = {int(c): float(p) for c, p in zip(le.classes_, prob)} if le is not None \
                else {0: float(prob[0]), 1: float(prob[1])}
            result = {
                'p_up': class_to_prob.get(1, 0.5),
                'p_down': class_to_prob.get(0, 0.5),
                'model_version': str(int(os.path.getmtime(info['path']))),
                'val_acc': (info.get('metrics') or {}).get('val_acc'),
                'val_brier': (info.get('metrics') or {}).get('val_brier'),
                'train_bars': getattr(self.feature_engine, '_live_bars_count', None),
            }
            self._contract_cache = (now_ts, result)
            return result
        except Exception as e:
            logger.error(f"[{self.symbol}] Contract prediction error: {e}")
            return None

    def predict(self, horizon: int, current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        cached = self._pred_cache.get(horizon)
        if cached is not None and (time.time() - cached[0]) < self.pred_cache_ttl:
            return cached[1]

        if current_price is None:
            current_price = self._current_price()
        if current_price is None:
            logger.error(f"[{self.symbol}:{horizon}m] No current price")
            return None

        if current_price < self.cfg.min_price or current_price > self.cfg.max_price:
            logger.warning(f"[{self.symbol}:{horizon}m] Price {current_price} outside sanity range")

        model_info = self.model_manager.load(horizon)
        clf_info = self.model_manager.load_classifier(horizon)
        features, feature_names = self.feature_engine.get_current_features(horizon=horizon)

        fallback = False
        model_used = 'xgboost'
        model_version = 'v1'
        val_mae = None
        raw_return = 0.0
        predicted_return = 0.0
        signal_strength = 1.0
        probs = {'up': 1/3, 'down': 1/3, 'flat': 1/3}
        expected_pnl_pct = -0.05

        if model_info is None or features is None:
            fallback = True
            model_used = 'fallback'
        else:
            try:
                mdl = model_info['model']
                fnames = model_info.get('feature_names') or feature_names
                val_mae = model_info.get('val_mae')

                if len(features) != len(fnames):
                    logger.warning(f"[{self.symbol}:{horizon}m] Feature mismatch, retraining")
                    if self.model_manager.train(horizon, force=True):
                        model_info = self.model_manager.load(horizon)
                    if model_info is None:
                        fallback = True
                        model_used = 'fallback'
                    else:
                        mdl = model_info['model']
                        fnames = model_info.get('feature_names') or feature_names
                        val_mae = model_info.get('val_mae')

                if not fallback:
                    X = np.array(features).reshape(1, -1)
                    raw_return = float(mdl.predict(X)[0])
                    model_version = str(int(os.path.getmtime(model_info['path'])))

                    # Blend with classifier probability
                    if clf_info is not None:
                        clf = clf_info['model']
                        le = clf_info['le']
                        prob = clf.predict_proba(X)[0]
                        class_to_prob = {int(c): float(p) for c, p in zip(le.classes_, prob)}
                        if len(le.classes_) == 2 and 0 in class_to_prob and 1 in class_to_prob:
                            probs = {
                                'up': class_to_prob.get(1, 0.5),
                                'down': class_to_prob.get(0, 0.5),
                                'flat': 0.0,
                            }
                        else:
                            probs = {
                                'up': class_to_prob.get(1, 0.0),
                                'down': class_to_prob.get(-1, 0.0),
                                'flat': class_to_prob.get(0, 0.0),
                            }
                        signal_strength = probs['up'] - probs['down']
                        # Correct directional consensus (NEVER invert negative * negative):
                        reg_dir = 1.0 if raw_return > 0 else (-1.0 if raw_return < 0 else 0.0)
                        clf_dir = 1.0 if signal_strength > 0.04 else (-1.0 if signal_strength < -0.04 else 0.0)

                        if clf_dir != 0.0:
                            # If classifier has directional conviction, sign follows classifier
                            final_sign = clf_dir
                            # If regressor agrees, amplify magnitude; if it disagrees, dampen
                            mult = (1.0 + abs(signal_strength)) if reg_dir == clf_dir else max(0.2, 1.0 - abs(signal_strength))
                            mag = max(abs(raw_return), abs(signal_strength) * 0.003) * mult
                            predicted_return = float(np.clip(final_sign * mag, -0.05, 0.05))
                        else:
                            # Classifier is neutral, follow regressor with dampening
                            predicted_return = float(np.clip(raw_return * (1.0 - probs.get('flat', 0.3) * 0.5), -0.05, 0.05))
                    else:
                        predicted_return = float(np.clip(raw_return, -0.05, 0.05))
            except Exception as e:
                logger.error(f"[{self.symbol}:{horizon}m] Prediction error: {e}")
                fallback = True
                model_used = 'fallback'

        if fallback:
            predicted_return = self._trend_fallback()

        # Factor in real-time crypto news sentiment for short-horizon spans (5-17 mins)
        news_signal = 0.0
        try:
            news_feats = get_news_engine().get_features(self.symbol)
            news_signal = news_feats.get('news_compound_signal', 0.0)
            # 5-17 minute impact drift: up to ~0.15% (0.0015) based on news sentiment
            news_drift = news_signal * 0.0015 * (horizon / 15.0)
            predicted_return = float(np.clip(predicted_return + news_drift, -0.05, 0.05))
        except Exception as e:
            logger.debug(f"News drift calculation failed: {e}")

        fee_pct = getattr(self.cfg, 'fee_pct', 0.05)
        predicted_return = float(np.clip(predicted_return, -0.05, 0.05))
        predicted_price = float(current_price * (1.0 + predicted_return))
        change_pct = round(predicted_return * 100, 4)
        raw_change_pct = round(np.clip(raw_return, -0.05, 0.05) * 100, 4)
        expected_pnl_pct = round(predicted_return * 100 - fee_pct, 4)

        if val_mae is None or not np.isfinite(val_mae):
            val_mae = 0.002
        width = float(max(0.01, val_mae * current_price))

        pred_id = self.data_store.log_prediction(
            horizon_min=horizon,
            current_price=current_price,
            predicted_price=predicted_price,
            model_used=model_used,
            model_version=model_version
        )

        result = {
            'success': True,
            'coin_id': self.cfg.db_name,
            'symbol': self.symbol,
            'horizon_minutes': horizon,
            'current_price': round(float(current_price), 4),
            'predicted_price': round(float(predicted_price), 4),
            'change_percent': float(change_pct),
            'raw_change_percent': float(raw_change_pct),
            'signal_strength': round(float(signal_strength), 4),
            'news_sentiment': round(float(news_signal), 4),
            'prob_up': round(float(probs['up']), 4),
            'prob_down': round(float(probs['down']), 4),
            'prob_flat': round(float(probs['flat']), 4),
            'expected_pnl_pct': float(expected_pnl_pct),
            'confidence_interval': [
                round(float(predicted_price - width), 4),
                round(float(predicted_price + width), 4)
            ],
            'model_used': model_used,
            'model_version': model_version,
            'threshold_pct': self.cfg.prediction_threshold_pct,
            'prediction_id': pred_id
        }
        self._pred_cache[horizon] = (time.time(), result)
        return result
