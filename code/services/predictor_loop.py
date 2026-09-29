#!/usr/bin/env python3
"""Run predictions, validation, and periodic retraining for one coin."""

import os
import sys
import time
import signal
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.coin_config import get_coin_config
from shared.data_store import DataStore
from shared.feature_engine import FeatureEngine
from shared.model_manager import ModelManager
from shared.predictor_core import Predictor
from shared.validator import Validator
from shared.utils import setup_logging

logger = setup_logging("PredictorLoop")


class PredictorLoop:
    def __init__(self, coin_id: str):
        self.cfg = get_coin_config(coin_id)
        self.ds = DataStore(self.cfg.db_path, self.cfg.symbol)
        self.fe = FeatureEngine(self.ds, self.cfg.symbol)
        if getattr(self.cfg, 'hl_features', False):
            from shared.hl_features import attach_extras
            if attach_extras(self.fe, self.cfg.db_name, micro=getattr(self.cfg, 'hl_micro', False)):
                logger.info(f"[{self.cfg.symbol}] Hyperliquid features enabled "
                            f"(micro={getattr(self.cfg, 'hl_micro', False)})")
        # Cross-asset lead-lag: BTC drives short-horizon moves in BNB/HYPE.
        if coin_id.lower() != 'btc':
            btc_cfg = get_coin_config('btc')
            btc_ds = DataStore(btc_cfg.db_path, btc_cfg.symbol)
            btc_fe = FeatureEngine(btc_ds, btc_cfg.symbol)

            def _btc_bars_for(bars):
                # Live window (last bar is fresh): reuse BTC's 4s bar cache via
                # the limit path instead of re-querying per call.
                try:
                    age = abs(time.time() - bars.index[-1].timestamp())
                except Exception:
                    age = 1e9
                if age < 900:
                    return btc_fe.build_bars(limit=12000)
                # Historical/training window: fetch the exact matching range.
                return btc_fe.build_bars(
                    since=bars.index[0].isoformat(),
                    until=bars.index[-1].isoformat())

            self.fe.cross_bars_fn = _btc_bars_for
            logger.info(f"[{self.cfg.symbol}] BTC lead-lag features enabled")
        self.mm = ModelManager(self.cfg, self.ds, self.fe)
        self.predictor = Predictor(self.cfg, self.ds, self.fe, self.mm)
        self.validator = Validator(self.cfg, self.ds)
        self.running = True
        self._setup_signals()

    def _setup_signals(self):
        def handler(signum, frame):
            self.running = False
            logger.info("Shutdown signal received")
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _predict_all(self):
        for h in self.cfg.prediction_horizons:
            try:
                result = self.predictor.predict(h)
                if result:
                    logger.info(
                        f"[{self.cfg.symbol}:{h}m] Predicted {result['predicted_price']} "
                        f"(change {result['change_percent']}% model={result['model_used']})"
                    )
            except Exception as e:
                logger.error(f"[{self.cfg.symbol}:{h}m] Prediction failed: {e}")
        self.predictor.maybe_log_contract_signal()
        self._log_news()

    def _log_news(self):
        """Persist news-engine features so training frames can join them by
        timestamp (they were previously constant across all history)."""
        try:
            from shared.news_engine import get_news_engine
            self.ds.log_news_features(
                get_news_engine().get_features(self.cfg.symbol))
        except Exception as e:
            logger.debug(f"[{self.cfg.symbol}] news log failed: {e}")

    def _validate(self):
        try:
            n = self.validator.validate()
            if n:
                logger.info(f"[{self.cfg.symbol}] Validated {n} predictions")
            m = self.predictor.resolve_due_signals()
            if m:
                logger.info(f"[{self.cfg.symbol}] Resolved {m} contract signals")
        except Exception as e:
            logger.error(f"[{self.cfg.symbol}] Validation failed: {e}")

    def _maybe_retrain(self):
        try:
            # Snapshot due-ness BEFORE train() consumes the tick counter —
            # otherwise train_classifier's should_train() always sees delta=0
            # and the direction classifiers silently never retrain.
            due = [h for h in self.cfg.prediction_horizons
                   if self.mm.should_train(h)]
            for h in due:
                logger.info(f"[{self.cfg.symbol}:{h}m] Retraining model")
                self.mm.train(h, force=True)
                self.mm.train_classifier(h, force=True)
            if self.mm.should_train_contract():
                logger.info(f"[{self.cfg.symbol}] Retraining Kalshi contract model")
                self.mm.train_contract(force=False)
        except Exception as e:
            logger.error(f"[{self.cfg.symbol}] Retraining failed: {e}")

    def run(self):
        logger.info(f"{self.cfg.symbol} predictor loop started")
        last_predict = 0
        last_validate = 0
        last_retrain = 0
        while self.running:
            now = time.time()
            if now - last_validate >= 30:
                self._validate()
                last_validate = now
            if now - last_predict >= 15:
                self._predict_all()
                last_predict = now
            if now - last_retrain >= 1800:
                self._maybe_retrain()
                last_retrain = now
            time.sleep(2)
        logger.info(f"{self.cfg.symbol} predictor loop stopped")


def main():
    coin_id = sys.argv[1] if len(sys.argv) > 1 else 'btc'
    loop = PredictorLoop(coin_id)
    loop.run()


if __name__ == '__main__':
    main()
