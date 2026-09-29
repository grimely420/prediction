#!/usr/bin/env python3
"""
Kalshi 15-Minute Round Tracker & Decision Engine.
Aligns predictions with Kalshi's fixed quarter-hour cycles (:00, :15, :30, :45),
tracks CF Benchmarks Real-Time Index (BRTI & RTIs) round open and current delta,
and generates actionable trading signals for the current and upcoming 5-17 minute rounds.
"""

import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from .utils import setup_logging

logger = setup_logging("KalshiEngine")


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class KalshiRoundTracker:
    """Manages round boundaries, round open strike, and probability calculations."""

    SERIES_MAP = {
        "btc": "KXBTC15M",
        "bnb": "KXBNB15M",
        "hype": "KXHYPE15M",
        "eth": "KXETH15M",
        "sol": "KXSOL15M",
        "xrp": "KXXRP15M",
        "doge": "KXDOGE15M",
        "near": "KXNEAR15M",
        "zec": "KXZEC15M"
    }

    RTI_NAME_MAP = {
        "btc": "CME CF BRTI (Bitcoin RTI)",
        "bnb": "CF Benchmarks BNB RTI",
        "hype": "CF Benchmarks HYPE RTI"
    }

    def __init__(self, data_store, coin_id: str):
        self.data_store = data_store
        self.coin_id = coin_id.lower()
        self.series = self.SERIES_MAP.get(self.coin_id, f"KX{self.coin_id.upper()}15M")
        self.rti_name = self.RTI_NAME_MAP.get(self.coin_id, f"CFB {self.coin_id.upper()} RTI")
        self._analysis_cache: Optional[Dict[str, Any]] = None
        self._analysis_cache_time: float = 0.0
        self._prev_prob: Dict[str, float] = {}
        self._last_round_start: Optional[str] = None
        self._calib_stats: Optional[Dict[str, Any]] = None
        self._calib_time: float = 0.0

    def _calib_win_rate(self, side_prob: float) -> Optional[float]:
        """Empirical win rate for the probability bucket containing side_prob.

        Reads the resolved contract-signal calibration table (cached 60s).
        Returns None when the bucket is too thin to trust (<15 samples).
        """
        now = time.time()
        if now - self._calib_time > 60.0 or self._calib_stats is None:
            try:
                self._calib_stats = self.data_store.get_contract_signal_stats()
                self._calib_time = now
            except Exception:
                return None
        buckets = (self._calib_stats or {}).get('calibration') or {}
        for rng, b in buckets.items():
            try:
                lo, hi = (float(x) for x in str(rng).split('-'))
            except Exception:
                continue
            if lo <= side_prob * 100.0 < hi:
                n = b.get('n') or 0
                w = b.get('empirical_win_pct')
                if n >= 15 and w is not None:
                    return float(w) / 100.0
                return None
        return None

    def get_round_timing(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Compute current and next 15-minute round timing aligned to :00, :15, :30, :45."""
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Current round
        minute_slot = (now.minute // 15) * 15
        round_start = now.replace(minute=minute_slot, second=0, microsecond=0)
        round_end = round_start + timedelta(minutes=15)

        elapsed_sec = max(0, int((now - round_start).total_seconds()))
        remaining_sec = max(0, int((round_end - now).total_seconds()))
        elapsed_min = elapsed_sec / 60.0
        remaining_min = remaining_sec / 60.0
        progress_pct = round(min(100.0, (elapsed_sec / 900.0) * 100.0), 1)

        # Next round (upcoming 15m cycle, 5-17 mins ahead)
        next_start = round_end
        next_end = next_start + timedelta(minutes=15)

        return {
            "current_round": {
                "start_time": round_start.isoformat(),
                "end_time": round_end.isoformat(),
                "label": f"{round_start.strftime('%H:%M')} – {round_end.strftime('%H:%M')} UTC",
                "elapsed_seconds": elapsed_sec,
                "remaining_seconds": remaining_sec,
                "elapsed_minutes": round(elapsed_min, 2),
                "remaining_minutes": round(remaining_min, 2),
                "progress_pct": progress_pct,
                "is_settlement_minute": remaining_sec <= 60
            },
            "next_round": {
                "start_time": next_start.isoformat(),
                "end_time": next_end.isoformat(),
                "label": f"{next_start.strftime('%H:%M')} – {next_end.strftime('%H:%M')} UTC",
                "starts_in_seconds": remaining_sec,
                "starts_in_minutes": round(remaining_min, 2)
            }
        }

    def get_round_open_price(self, round_start_iso: str) -> Optional[float]:
        """Find the RTI price at or nearest after the start of the 15m round."""
        try:
            # Query price at or immediately after round start (within 90s)
            res = self.data_store.price_at_or_after(round_start_iso, window_seconds=90)
            if res:
                return float(res[0])
            # Fallback to nearest before
            res_before = self.data_store.price_at_or_before(round_start_iso, window_seconds=90)
            if res_before:
                return float(res_before[0])
        except Exception as e:
            logger.warning(f"Error getting round open price: {e}")
        return None

    def get_realized_volatility(self) -> float:
        """Live realized 1-minute volatility, correctly scaled from tick cadence.

        The last ~90 ticks are ~2.3s apart, so their per-tick return std is
        a *per-tick* vol — it must be scaled by sqrt(60/cadence) to become a
        per-minute vol before being used in sigma * sqrt(minutes). Previously
        the raw per-tick std was used directly, understating sigma ~5x and
        over-inflating z-scores into false 90c+ conviction.
        """
        try:
            records = self.data_store.get_prices(limit=90)
            if records and len(records) >= 15:
                prices = [float(r['price']) for r in records if r.get('price')]
                # actual sampling cadence in seconds
                t_first = datetime.fromisoformat(
                    str(records[0]['timestamp']).replace('Z', '+00:00')).timestamp()
                t_last = datetime.fromisoformat(
                    str(records[-1]['timestamp']).replace('Z', '+00:00')).timestamp()
                cadence = (t_last - t_first) / max(1, len(records) - 1)
                cadence = max(0.5, min(30.0, cadence))
                if len(prices) >= 15:
                    rets = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                    mean_r = sum(rets) / len(rets)
                    var_r = sum((r - mean_r) ** 2 for r in rets) / len(rets)
                    std_tick = math.sqrt(var_r)
                    std_1m = std_tick * math.sqrt(60.0 / cadence)
                    if 0.0002 <= std_1m <= 0.02:
                        return std_1m
        except Exception as e:
            logger.debug(f"Error calculating realized vol: {e}")
        return {
            "btc": 0.0008, "eth": 0.0011, "sol": 0.0014,
            "bnb": 0.0012, "xrp": 0.0014, "doge": 0.0016,
            "hype": 0.0018, "near": 0.0016, "zec": 0.0020,
        }.get(self.coin_id, 0.0010)

    def _tick_volatility(self, limit: int = 120) -> float:
        """Per-tick return volatility for the settlement-TWAP engine."""
        try:
            records = self.data_store.get_prices(limit=limit)
            if records and len(records) >= 10:
                prices = [float(r['price']) for r in records if r.get('price')]
                rets = [(prices[i] - prices[i-1]) / prices[i-1]
                        for i in range(1, len(prices)) if prices[i-1] > 0]
                if len(rets) >= 8:
                    mean_r = sum(rets) / len(rets)
                    var_r = sum((r - mean_r) ** 2 for r in rets) / len(rets)
                    return max(1e-6, math.sqrt(var_r))
        except Exception:
            pass
        return 0.0003

    def _settlement_projection(self, cur_rnd: Dict[str, Any],
                               current_price: float,
                               round_open: float) -> Optional[Dict[str, Any]]:
        """Kalshi final-60s settlement TWAP engine.

        Kalshi settles on the time-weighted average RTI over the last 60
        seconds of the round. Once that window opens, the observed samples are
        locked into the average — only the unobserved tail is uncertain, so the
        contract probability converges to certainty as the clock runs out.

        Returns None when we are more than 75s from settlement.
        """
        remaining_sec = cur_rnd.get("remaining_seconds", 999)
        if remaining_sec > 75:
            return None
        try:
            round_end = datetime.fromisoformat(cur_rnd["end_time"])
            window_start = (round_end - timedelta(seconds=60)).isoformat()
            ticks = self.data_store.get_prices(since=window_start)
            prices = [float(r['price']) for r in ticks if r.get('price')]
            if not prices:
                return None

            n_obs = len(prices)
            twap_partial = sum(prices) / n_obs

            # Estimate ticks remaining: assume ~current collection cadence
            elapsed_in_window = max(1.0, 60.0 - remaining_sec)
            cadence = max(2.0, elapsed_in_window / max(1, n_obs))
            n_rem = max(0.0, remaining_sec / cadence)

            # Future ticks centred on current price; only they carry variance
            final_est = (twap_partial * n_obs + current_price * n_rem) / (n_obs + n_rem)
            sigma_tick = self._tick_volatility()
            std_final = sigma_tick * math.sqrt(max(n_rem, 0.0)) / (n_obs + n_rem) * current_price
            std_final = max(std_final, 1e-6)

            z = (final_est - round_open) / std_final
            p_up = normal_cdf(z)
            return {
                'p_up': max(0.001, min(0.999, p_up)),
                'twap_partial': round(twap_partial, 4),
                'twap_projected': round(final_est, 4),
                'ticks_observed': n_obs,
                'est_ticks_remaining': round(n_rem, 1),
                'std_remaining': round(std_final, 4),
                'seconds_left': remaining_sec,
            }
        except Exception as e:
            logger.debug(f"settlement projection error: {e}")
            return None

    def analyze_round(self,
                      current_price: float,
                      predicted_returns: Dict[int, float],
                      classifier_probs: Dict[int, Dict[str, float]],
                      news_features: Optional[Dict[str, float]] = None,
                      contract_prob: Optional[Dict[str, Any]] = None,
                      now: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate full Kalshi round analysis, probabilities, and actionable trading decision."""
        now_ts = time.time()
        if now is None and (now_ts - self._analysis_cache_time < 2.5) and self._analysis_cache is not None:
            return self._analysis_cache

        timing = self.get_round_timing(now)
        cur_rnd = timing["current_round"]
        next_rnd = timing["next_round"]

        round_open = self.get_round_open_price(cur_rnd["start_time"])
        if round_open is None or round_open <= 0:
            round_open = current_price

        # Current Delta inside active round
        delta_usd = current_price - round_open
        delta_pct = (delta_usd / round_open) * 100.0 if round_open > 0 else 0.0

        # News contribution
        news_sentiment = 0.0
        if news_features:
            news_sentiment = news_features.get("news_compound_signal", 0.0)

        # Dynamic live realized 1-minute volatility from recent market ticks
        base_vol_1m = self.get_realized_volatility()

        rem_min = max(0.2, cur_rnd["remaining_minutes"])
        sigma_rem = base_vol_1m * math.sqrt(rem_min)

        # Model predicted return for the remaining span (interpolate between 5m, 10m, 15m)
        pred_15 = predicted_returns.get(15, 0.0)
        pred_10 = predicted_returns.get(10, 0.0)
        pred_5 = predicted_returns.get(5, 0.0)

        # Weight predictions by remaining time
        if rem_min <= 5:
            model_drift = pred_5 * (rem_min / 5.0)
        elif rem_min <= 10:
            weight_5 = (10 - rem_min) / 5.0
            model_drift = weight_5 * pred_5 + (1.0 - weight_5) * pred_10
        else:
            weight_10 = (15 - rem_min) / 5.0
            model_drift = weight_10 * pred_10 + (1.0 - weight_10) * pred_15

        # News drift impact over remaining span
        news_drift = (news_sentiment * 0.0015) * (rem_min / 15.0)

        # Total expected price change from NOW to settlement
        total_expected_return = model_drift + news_drift
        projected_settlement_price = current_price * (1.0 + total_expected_return)

        # Distance from Strike (Log Return)
        log_ratio = math.log(max(1e-4, current_price) / max(1e-4, round_open))

        # Binary normalization of classifier (removes ternary 'flat' dilution)
        clf_15 = classifier_probs.get(15, {})
        up_raw = clf_15.get('up', 0.33)
        down_raw = clf_15.get('down', 0.33)
        total_dir = up_raw + down_raw
        norm_clf_up = (up_raw / total_dir) if total_dir > 0.05 else 0.50

        # Normalized multi-horizon drift: model + news + classifier
        model_bias = (pred_5 + pred_10 + pred_15) / 3.0
        news_bias = news_sentiment * 0.0015
        clf_bias = (norm_clf_up - 0.50) * 0.0025
        total_drift = model_bias + news_bias + clf_bias

        # Continuous Digital Option Model (theoretical prior):
        # z = [ln(S_t/S_0) + drift * (tau/15)] / sigma_rem
        z_score = (log_ratio + total_drift * (rem_min / 15.0)) / max(1e-5, sigma_rem)
        bs_prob_up = normal_cdf(z_score)

        # Settlement TWAP engine takes over inside the final 75 seconds —
        # observed samples are locked into the settlement average, so this
        # dominates every other signal once the window opens.
        settlement = self._settlement_projection(cur_rnd, current_price, round_open)
        prob_source = 'bs_digital'
        if settlement is not None:
            raw_prob_up = settlement['p_up']
            prob_source = 'settlement_twap'
        elif contract_prob and contract_prob.get('p_up') is not None:
            # Strike-aware learned model is primary; BS digital is the prior.
            # Scale contract weight by data maturity: a coin with only a few
            # hours of 1m bars gets a model we don't fully trust yet; ~33h+
            # of bars (2000) earns the full 65% weight.
            n_bars = contract_prob.get('train_bars')
            maturity = 1.0 if n_bars is None else min(1.0, n_bars / 2000.0)
            w_contract = 0.65 * maturity
            raw_prob_up = w_contract * contract_prob['p_up'] + (1.0 - w_contract) * bs_prob_up
            prob_source = 'contract_model'
        else:
            raw_prob_up = bs_prob_up

        # Adaptive EMA: fast tracking on real moves, smoothing on tick noise.
        # alpha scales with how far the raw probability jumped this tick.
        round_changed = self._last_round_start != cur_rnd["start_time"]
        if round_changed:
            self._last_round_start = cur_rnd["start_time"]
            self._prev_prob.pop(self.coin_id, None)
        prev_p = self._prev_prob.get(self.coin_id)
        if prev_p is not None and cur_rnd["elapsed_seconds"] > 10:
            alpha = min(0.95, max(0.35, abs(raw_prob_up - prev_p) / 0.08))
            prob_up = alpha * raw_prob_up + (1.0 - alpha) * prev_p
        else:
            prob_up = raw_prob_up
        self._prev_prob[self.coin_id] = prob_up

        prob_up = max(0.01, min(0.99, prob_up))
        prob_down = round(1.0 - prob_up, 4)
        prob_up = round(prob_up, 4)

        # Exact Kalshi Cents Pricing (1c to 99c)
        yes_cents = int(round(prob_up * 100))
        no_cents = 100 - yes_cents

        edge = round(abs(prob_up - 0.50) * 100.0, 1)
        confidence = round(edge * 2.0, 1)

        elapsed_min = cur_rnd["elapsed_minutes"]
        is_entry_window = elapsed_min <= 6.5
        is_settlement_window = cur_rnd["remaining_seconds"] <= 75
        phase_label = ("EARLY ENTRY WINDOW (Mins 0–6)" if is_entry_window
                       else "FINAL SETTLEMENT TWAP (Last 75s)" if is_settlement_window
                       else "SETTLEMENT TRACKING (Mins 7–15)")

        # Do the strike-aware contract model and the theoretical digital model
        # agree on direction? Agreement is a meaningful confidence booster.
        model_agreement = True
        if contract_prob and contract_prob.get('p_up') is not None:
            n_bars = contract_prob.get('train_bars')
            mature = n_bars is None or n_bars >= 1000
            model_agreement = (contract_prob['p_up'] - 0.5) * (bs_prob_up - 0.5) >= 0 if mature else True

        # Kalshi entry caps: the maximum price worth paying while keeping a
        # 3-cent edge buffer (covers fees ~1.75¢ max + slippage + margin).
        edge_buffer = 3
        max_yes_entry = max(1, yes_cents - edge_buffer)
        max_no_entry = max(1, no_cents - edge_buffer)

        if is_settlement_window and settlement is not None:
            if prob_up >= 0.90:
                decision = f"SETTLEMENT LOCK: YES ({yes_cents}¢)"
                bias, color = "UP", "#10b981"
            elif prob_up <= 0.10:
                decision = f"SETTLEMENT LOCK: NO ({no_cents}¢)"
                bias, color = "DOWN", "#ef4444"
            else:
                decision = f"SETTLEMENT UNCERTAIN ({yes_cents}¢/{no_cents}¢)"
                bias, color = "NEUTRAL", "#f59e0b"
        elif yes_cents >= 60:
            decision = f"BUY YES (TARGET {yes_cents}¢)"
            bias = "UP"
            color = "#10b981" if model_agreement else "#34d399"
        elif yes_cents >= 54:
            decision = f"LEAN BUY YES ({yes_cents}¢)"
            bias = "UP"
            color = "#34d399"
        elif no_cents >= 60:
            decision = f"BUY NO (TARGET {no_cents}¢)"
            bias = "DOWN"
            color = "#ef4444" if model_agreement else "#f87171"
        elif no_cents >= 54:
            decision = f"LEAN BUY NO ({no_cents}¢)"
            bias = "DOWN"
            color = "#f87171"
        else:
            decision = f"STRIKE CHOP ({yes_cents}¢ / {no_cents}¢)"
            bias = "NEUTRAL"
            color = "#f59e0b"

        # Calibration gate: the contract-signal ledger knows which cent
        # buckets have actually been winning. If this side's probability lands
        # in a bucket that empirically wins <52% (coin flip + fees = losing
        # trade), suppress the BUY and mark it as a measured no-edge zone.
        calib_wr = None
        if not is_settlement_window and 'BUY' in decision:
            side_p = prob_up if yes_cents >= 50 else prob_down
            calib_wr = self._calib_win_rate(side_p)
            if calib_wr is not None and calib_wr < 0.52:
                decision = f"SKIP — BUCKET WIN {calib_wr * 100:.0f}% ({yes_cents}¢/{no_cents}¢)"
                bias = "NEUTRAL"
                color = "#9ca3af"

        # Analysis of Next Upcoming Round (5-17 minute span)
        next_model_ret = pred_15
        next_news_impact = news_sentiment * 0.0020
        next_composite_ret = round((next_model_ret + next_news_impact) * 100.0, 4)

        if next_composite_ret > 0.06:
            next_bias = "BULLISH (UP)"
            next_action = "PREPARE BUY YES"
        elif next_composite_ret < -0.06:
            next_bias = "BEARISH (DOWN)"
            next_action = "PREPARE BUY NO"
        else:
            next_bias = "RANGE-BOUND"
            next_action = "WAIT FOR BREAKOUT"

        result = {
            "series": self.series,
            "benchmark_index": self.rti_name,
            "timing": timing,
            "phase": {
                "label": phase_label,
                "is_entry_window": is_entry_window,
                "is_settlement_window": is_settlement_window,
            },
            "pricing": {
                "current_price": round(current_price, 4),
                "round_open_price": round(round_open, 4),
                "delta_usd": round(delta_usd, 4),
                "delta_pct": round(delta_pct, 4),
                "projected_settlement": round(projected_settlement_price, 4),
                "yes_cents": yes_cents,
                "no_cents": no_cents,
            },
            "settlement": settlement,
            "contract_model": {
                "p_up": round(contract_prob['p_up'], 4) if contract_prob else None,
                "val_acc": contract_prob.get('val_acc') if contract_prob else None,
                "val_brier": contract_prob.get('val_brier') if contract_prob else None,
            },
            "decision": {
                "recommendation": decision,
                "bias": bias,
                "phase_label": phase_label,
                "is_entry_window": is_entry_window,
                "yes_cents": yes_cents,
                "no_cents": no_cents,
                "confidence_pct": confidence,
                "badge_color": color,
                "prob_yes_up": prob_up,
                "prob_no_down": prob_down,
                "edge_pct": edge,
                "model_drift_pct": round(model_drift * 100.0, 4),
                "news_drift_pct": round(news_drift * 100.0, 4),
                "prob_source": prob_source,
                "model_agreement": model_agreement,
                "fair_yes_cents": yes_cents,
                "fair_no_cents": no_cents,
                "max_yes_entry_cents": max_yes_entry,
                "max_no_entry_cents": max_no_entry,
                "min_edge_cents": edge_buffer,
                "bucket_win_pct": round(calib_wr * 100.0, 1) if calib_wr is not None else None,
            },
            "next_round_outlook": {
                "label": next_rnd["label"],
                "bias": next_bias,
                "suggested_action": next_action,
                "projected_change_pct": next_composite_ret,
                "starts_in_min": next_rnd["starts_in_minutes"]
            }
        }
        self._analysis_cache = result
        self._analysis_cache_time = time.time()
        return result
