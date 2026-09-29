#!/usr/bin/env python3
"""
Real-time Crypto News Sentiment & Intelligence Engine for short-horizon predictions (5-17 mins).
Fetches breaking crypto news from top RSS feeds, calculates weighted sentiment scores,
and extracts market-moving signals for BTC, BNB, and HYPE.
"""

import re
import time
import threading
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Any, Optional

from .utils import setup_logging

logger = setup_logging("NewsEngine")

# Bullish and Bearish Financial/Crypto Lexicon with weights
BULLISH_KEYWORDS = {
    "surge": 1.5, "surges": 1.5, "surging": 1.5,
    "rally": 1.6, "rallies": 1.6, "rallying": 1.6,
    "breakout": 1.8, "breaks out": 1.8, "ath": 2.0, "all-time high": 2.0,
    "bull": 1.2, "bullish": 1.5, "bulls": 1.2,
    "soar": 1.6, "soars": 1.6, "soaring": 1.6,
    "gain": 1.1, "gains": 1.1, "gaining": 1.1,
    "pump": 1.4, "skyrocket": 2.0, "skyrockets": 2.0,
    "inflow": 1.4, "inflows": 1.4, "record inflow": 2.0,
    "approval": 1.8, "approved": 1.8, "etf approved": 2.5,
    "rate cut": 1.8, "rate cuts": 1.8, "easing": 1.2,
    "adoption": 1.3, "accumulate": 1.3, "accumulation": 1.4,
    "bounce": 1.2, "bounces": 1.2, "recovery": 1.3, "recovers": 1.3,
    "outperform": 1.4, "milestone": 1.2, "partnership": 1.2,
    "upgrade": 1.2, "expansion": 1.1, "bull run": 1.8,
}

BEARISH_KEYWORDS = {
    "crash": 2.0, "crashes": 2.0, "crashing": 2.0,
    "plunge": 1.8, "plunges": 1.8, "plunging": 1.8,
    "dump": 1.6, "dumps": 1.6, "dumping": 1.6,
    "slump": 1.5, "slumps": 1.5, "tank": 1.7, "tanks": 1.7,
    "bear": 1.2, "bearish": 1.5, "bears": 1.2,
    "hack": 2.2, "hacked": 2.2, "exploit": 2.0, "exploited": 2.0, "stolen": 2.0,
    "outflow": 1.4, "outflows": 1.4, "record outflow": 2.0,
    "rejection": 1.7, "rejected": 1.7, "ban": 2.0, "banned": 2.0,
    "crackdown": 1.8, "lawsuit": 1.8, "sues": 1.8, "sued": 1.8,
    "sec charges": 2.2, "indictment": 2.2, "investigation": 1.5,
    "rate hike": 1.8, "inflation climbs": 1.5,
    "liquidation": 1.7, "liquidations": 1.8, "liquidated": 1.8,
    "selloff": 1.7, "sell-off": 1.7, "capitulation": 2.0,
    "fear": 1.2, "panic": 1.8, "crisis": 1.8, "insolvent": 2.5,
    "bankruptcy": 2.5, "insolvency": 2.5, "scam": 2.0, "rug pull": 2.5
}

COIN_KEYWORDS = {
    "btc": ["bitcoin", "btc", "satoshi", "halving", "spot btc"],
    "bnb": ["bnb", "binance", "cz", "bsc", "build and build", "binance coin"],
    "hype": ["hyperliquid", "hype", "purr", "hyperbft", "hl foundation"],
    "eth": ["ethereum", "ether", "eth", "vitalik", "erc-20", "erc20"],
    "sol": ["solana", "sol", "phantom wallet", "solana network"],
    "xrp": ["xrp", "ripple", "xrp ledger", "xrpl"],
    "doge": ["dogecoin", "doge", "doge army"],
    "near": ["near protocol", "near token", "$near"],
    "zec": ["zcash", "zec", "z-address", "shielded pool"],
}

MACRO_KEYWORDS = [
    "fed", "federal reserve", "powell", "interest rate", "cpi", "inflation",
    "sec", "treasury", "fomc", "jobs report", "tariff", "regulation"
]


class NewsEngine:
    """Singleton engine collecting and scoring live crypto news feeds."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NewsEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, update_interval: int = 60):
        if getattr(self, '_initialized', False):
            return
        self.update_interval = update_interval
        self.sources = [
            ("CoinTelegraph", "https://cointelegraph.com/rss"),
            ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("Decrypt", "https://decrypt.co/feed"),
        ]
        self.articles: List[Dict[str, Any]] = []
        self.last_updated: float = 0
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._data_lock = threading.Lock()
        self._initialized = True
        self.start()

    def start(self):
        """Start the background news polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self.running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="NewsEngineWorker")
        self._thread.start()
        logger.info("NewsEngine background worker started")

    def stop(self):
        self.running = False

    def _worker_loop(self):
        # Initial fetch immediately
        self.refresh()
        while self.running:
            time.sleep(self.update_interval)
            try:
                self.refresh()
            except Exception as e:
                logger.error(f"Error in news worker loop: {e}")

    def refresh(self):
        """Fetch and score latest articles from all feeds."""
        new_articles = []
        now = datetime.now(timezone.utc)

        for src_name, url in self.sources:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PredictionBot/2.0"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    xml_content = resp.read()
                    root = ET.fromstring(xml_content)
                    items = root.findall(".//item")
                    for item in items[:15]:
                        title_el = item.find("title")
                        desc_el = item.find("description")
                        pub_el = item.find("pubDate")
                        link_el = item.find("link")

                        title = (title_el.text or "").strip() if title_el is not None else ""
                        desc = (desc_el.text or "").strip() if desc_el is not None else ""
                        link = (link_el.text or "").strip() if link_el is not None else ""
                        pub_str = pub_el.text if pub_el is not None else None

                        if not title:
                            continue

                        # Clean HTML from description
                        desc_clean = re.sub(r'<[^>]+>', ' ', desc).strip()

                        # Parse timestamp
                        pub_dt = now
                        if pub_str:
                            try:
                                pub_dt = parsedate_to_datetime(pub_str)
                                if pub_dt.tzinfo is None:
                                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                            except Exception:
                                pub_dt = now

                        # Score and tag
                        score, bull_hits, bear_hits = self._score_text(title + " " + desc_clean)
                        coins = self._extract_coins(title + " " + desc_clean)
                        is_macro = any(m in (title + " " + desc_clean).lower() for m in MACRO_KEYWORDS)

                        age_seconds = (now - pub_dt).total_seconds()
                        if age_seconds < 0:
                            age_seconds = 0

                        new_articles.append({
                            "source": src_name,
                            "title": title,
                            "summary": desc_clean[:200] + ("..." if len(desc_clean) > 200 else ""),
                            "link": link,
                            "published_at": pub_dt.isoformat(),
                            "age_seconds": round(age_seconds),
                            "sentiment_score": round(score, 3),
                            "bullish_signals": bull_hits,
                            "bearish_signals": bear_hits,
                            "coins": coins,
                            "is_macro": is_macro
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch {src_name} ({url}): {e}")

        # Deduplicate by title similarity
        seen_titles = set()
        deduped = []
        for art in new_articles:
            # normalized key
            key = re.sub(r'[^a-zA-Z0-9]', '', art['title'].lower()[:40])
            if key not in seen_titles:
                seen_titles.add(key)
                deduped.append(art)

        # Sort by publication recency
        deduped.sort(key=lambda x: x['age_seconds'])

        with self._data_lock:
            self.articles = deduped[:50]
            self.last_updated = time.time()

        logger.info(f"NewsEngine refreshed: {len(self.articles)} unique articles indexed")

    def _score_text(self, text: str) -> (float, List[str], List[str]):
        """Compute compound sentiment score (-1.0 to 1.0) and matched keywords."""
        lower = text.lower()
        bull_score = 0.0
        bear_score = 0.0
        bull_hits = []
        bear_hits = []

        for kw, weight in BULLISH_KEYWORDS.items():
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, lower):
                bull_score += weight
                bull_hits.append(kw)

        for kw, weight in BEARISH_KEYWORDS.items():
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, lower):
                bear_score += weight
                bear_hits.append(kw)

        total = bull_score + bear_score
        if total == 0:
            return 0.0, [], []

        # Compound normalized score
        compound = (bull_score - bear_score) / (bull_score + bear_score + 1.0)
        compound = max(-1.0, min(1.0, compound))
        return compound, bull_hits, bear_hits

    def _extract_coins(self, text: str) -> List[str]:
        lower = text.lower()
        matched = []
        for coin, kws in COIN_KEYWORDS.items():
            for kw in kws:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                    matched.append(coin)
                    break
        return matched

    def get_summary(self, max_articles: int = 15) -> Dict[str, Any]:
        """Return formatted sentiment summary and recent news for dashboard and decision-making."""
        with self._data_lock:
            arts = list(self.articles)
            last_ts = self.last_updated

        # Calculate time-weighted sentiments (decay half-life: 45 minutes)
        # News from last 15 mins has highest weight for 5-17 min Kalshi rounds
        global_weight = 0.0
        global_weighted_score = 0.0

        coin_weights = {c: 0.0 for c in COIN_KEYWORDS}
        coin_scores = {c: 0.0 for c in COIN_KEYWORDS}

        recent_15m_count = 0
        recent_30m_count = 0

        for a in arts:
            age_min = a['age_seconds'] / 60.0
            if age_min <= 15:
                recent_15m_count += 1
            if age_min <= 30:
                recent_30m_count += 1

            # Exponential decay weight: w = exp(-age_min / 45)
            weight = max(0.05, 2.71828 ** (-age_min / 45.0))
            score = a['sentiment_score']

            global_weighted_score += score * weight
            global_weight += weight

            # If article tags specific coins, weight towards that coin
            coins = a['coins']
            if not coins and a['is_macro']:
                # Macro news affects all coins
                coins = list(COIN_KEYWORDS)

            for c in coins:
                if c in coin_scores:
                    coin_scores[c] += score * weight
                    coin_weights[c] += weight

        global_sentiment = (global_weighted_score / global_weight) if global_weight > 0 else 0.0
        coin_sentiment = {}
        for c in COIN_KEYWORDS:
            # Blend coin-specific sentiment with global sentiment (70% coin, 30% global)
            specific = (coin_scores[c] / coin_weights[c]) if coin_weights[c] > 0 else 0.0
            blended = 0.7 * specific + 0.3 * global_sentiment if coin_weights[c] > 0 else global_sentiment
            coin_sentiment[c] = round(max(-1.0, min(1.0, blended)), 3)

        return {
            "global_sentiment": round(max(-1.0, min(1.0, global_sentiment)), 3),
            "coin_sentiment": coin_sentiment,
            "recent_15m_news_count": recent_15m_count,
            "recent_30m_news_count": recent_30m_count,
            "total_articles": len(arts),
            "last_updated": datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat() if last_ts else None,
            "articles": arts[:max_articles]
        }

    def get_features(self, coin_id: str) -> Dict[str, float]:
        """Extract numeric features suitable for prediction models and decision rules."""
        summary = self.get_summary()
        cid = coin_id.lower()
        coin_score = summary["coin_sentiment"].get(cid, 0.0)
        global_score = summary["global_sentiment"]

        return {
            "news_coin_sentiment": coin_score,
            "news_global_sentiment": global_score,
            "news_15m_velocity": float(summary["recent_15m_news_count"]),
            "news_compound_signal": round(0.65 * coin_score + 0.35 * global_score, 4)
        }


# Global helper instance
_news_engine: Optional[NewsEngine] = None

def get_news_engine() -> NewsEngine:
    global _news_engine
    if _news_engine is None:
        _news_engine = NewsEngine()
    return _news_engine
