#!/usr/bin/env python3
"""Unified Flask API for all coins and horizons."""

import os

from flask import Flask, Response, jsonify, request
from datetime import datetime, timezone

from .coin_config import STATS_SINCE, get_coin_config, list_coins
from .data_store import DataStore
from .feature_engine import FeatureEngine
from .model_manager import ModelManager
from .predictor_core import Predictor
from .validator import Validator
from .kalshi_engine import KalshiRoundTracker
from .news_engine import get_news_engine
from . import dashboard
from .utils import setup_logging

logger = setup_logging("API")

_CACHE = {}


def _round_series_payload(ds, analysis, max_pts=220):
    """RTI ticks since the current round opened, for the price-vs-strike chart."""
    try:
        start_iso = analysis.get("timing", {}).get("current_round", {}).get("start_time")
        strike = analysis.get("pricing", {}).get("round_open_price")
        if not start_iso:
            return None
        rows = ds.get_prices(since=start_iso)
        t0 = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
        pts = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00")).timestamp()
                pts.append([round(ts - t0, 1), float(r["price"])])
            except Exception:
                continue
        if len(pts) > max_pts:
            step = len(pts) / max_pts
            pts = [pts[int(i * step)] for i in range(max_pts)]
        return {"strike": strike, "round_seconds": 900, "points": pts}
    except Exception:
        return None


def get_components(coin_id: str):
    """Return cached components for a coin."""
    key = coin_id.lower()
    if key not in _CACHE:
        cfg = get_coin_config(key)
        ds = DataStore(cfg.db_path, cfg.symbol)
        fe = FeatureEngine(ds, cfg.symbol)
        if getattr(cfg, 'hl_features', False):
            from .hl_features import attach_extras
            attach_extras(fe, cfg.db_name, micro=getattr(cfg, 'hl_micro', False))
        # Cross-asset lead-lag: BTC drives short-horizon moves in BNB/HYPE.
        if key != 'btc':
            btc_cfg = get_coin_config('btc')
            btc_ds = DataStore(btc_cfg.db_path, btc_cfg.symbol)
            btc_fe = FeatureEngine(btc_ds, btc_cfg.symbol)

            def _btc_bars_for(bars, _fe=btc_fe):
                import time as _t
                try:
                    age = abs(_t.time() - bars.index[-1].timestamp())
                except Exception:
                    age = 1e9
                if age < 900:
                    return _fe.build_bars(limit=12000)
                return _fe.build_bars(
                    since=bars.index[0].isoformat(),
                    until=bars.index[-1].isoformat())

            fe.cross_bars_fn = _btc_bars_for
        mm = ModelManager(cfg, ds, fe)
        pred = Predictor(cfg, ds, fe, mm)
        val = Validator(cfg, ds)
        kt = KalshiRoundTracker(ds, cfg.symbol)
        _CACHE[key] = (cfg, ds, fe, mm, pred, val, kt)
    return _CACHE[key]


def create_app() -> Flask:
    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

    @app.route('/predict/<coin_id>/<int:horizon>', methods=['GET'])
    def predict_one(coin_id, horizon):
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        if horizon not in cfg.prediction_horizons:
            return jsonify(error=f"Invalid horizon: {horizon}"), 400
        result = pred.predict(horizon)
        if result is None:
            return jsonify(error='Prediction failed'), 503
        return jsonify(result)

    @app.route('/predict/<coin_id>', methods=['GET'])
    def predict_all(coin_id):
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        results = []
        for h in cfg.prediction_horizons:
            r = pred.predict(h)
            if r:
                results.append(r)
        if not results:
            return jsonify(error='No predictions available'), 503
        return jsonify({'coin_id': coin_id, 'predictions': results})

    @app.route('/validate/<coin_id>', methods=['POST'])
    def validate_coin(coin_id):
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        n = val.validate()
        return jsonify({'coin_id': coin_id, 'validated': n})

    @app.route('/stats/<coin_id>', methods=['GET'])
    def stats(coin_id):
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        return jsonify({
            'coin_id': coin_id,
            'price_count': ds.count_prices(),
            'stats': {
                h: ds.get_prediction_stats(horizon=h, since=STATS_SINCE)
                for h in cfg.prediction_horizons
            }
        })

    INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cryptocurrency Prediction API | Unified Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f1115;
            --surface: #161922;
            --surface-2: #1e212b;
            --border: rgba(255, 255, 255, 0.08);
            --text: #f0f2f5;
            --text-muted: #8b92a8;
            --accent: #6366f1;
            --accent-2: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #06b6d4;
            --radius: 14px;
            --shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            min-height: 100vh;
        }
        .wrap { max-width: 1280px; margin: 0 auto; padding: 28px; }
        header {
            display: flex; align-items: center; justify-content: space-between;
            gap: 20px; flex-wrap: wrap; margin-bottom: 28px;
        }
        .brand { display: flex; align-items: center; gap: 14px; }
        .brand-icon {
            width: 48px; height: 48px; border-radius: var(--radius);
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            display: grid; place-items: center; font-size: 22px; box-shadow: var(--shadow);
        }
        .brand h1 { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
        .brand p { color: var(--text-muted); font-size: 0.9rem; margin-top: 2px; }
        .status-pill {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 16px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
            background: var(--surface); border: 1px solid var(--border);
        }
        .dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot.ok { background: var(--success); box-shadow: 0 0 10px var(--success); }
        .dot.warn { background: var(--warning); box-shadow: 0 0 10px var(--warning); }
        .dot.err { background: var(--danger); box-shadow: 0 0 10px var(--danger); }
        .refresh-btn {
            background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
            padding: 8px 16px; border-radius: 999px; cursor: pointer; font-weight: 500;
            transition: 0.2s; display: inline-flex; align-items: center; gap: 8px;
        }
        .refresh-btn:hover { background: var(--surface); border-color: var(--accent); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 28px; }
        .card {
            background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 22px; box-shadow: var(--shadow); transition: transform 0.2s, border-color 0.2s;
        }
        .card:hover { border-color: rgba(99, 102, 241, 0.35); }
        .card h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 10px; }
        .card .big { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; }
        .coin-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
        .coin-chip {
            background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px;
            padding: 6px 14px; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
        }
        .section-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 18px; display: flex; align-items: center; gap: 10px; }
        .endpoint-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
        .endpoint {
            background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
            overflow: hidden; display: flex; flex-direction: column;
        }
        .endpoint-head {
            padding: 16px 18px; border-bottom: 1px solid var(--border); display: flex;
            justify-content: space-between; align-items: center; gap: 12px;
        }
        .endpoint-meta { display: flex; align-items: center; gap: 10px; }
        .method {
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700;
            padding: 4px 8px; border-radius: 6px; color: #fff;
        }
        .method.get { background: var(--accent); }
        .method.post { background: var(--success); }
        .path { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text); font-weight: 500; }
        .endpoint-body { padding: 16px 18px; flex: 1; display: flex; flex-direction: column; gap: 14px; }
        .endpoint-desc { color: var(--text-muted); font-size: 0.9rem; }
        .try-btn {
            align-self: flex-start; background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
            padding: 8px 16px; border-radius: 999px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
            transition: 0.2s;
        }
        .try-btn:hover { background: var(--accent); border-color: var(--accent); }
        .response {
            display: none; background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
            padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; overflow-x: auto;
            white-space: pre-wrap; color: #c7d2fe;
        }
        .response.visible { display: block; }
        .prediction-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
        .prediction-card { background: var(--surface); border-radius: var(--radius); padding: 20px; border: 1px solid var(--border); }
        .prediction-card h4 { font-size: 1rem; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
        .price-line { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
        .price-label { color: var(--text-muted); font-size: 0.85rem; }
        .price-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.15rem; }
        .positive { color: var(--success); }
        .negative { color: var(--danger); }
        .muted { color: var(--text-muted); }
        .horizon-list { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; }
        .horizon-item {
            background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
            padding: 12px 14px; display: grid; grid-template-columns: 1fr auto auto; gap: 14px; align-items: center;
        }
        .horizon-item .time { font-size: 0.8rem; color: var(--text-muted); }
        .horizon-item .pred { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
        .horizon-item .change { font-size: 0.85rem; font-weight: 600; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .stats-table th, .stats-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
        .stats-table th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }
        .stats-table td { font-family: 'JetBrains Mono', monospace; }
        .bar-wrap { background: var(--surface-2); border-radius: 999px; height: 8px; overflow: hidden; margin-top: 6px; }
        .bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
        footer { text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 30px 0; }
        .empty { color: var(--text-muted); font-size: 0.9rem; padding: 12px 0; }
        @media (max-width: 640px) {
            .wrap { padding: 18px; }
            .brand h1 { font-size: 1.2rem; }
            .endpoint-grid, .prediction-grid { grid-template-columns: 1fr; }
            .horizon-item { grid-template-columns: 1fr; gap: 6px; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <header>
            <div class="brand">
                <div class="brand-icon">&#128200;</div>
                <div>
                    <h1>Prediction API Dashboard</h1>
                    <p>Unified multi-coin, multi-horizon cryptocurrency forecasting</p>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <span class="status-pill" id="healthPill"><span class="dot ok" id="healthDot"></span><span id="healthText">Loading...</span></span>
                <button class="refresh-btn" onclick="refreshAll()">&#x21bb; Refresh</button>
            </div>
        </header>

        <section class="grid">
            <div class="card">
                <h3>API Health</h3>
                <div class="big" id="healthBig">—</div>
                <p class="muted" id="healthSub">checking service status</p>
            </div>
            <div class="card">
                <h3>Supported Coins</h3>
                <div class="big" id="coinCount">—</div>
                <div class="coin-row" id="coinRow"></div>
            </div>
            <div class="card">
                <h3>Total Price Points</h3>
                <div class="big" id="priceCount">—</div>
                <p class="muted" id="priceSub">across all databases</p>
            </div>
            <div class="card">
                <h3>Last Update</h3>
                <div class="big" id="lastUpdate">—</div>
                <p class="muted">auto-refreshes every 15s</p>
            </div>
        </section>

        <section style="margin-bottom: 28px;">
            <h2 class="section-title">&#128161; Endpoint Explorer</h2>
            <div class="endpoint-grid" id="endpointGrid"></div>
        </section>

        <section style="margin-bottom: 28px;">
            <h2 class="section-title">&#128200; Live Predictions</h2>
            <div class="prediction-grid" id="predictionGrid"></div>
        </section>

        <section style="margin-bottom: 28px;">
            <h2 class="section-title">&#128202; Prediction Accuracy Stats</h2>
            <div class="prediction-grid" id="statsGrid"></div>
        </section>

        <footer>
            Cryptocurrency Prediction API &middot; BTC, BNB, HYPE &middot; 5 / 10 / 15 minute horizons
        </footer>
    </div>

    <script>
        const COINS = ['btc', 'eth', 'sol', 'bnb', 'xrp', 'doge', 'hype', 'near', 'zec'];

        function buildEndpoints() {
            const endpoints = [
                { method: 'GET', path: '/', desc: 'This dashboard page. Returns the API overview and documentation.' },
                { method: 'GET', path: '/health', desc: 'Service health check. Returns the current API status.' },
                { method: 'GET', path: '/coins', desc: 'List every supported coin identifier.' }
            ];
            for (const coin of COINS) {
                endpoints.push(
                    { method: 'GET', path: `/predict/${coin}`, desc: `All horizon predictions for ${coin.toUpperCase()} (5, 10, 15 min).` },
                    { method: 'GET', path: `/predict/${coin}/5`, desc: `Single 5-minute prediction for ${coin.toUpperCase()}.` },
                    { method: 'GET', path: `/predict/${coin}/10`, desc: `Single 10-minute prediction for ${coin.toUpperCase()}.` },
                    { method: 'GET', path: `/predict/${coin}/15`, desc: `Single 15-minute prediction for ${coin.toUpperCase()}.` },
                    { method: 'GET', path: `/stats/${coin}`, desc: `Accuracy and error statistics per horizon for ${coin.toUpperCase()}.` },
                    { method: 'POST', path: `/validate/${coin}`, desc: `Validate recent ${coin.toUpperCase()} predictions against live prices.` }
                );
            }
            return endpoints;
        }

        function renderEndpoints() {
            const endpoints = buildEndpoints();
            const grid = document.getElementById('endpointGrid');
            grid.innerHTML = endpoints.map((ep, i) => `
                <div class="endpoint">
                    <div class="endpoint-head">
                        <div class="endpoint-meta">
                            <span class="method ${ep.method.toLowerCase()}">${ep.method}</span>
                            <span class="path">${ep.path}</span>
                        </div>
                    </div>
                    <div class="endpoint-body">
                        <div class="endpoint-desc">${ep.desc}</div>
                        <button class="try-btn" onclick="tryEndpoint(${i}, '${ep.method}', '${ep.path}')">Try it</button>
                        <pre class="response" id="resp-${i}"></pre>
                    </div>
                </div>
            `).join('');
        }

        async function tryEndpoint(idx, method, path) {
            const panel = document.getElementById('resp-' + idx);
            panel.classList.add('visible');
            panel.textContent = 'Loading...';
            try {
                const res = await fetch(path, { method });
                const data = await res.json();
                panel.textContent = `HTTP ${res.status} ${res.statusText}\n${JSON.stringify(data, null, 2)}`;
            } catch (e) {
                panel.textContent = 'Error: ' + e.message;
            }
        }

        function setHealth(ok, text) {
            const pill = document.getElementById('healthPill');
            const dot = document.getElementById('healthDot');
            const txt = document.getElementById('healthText');
            const big = document.getElementById('healthBig');
            const sub = document.getElementById('healthSub');
            dot.className = 'dot ' + (ok ? 'ok' : 'err');
            txt.textContent = text;
            big.textContent = ok ? 'Healthy' : 'Degraded';
            sub.textContent = text;
        }

        async function fetchHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                setHealth(data.status === 'ok', data.status === 'ok' ? 'All systems operational' : 'Service issue detected');
            } catch (e) {
                setHealth(false, 'Unreachable');
            }
        }

        async function fetchCoins() {
            try {
                const data = await (await fetch('/coins')).json();
                const coins = data.coins || [];
                document.getElementById('coinCount').textContent = coins.length;
                document.getElementById('coinRow').innerHTML = coins.map(c => `<span class="coin-chip">${c}</span>`).join('');
            } catch (e) {
                document.getElementById('coinCount').textContent = '—';
            }
        }

        function fmtUsd(n) {
            if (n == null) return '—';
            return '$' + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }

        function fmtPct(n) {
            if (n == null) return '—';
            const v = Number(n);
            const sign = v > 0 ? '+' : '';
            return `${sign}${v.toFixed(4)}%`;
        }

        async function fetchPredictions() {
            const grid = document.getElementById('predictionGrid');
            grid.innerHTML = '';
            for (const coin of COINS) {
                const card = document.createElement('div');
                card.className = 'prediction-card';
                card.innerHTML = `<h4>&#11088; ${coin.toUpperCase()}</h4><div class="empty">Loading...</div>`;
                grid.appendChild(card);
                try {
                    const data = await (await fetch(`/predict/${coin}`)).json();
                    const preds = data.predictions || [];
                    if (!preds.length) {
                        card.querySelector('div').textContent = 'No predictions available';
                        continue;
                    }
                    const first = preds[0];
                    card.innerHTML = `
                        <h4>&#11088; ${coin.toUpperCase()}</h4>
                        <div class="price-line"><span class="price-label">Current price</span><span class="price-value">${fmtUsd(first.current_price)}</span></div>
                        <div class="price-line"><span class="price-label">Model</span><span class="price-value muted">${first.model_used || '—'}</span></div>
                        <div class="horizon-list">
                            ${preds.map(p => {
                                const cls = p.change_percent > 0 ? 'positive' : (p.change_percent < 0 ? 'negative' : 'muted');
                                const conf = Array.isArray(p.confidence_interval) ? `${fmtUsd(p.confidence_interval[0])} – ${fmtUsd(p.confidence_interval[1])}` : '—';
                                return `<div class="horizon-item">
                                    <div><div class="time">${p.horizon_minutes}-minute horizon</div><div class="pred">${fmtUsd(p.predicted_price)}</div></div>
                                    <div class="change ${cls}">${fmtPct(p.change_percent)}</div>
                                    <div class="muted" style="font-size:0.78rem;text-align:right;">${conf}</div>
                                </div>`;
                            }).join('')}
                        </div>
                    `;
                } catch (e) {
                    card.innerHTML = `<h4>&#11088; ${coin.toUpperCase()}</h4><div class="empty">Error loading predictions</div>`;
                }
            }
        }

        async function fetchStats() {
            const grid = document.getElementById('statsGrid');
            grid.innerHTML = '';
            let totalPrices = 0;
            for (const coin of COINS) {
                const card = document.createElement('div');
                card.className = 'prediction-card';
                card.innerHTML = `<h4>&#128293; ${coin.toUpperCase()} Stats</h4><div class="empty">Loading...</div>`;
                grid.appendChild(card);
                try {
                    const data = await (await fetch(`/stats/${coin}`)).json();
                    totalPrices += data.price_count || 0;
                    const stats = data.stats || {};
                    const rows = Object.entries(stats).map(([h, s]) => {
                        const acc = s.accuracy_pct_1pct || 0;
                        return `<tr>
                            <td>${h} min</td>
                            <td>${acc.toFixed(1)}%</td>
                            <td>${(s.avg_abs_error_pct || 0).toFixed(4)}%</td>
                            <td>${s.total || 0}</td>
                        </tr>`;
                    }).join('');
                    card.innerHTML = `
                        <h4>&#128293; ${coin.toUpperCase()} Stats</h4>
                        <table class="stats-table">
                            <thead><tr><th>Horizon</th><th>Accuracy</th><th>Avg Error</th><th>Total</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    `;
                } catch (e) {
                    card.innerHTML = `<h4>&#128293; ${coin.toUpperCase()} Stats</h4><div class="empty">Error loading stats</div>`;
                }
            }
            document.getElementById('priceCount').textContent = totalPrices.toLocaleString();
        }

        function updateTimestamp() {
            const now = new Date();
            document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
            document.getElementById('priceSub').textContent = 'across all databases · updated ' + now.toLocaleTimeString();
        }

        async function refreshAll() {
            await Promise.all([fetchHealth(), fetchCoins(), fetchPredictions(), fetchStats()]);
            updateTimestamp();
        }

        renderEndpoints();
        refreshAll();
        setInterval(refreshAll, 15000);
    </script>
</body>
</html>"""

    @app.route('/', methods=['GET'])
    def index():
        return Response(dashboard.DASHBOARD_HTML, mimetype='text/html')

    @app.route('/coins', methods=['GET'])
    def coins():
        return jsonify({'coins': list_coins()})

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})

    @app.route('/news', methods=['GET'])
    def news():
        """Real-time crypto news sentiment summary and headlines."""
        engine = get_news_engine()
        return jsonify(engine.get_summary())

    @app.route('/kalshi/<coin_id>', methods=['GET'])
    def kalshi_coin(coin_id):
        """Kalshi 15-minute round analysis, timing, open strike, delta, and trading decision."""
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        last = ds.last_price()
        if not last:
            return jsonify(error='No price data'), 503
        current_price, _ = last
        kt = rest[0] if rest else KalshiRoundTracker(ds, cfg.symbol)

        predictions_list = []
        predicted_returns = {}
        classifier_probs = {}
        for h in cfg.prediction_horizons:
            r = pred.predict(h, current_price=current_price)
            if r:
                predictions_list.append(r)
                predicted_returns[h] = r.get('change_percent', 0.0) / 100.0
                classifier_probs[h] = {
                    'up': r.get('prob_up', 0.33),
                    'down': r.get('prob_down', 0.33),
                    'flat': r.get('prob_flat', 0.33)
                }

        news_feats = get_news_engine().get_features(cfg.symbol)
        contract_prob = pred.get_contract_prob()
        analysis = kt.analyze_round(current_price, predicted_returns, classifier_probs,
                                    news_feats, contract_prob=contract_prob)
        analysis['predictions'] = predictions_list
        analysis['round_series'] = _round_series_payload(ds, analysis)
        return jsonify(analysis)

    @app.route('/kalshi', methods=['GET'])
    def kalshi_all():
        """Kalshi 15-minute round analysis for all supported coins."""
        results = {}
        for cid in list_coins():
            try:
                cfg, ds, fe, mm, pred, val, *rest = get_components(cid)
                last = ds.last_price()
                if not last:
                    continue
                current_price, _ = last
                kt = rest[0] if rest else KalshiRoundTracker(ds, cfg.symbol)
                predictions_list = []
                predicted_returns = {}
                classifier_probs = {}
                for h in cfg.prediction_horizons:
                    r = pred.predict(h, current_price=current_price)
                    if r:
                        predictions_list.append(r)
                        predicted_returns[h] = r.get('change_percent', 0.0) / 100.0
                        classifier_probs[h] = {
                            'up': r.get('prob_up', 0.33),
                            'down': r.get('prob_down', 0.33),
                            'flat': r.get('prob_flat', 0.33)
                        }
                news_feats = get_news_engine().get_features(cfg.symbol)
                contract_prob = pred.get_contract_prob()
                analysis = kt.analyze_round(current_price, predicted_returns, classifier_probs,
                                            news_feats, contract_prob=contract_prob)
                analysis['predictions'] = predictions_list
                analysis['round_series'] = _round_series_payload(ds, analysis)
                results[cid] = analysis
            except Exception as e:
                logger.error(f"Error in kalshi_all for {cid}: {e}")
        return jsonify(results)

    @app.route('/contract_stats/<coin_id>', methods=['GET'])
    def contract_stats(coin_id):
        """Live-scored Kalshi contract calls: win rate by phase + calibration."""
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        return jsonify({
            'coin_id': coin_id,
            'symbol': cfg.symbol,
            **ds.get_contract_signal_stats()
        })

    @app.route('/feed/<coin_id>', methods=['GET'])
    def feed(coin_id):
        """Unified trading-bot feed with current price, predictions and trend."""
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        last = ds.last_price()
        if not last:
            return jsonify(error='No price data'), 503
        current_price, price_ts = last
        by_horizon = {}
        for h in cfg.prediction_horizons:
            r = pred.predict(h)
            if r:
                by_horizon[h] = r
        if not by_horizon:
            return jsonify(error='No predictions available'), 503
        primary = cfg.prediction_horizons[0]
        primary_pred = by_horizon.get(primary, {})
        predicted_price = primary_pred.get('predicted_price', current_price)
        conf = primary_pred.get('confidence_interval', [current_price, current_price])
        change_5 = by_horizon.get(5, {}).get('change_percent')
        change_10 = by_horizon.get(10, {}).get('change_percent')
        change_15 = by_horizon.get(15, {}).get('change_percent')
        cp = change_5 if change_5 is not None else primary_pred.get('change_percent', 0)
        if cp > 0.05:
            trend = 'up'
        elif cp < -0.05:
            trend = 'down'
        else:
            trend = 'flat'
        accuracy = {}
        for h in cfg.prediction_horizons:
            stats = ds.get_prediction_stats(window=500, horizon=h, since=STATS_SINCE)
            accuracy[h] = {
                'accuracy_pct_1pct': stats.get('accuracy_pct_1pct', 0),
                'direction_accuracy_pct': stats.get('direction_accuracy_pct', 0),
                'beat_naive_pct': stats.get('beat_naive_pct', 0),
                'avg_abs_error_pct': stats.get('avg_abs_error_pct'),
                'total': stats.get('total', 0),
            }
        # Scale position sizing by how much better than a coin flip the
        # primary horizon has been recently (0.5 .. 1.5).
        prim_stats = accuracy.get(primary, {})
        dir_acc = prim_stats.get('direction_accuracy_pct') or 50.0
        if (prim_stats.get('total') or 0) < 200:
            conf_mult = 1.0
        else:
            conf_mult = round(min(1.5, max(0.5, 1.0 + (dir_acc - 50.0) / 20.0)), 3)
        signal_5 = by_horizon.get(5, {}).get('signal_strength')
        signal_10 = by_horizon.get(10, {}).get('signal_strength')
        signal_15 = by_horizon.get(15, {}).get('signal_strength')
        kt = rest[0] if rest else KalshiRoundTracker(ds, cfg.symbol)
        news_feats = get_news_engine().get_features(cfg.symbol)
        kalshi_round = kt.analyze_round(
            current_price=current_price,
            predicted_returns={h: by_horizon[h].get('change_percent', 0.0) / 100.0 for h in by_horizon},
            classifier_probs={h: {'up': by_horizon[h].get('prob_up', 0.33), 'down': by_horizon[h].get('prob_down', 0.33)} for h in by_horizon},
            news_features=news_feats,
            contract_prob=pred.get_contract_prob()
        )
        return jsonify({
            'symbol': cfg.symbol,
            'coin_id': coin_id,
            'price': current_price,
            'price_timestamp': price_ts,
            'predictions': list(by_horizon.values()),
            'predicted_price': round(predicted_price, 4) if predicted_price else None,
            'predicted_prices': {str(h): round(r.get('predicted_price'), 4)
                                 for h, r in by_horizon.items()},
            'prediction_id': primary_pred.get('prediction_id'),
            'prediction_ids': {str(h): r.get('prediction_id') for h, r in by_horizon.items()},
            'kalshi_round': kalshi_round,
            'news_sentiment': news_feats,
            'change_5': change_5,
            'change_10': change_10,
            'change_15': change_15,
            'raw_change_5': by_horizon.get(5, {}).get('raw_change_percent'),
            'raw_change_10': by_horizon.get(10, {}).get('raw_change_percent'),
            'raw_change_15': by_horizon.get(15, {}).get('raw_change_percent'),
            'signal_5': signal_5,
            'signal_10': signal_10,
            'signal_15': signal_15,
            'prob_up_5': by_horizon.get(5, {}).get('prob_up'),
            'prob_up_10': by_horizon.get(10, {}).get('prob_up'),
            'prob_up_15': by_horizon.get(15, {}).get('prob_up'),
            'prob_down_5': by_horizon.get(5, {}).get('prob_down'),
            'prob_down_10': by_horizon.get(10, {}).get('prob_down'),
            'prob_down_15': by_horizon.get(15, {}).get('prob_down'),
            'expected_pnl_5': by_horizon.get(5, {}).get('expected_pnl_pct'),
            'expected_pnl_10': by_horizon.get(10, {}).get('expected_pnl_pct'),
            'expected_pnl_15': by_horizon.get(15, {}).get('expected_pnl_pct'),
            'confidence': conf,
            'confidence_interval': conf,
            'model': primary_pred.get('model_used'),
            'model_version': primary_pred.get('model_version'),
            'price_trend': trend,
            'acc_5': accuracy.get(5, {}).get('direction_accuracy_pct'),
            'acc_10': accuracy.get(10, {}).get('direction_accuracy_pct'),
            'acc_15': accuracy.get(15, {}).get('direction_accuracy_pct'),
            'accuracy': accuracy,
            'confidence_multiplier': conf_mult,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    @app.route('/history/<coin_id>', methods=['GET'])
    def history(coin_id):
        """Export historical prices and predictions for backtesting / model training."""
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        limit = request.args.get('limit', 1000, type=int)
        since = request.args.get('since')
        prices = ds.get_prices(limit=limit, since=since)
        predictions = ds.get_recent_predictions(limit=min(limit, 500))
        return jsonify({
            'coin_id': coin_id,
            'symbol': cfg.symbol,
            'price_count': len(prices),
            'prediction_count': len(predictions),
            'prices': prices,
            'predictions': predictions
        })

    @app.route('/trades/<coin_id>', methods=['GET', 'POST'])
    def trades(coin_id):
        """Log or retrieve trades executed by the trading bot."""
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        if request.method == 'POST':
            data = request.get_json() or {}
            missing = [f for f in ['action', 'price'] if f not in data]
            if missing:
                return jsonify(error=f'Missing fields: {missing}'), 400
            trade_id = ds.log_trade(
                action=data['action'],
                price=float(data['price']),
                qty=data.get('qty'),
                amount_usd=data.get('amount_usd'),
                signal_source=data.get('signal_source', 'bot'),
                prediction_id=data.get('prediction_id'),
                pnl=data.get('pnl'),
                pnl_pct=data.get('pnl_pct'),
                metadata=data.get('metadata')
            )
            return jsonify({'coin_id': coin_id, 'trade_id': trade_id})
        limit = request.args.get('limit', 100, type=int)
        action = request.args.get('action')
        return jsonify({
            'coin_id': coin_id,
            'symbol': cfg.symbol,
            'trades': ds.get_recent_trades(limit=limit, action=action)
        })

    @app.route('/trades/<coin_id>/markout', methods=['GET'])
    def trade_markout(coin_id):
        """How did price move after each trade? Positive = favourable for the side taken."""
        try:
            cfg, ds, fe, mm, pred, val, *rest = get_components(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        limit = request.args.get('limit', 2000, type=int)
        horizons = [int(h) for h in request.args.get('horizons', '5,15,60').split(',')]
        rows = ds.trade_markouts(limit=limit, horizons_min=horizons)
        summary = {}
        for h in horizons:
            key = f'markout_{h}m_pct'
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                summary[str(h)] = {
                    'n': len(vals),
                    'avg_pct': round(sum(vals) / len(vals), 4),
                    'win_rate_pct': round(100 * sum(1 for v in vals if v > 0) / len(vals), 2),
                }
        by_action = {}
        for a in ('BUY', 'SELL'):
            sub = [r for r in rows if r['action'] == a]
            by_action[a] = {}
            for h in horizons:
                key = f'markout_{h}m_pct'
                vals = [r[key] for r in sub if r.get(key) is not None]
                if vals:
                    by_action[a][str(h)] = {
                        'n': len(vals),
                        'avg_pct': round(sum(vals) / len(vals), 4),
                        'win_rate_pct': round(100 * sum(1 for v in vals if v > 0) / len(vals), 2),
                    }
        return jsonify({
            'coin_id': coin_id,
            'symbol': cfg.symbol,
            'trade_count': len(rows),
            'summary': summary,
            'by_action': by_action,
            'trades': rows if request.args.get('detail') == '1' else None,
        })

    @app.route('/hl/<coin_id>', methods=['GET'])
    def hl_data(coin_id):
        """Hyperliquid market data: latest snapshot, candles, funding, freshness."""
        try:
            cfg = get_coin_config(coin_id)
        except Exception as e:
            return jsonify(error=str(e)), 400
        from .hyperliquid import HLStore, hl_db_path
        path = hl_db_path(cfg.db_name)
        if not os.path.exists(path):
            return jsonify(error=f'no Hyperliquid data for {coin_id}'), 404
        try:
            store = HLStore(path)
            status = store.status()
            latest = store.latest_snapshot()
            candles = store.candles_frame().tail(60)
            funding = store.funding_frame().tail(24)
            return jsonify({
                'coin_id': coin_id,
                'symbol': cfg.symbol,
                'latest_snapshot': latest,
                'candles_1m_tail': candles.reset_index().to_dict('records') if not candles.empty else [],
                'funding_tail': funding.reset_index().to_dict('records') if not funding.empty else [],
                'table_status': status,
            })
        except Exception as e:
            return jsonify(error=f'Hyperliquid data unavailable: {e}'), 503

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, threaded=True)
