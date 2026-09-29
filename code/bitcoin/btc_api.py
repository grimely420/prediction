#!/usr/bin/env python3
"""
Enhanced Bitcoin API with ML predictions and improved validation.
"""

import sqlite3
import json
import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import threading
import time
import pytz

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bitcoin.config import DB_PATH, MODEL_DIR, THRESHOLD_FILE, SYMBOL, PREDICTION_HORIZON
from shared.predictor import get_predictor
from bitcoin import engine as btc_engine
from shared.utils import setup_logging

app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logging("BTC-API")

# Initialize ML predictor
btc_predictor = get_predictor(
    symbol=SYMBOL,
    model_dir=MODEL_DIR,
    feature_engineer=btc_engine,
    prediction_horizon=PREDICTION_HORIZON
)

# Set Eastern Time Zone
ET = pytz.timezone('US/Eastern')

def get_local_time():
    """Get current Eastern Time"""
    return datetime.now(ET)

def format_local_time(dt_str):
    """Convert UTC timestamp to Eastern Time"""
    try:
        # Parse UTC timestamp
        if dt_str.endswith('Z'):
            dt_str = dt_str.replace('Z', '+00:00')
        utc_dt = datetime.fromisoformat(dt_str)
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        # Convert to Eastern Time
        et_dt = utc_dt.astimezone(ET)
        return et_dt.strftime('%I:%M:%S %p ET').lstrip('0')
    except:
        return dt_str

def get_threshold():
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            return float(f.read().strip())
    except:
        return 1.0

def get_current_price():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT price, timestamp FROM prices ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            # Convert timestamp to Eastern Time for display
            local_ts = format_local_time(row[1])
            return row[0], local_ts
        return None, None
    except:
        return None, None

def get_price_for_validation():
    """Get raw UTC timestamp for validation"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT price, timestamp FROM prices ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return row if row else (None, None)
    except:
        return None, None

def log_prediction(predicted_price, model_used, confidence_interval=None):
    """Log prediction with enhanced metadata."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        pred_time = datetime.now(timezone.utc).isoformat()
        
        # Convert confidence interval to JSON if provided
        conf_json = json.dumps(confidence_interval) if confidence_interval else None
        
        c.execute('''INSERT INTO predictions 
                     (prediction_time, predicted_price, model_used, checked) 
                     VALUES (?, ?, ?, 0)''',
                  (pred_time, predicted_price, model_used))
        conn.commit()
        conn.close()
        
        logger.info(f"Logged prediction: ${predicted_price:.2f} using {model_used}")
    except Exception as e:
        logger.error(f"Log prediction error: {e}")

def validate_old_predictions():
    """Enhanced prediction validation with better error handling and logging."""
    while True:
        try:
            time.sleep(60)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            now = datetime.now(timezone.utc)
            cutoff = (now - timedelta(minutes=PREDICTION_HORIZON + 2)).isoformat()
            
            # Get predictions that need validation
            c.execute('''SELECT id, prediction_time, predicted_price, model_used 
                        FROM predictions 
                        WHERE (checked IS NULL OR checked=0) 
                        AND prediction_time <= ? 
                        ORDER BY prediction_time ASC''', (cutoff,))
            rows = c.fetchall()
            
            if rows:
                logger.info(f"Found {len(rows)} predictions to validate")
            
            threshold = get_threshold()
            validated_count = 0
            
            for pid, pt_str, pred_price, model_used in rows:
                try:
                    target_dt = datetime.fromisoformat(pt_str) + timedelta(minutes=PREDICTION_HORIZON)
                    target = target_dt.isoformat()
                    
                    # Get actual price at target time (allow 1 minute window)
                    c.execute('''SELECT price, timestamp 
                                FROM prices 
                                WHERE timestamp >= ? 
                                AND timestamp <= ?
                                ORDER BY timestamp ASC 
                                LIMIT 1''',
                              (target, (target_dt + timedelta(minutes=2)).isoformat()))
                    row = c.fetchone()
                    
                    if row:
                        actual, actual_timestamp = row
                        error = actual - pred_price
                        error_pct = (error / pred_price) * 100
                        is_correct = 1 if abs(error_pct) < threshold else 0
                        
                        c.execute('''UPDATE predictions 
                                    SET actual_price=?, error=?, checked=1, is_correct=? 
                                    WHERE id=?''',
                                  (actual, error, is_correct, pid))
                        conn.commit()
                        
                        validated_count += 1
                        logger.info(
                            f"Validated BTC pred {pid}: ${pred_price:.2f} → ${actual:.2f} "
                            f"(Error: {error_pct:+.2f}%, Correct: {is_correct}, Model: {model_used})"
                        )
                    else:
                        logger.warning(f"No actual price found for prediction {pid} at {target}")
                        
                except Exception as e:
                    logger.error(f"Error validating prediction {pid}: {e}")
            
            if validated_count > 0:
                logger.info(f"Validated {validated_count} predictions this cycle")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Validation cycle error: {e}")
            time.sleep(30)  # Wait before retry

threading.Thread(target=validate_old_predictions, daemon=True).start()

@app.route('/predict')
def predict():
    """Make prediction using ML model."""
    try:
        current, _ = get_price_for_validation()
        if not current:
            logger.warning("No price data available for prediction")
            return jsonify({'error': 'No price data'}), 400
        
        # Use ML predictor
        result = btc_predictor.predict(current)
        
        if result and result.get('success'):
            # Log the prediction
            log_prediction(
                result['predicted_price'], 
                result.get('model_used', 'unknown')
            )
            
            # Format response
            response = {
                'success': True,
                'current_price': round(result['current_price'], 2),
                f'predicted_price_{PREDICTION_HORIZON}min': round(result['predicted_price'], 2),
                'change_percent': round(result['change_percent'], 4),
                'model_used': result.get('model_used', 'unknown'),
                'prediction_horizon': PREDICTION_HORIZON,
                'confidence_interval': result.get('confidence_interval', {}),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Add fallback flag if applicable
            if result.get('fallback'):
                response['fallback_used'] = True
                logger.warning("Used fallback prediction for BTC")
            
            return jsonify(response)
        else:
            logger.error("ML prediction failed, using fallback")
            # Fallback to simple prediction
            predicted = current * 1.0005
            log_prediction(predicted, 'fallback')
            return jsonify({
                'success': True,
                'current_price': round(current, 2),
                f'predicted_price_{PREDICTION_HORIZON}min': round(predicted, 2),
                'change_percent': 0.05,
                'model_used': 'fallback',
                'fallback_used': True,
                'prediction_horizon': PREDICTION_HORIZON
            })
            
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Enhanced health check with comprehensive status."""
    try:
        # Check database connectivity
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get data stats
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM prices")
        count, min_ts, max_ts = cursor.fetchone()
        
        # Get prediction stats
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE checked=1 AND actual_price IS NOT NULL")
        validated_preds = cursor.fetchone()[0]
        
        # Get model info
        model_exists = os.path.exists(os.path.join(MODEL_DIR, "xgb_15min_latest.pkl"))
        
        conn.close()
        
        # Check if data is recent
        last_data_age = None
        if max_ts:
            try:
                last_ts = datetime.fromisoformat(max_ts)
                last_data_age = (datetime.now(timezone.utc) - last_ts).total_seconds()
            except:
                pass
        
        status = {
            'status': 'healthy',
            'service': 'Bitcoin',
            'version': '2.0.0',
            'data_points': count,
            'validated_predictions': validated_preds,
            'model_available': model_exists,
            'last_data_seconds_ago': round(last_data_age, 1) if last_data_age else None,
            'prediction_horizon': PREDICTION_HORIZON,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Determine health status
        if last_data_age and last_data_age > 120:
            status['status'] = 'degraded'
            status['message'] = 'Data collection may be delayed'
        
        if not model_exists:
            status['status'] = 'warning'
            status['message'] = 'ML model not yet trained'
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'service': 'Bitcoin',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/current_price')
def current_price():
    price, local_ts = get_current_price()
    if price:
        return jsonify({'price': price, 'timestamp': local_ts})
    return jsonify({'error': 'No data'}), 404

@app.route('/accuracy_summary')
def accuracy_summary():
    window = request.args.get('window', 100, type=int)
    threshold = get_threshold()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT AVG(ABS(error)/predicted_price*100), AVG(ABS(error)), COUNT(*),
                        SUM(CASE WHEN ABS(error)/predicted_price < ?/100 THEN 1 ELSE 0 END)
                 FROM (SELECT error, predicted_price FROM predictions WHERE checked=1 AND actual_price IS NOT NULL ORDER BY id DESC LIMIT ?)''', (threshold, window))
    row = c.fetchone()
    conn.close()
    avg_pct = round(row[0] or 0, 2)
    avg_usd = round(row[1] or 0, 2)
    cnt = row[2] or 0
    within = row[3] or 0
    acc = round(100*within/cnt, 1) if cnt else 0
    return jsonify({'last_n_predictions': cnt, 'avg_error_percent': avg_pct, 'avg_abs_error_usd': avg_usd,
                    'predictions_within_threshold': within, 'accuracy_percent': acc, 'threshold': threshold})

@app.route('/recent_predictions')
def recent_predictions():
    limit = request.args.get('limit', 20, type=int)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, prediction_time, predicted_price, actual_price, error, checked, is_correct, model_used,
                        ROUND(100.0*error/predicted_price,2) as error_pct
                 FROM predictions WHERE checked=1 ORDER BY id DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        # Convert prediction time to local time
        local_time = format_local_time(r[1])
        out.append({'id':r[0], 'prediction_time':local_time, 'predicted_price':r[2], 'actual_price':r[3],
                    'error':r[4], 'validated':bool(r[5]), 'is_correct':bool(r[6]) if r[6] is not None else None,
                    'model_used':r[7], 'error_percent':r[8]})
    return jsonify(out)

@app.route('/error_history')
def error_history():
    limit = request.args.get('limit', 50, type=int)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT prediction_time, 100.0 * error / predicted_price as error_percent 
                 FROM predictions WHERE checked=1 AND actual_price IS NOT NULL ORDER BY id DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    data = [{'time': format_local_time(r[0]), 'error_percent': round(r[1],2)} for r in reversed(rows)]
    return jsonify(data)

@app.route('/detailed_stats')
def detailed_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT model_used, COUNT(*), AVG(ABS(error)/predicted_price*100),
                        SUM(CASE WHEN ABS(error)/predicted_price*100 < 1 THEN 1 ELSE 0 END)
                 FROM predictions WHERE checked=1 AND error IS NOT NULL GROUP BY model_used''')
    rows = c.fetchall()
    conn.close()
    stats = {}
    for row in rows:
        stats[row[0]] = {'count': row[1], 'avg_error': round(row[2] or 0, 2), 'accuracy': round(100 * (row[3] or 0) / row[1], 2) if row[1] > 0 else 0}
    return jsonify(stats)

@app.route('/advanced_metrics')
def advanced_metrics():
    """Advanced validation metrics including directional accuracy and volatility-adjusted performance."""
    try:
        window = request.args.get('window', 100, type=int)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get recent predictions with actual prices
        c.execute('''SELECT predicted_price, actual_price, error, model_used
                    FROM predictions 
                    WHERE checked=1 AND actual_price IS NOT NULL 
                    ORDER BY id DESC LIMIT ?''', (window,))
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({'error': 'No validated predictions available'}), 404
        
        # Calculate advanced metrics
        predictions = []
        for pred, actual, error, model in rows:
            predictions.append({
                'predicted': pred,
                'actual': actual,
                'error': error,
                'error_pct': (error / pred) * 100,
                'model': model
            })
        
        # Directional accuracy (did we predict the right direction?)
        directional_correct = 0
        total = len(predictions)
        
        # Calculate various metrics
        total_abs_error = sum(abs(p['error']) for p in predictions)
        total_abs_error_pct = sum(abs(p['error_pct']) for p in predictions)
        max_error = max(abs(p['error']) for p in predictions)
        min_error = min(abs(p['error']) for p in predictions)
        
        # Error distribution
        errors_pct = [p['error_pct'] for p in predictions]
        mean_error_pct = sum(errors_pct) / total
        std_error_pct = (sum((x - mean_error_pct) ** 2 for x in errors_pct) / total) ** 0.5
        
        # Performance by model
        model_performance = {}
        for p in predictions:
            model = p['model']
            if model not in model_performance:
                model_performance[model] = {'count': 0, 'total_error': 0, 'errors': []}
            model_performance[model]['count'] += 1
            model_performance[model]['total_error'] += abs(p['error_pct'])
            model_performance[model]['errors'].append(abs(p['error_pct']))
        
        # Calculate per-model averages
        for model in model_performance:
            data = model_performance[model]
            data['avg_error_pct'] = data['total_error'] / data['count']
            data['std_error_pct'] = (sum((x - data['avg_error_pct']) ** 2 for x in data['errors']) / data['count']) ** 0.5
        
        metrics = {
            'total_predictions': total,
            'mean_abs_error_pct': round(total_abs_error_pct / total, 4),
            'std_error_pct': round(std_error_pct, 4),
            'max_abs_error_pct': round(max(abs(p['error_pct']) for p in predictions), 4),
            'min_abs_error_pct': round(min(abs(p['error_pct']) for p in predictions), 4),
            'mean_error_pct': round(mean_error_pct, 4),
            'model_performance': model_performance,
            'window': window,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Advanced metrics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_threshold', methods=['POST'])
def set_threshold_route():
    data = request.get_json()
    threshold = float(data.get('threshold', 1.0))
    with open(THRESHOLD_FILE, 'w') as f:
        f.write(str(threshold))
    return jsonify({'success': True})

@app.route('/retrain', methods=['POST'])
def force_retrain():
    return jsonify({'success': False, 'error': 'Auto-retrain active'})

@app.route('/')
@app.route('/dashboard')
def dashboard():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>₿ NEURAL PREDICTOR | CYBERPUNK EDITION</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --neon-orange: #ff6b35;
            --neon-gold: #ffd700;
            --neon-cyan: #00f3ff;
            --neon-pink: #ff00ff;
            --neon-green: #00ff88;
            --dark-bg: #0a0a0f;
            --glass-bg: rgba(10, 10, 15, 0.7);
        }
        
        body {
            background: var(--dark-bg);
            font-family: 'Inter', sans-serif;
            color: #fff;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 107, 53, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(0, 243, 255, 0.08) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
            animation: bgPulse 10s ease-in-out infinite;
        }
        
        @keyframes bgPulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        .glow-text {
            font-family: 'Orbitron', monospace;
            background: linear-gradient(135deg, var(--neon-orange), var(--neon-gold));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 30px rgba(255, 107, 53, 0.5);
            animation: textPulse 2s ease-in-out infinite;
        }
        
        @keyframes textPulse {
            0%, 100% { text-shadow: 0 0 20px rgba(255, 107, 53, 0.3); }
            50% { text-shadow: 0 0 40px rgba(255, 107, 53, 0.8); }
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 1.5rem; position: relative; z-index: 1; }
        
        .header {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 107, 53, 0.3);
            box-shadow: 0 0 30px rgba(255, 107, 53, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .logo i {
            font-size: 2.5rem;
            color: var(--neon-orange);
            filter: drop-shadow(0 0 10px var(--neon-orange));
            animation: iconPulse 1.5s ease-in-out infinite;
        }
        
        @keyframes iconPulse {
            0%, 100% { filter: drop-shadow(0 0 10px var(--neon-orange)); }
            50% { filter: drop-shadow(0 0 20px var(--neon-orange)); }
        }
        
        .logo h1 {
            font-family: 'Orbitron', monospace;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--neon-orange), var(--neon-gold));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: 2px;
        }
        
        .badge {
            background: linear-gradient(135deg, var(--neon-orange), var(--neon-pink));
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            font-size: 0.7rem;
            font-weight: 700;
            color: #fff;
            animation: badgePulse 1s ease-in-out infinite;
        }
        
        .live-price .label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            color: var(--neon-cyan);
            text-shadow: 0 0 5px var(--neon-cyan);
        }
        
        .live-price .price {
            font-family: 'Orbitron', monospace;
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--neon-gold);
            text-shadow: 0 0 20px var(--neon-gold);
        }
        
        .nav-buttons {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .nav-btn {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--neon-orange);
            padding: 0.7rem 1.5rem;
            border-radius: 50px;
            color: var(--neon-orange);
            cursor: pointer;
            font-family: 'Orbitron', monospace;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .nav-btn:hover {
            background: var(--neon-orange);
            color: #000;
            box-shadow: 0 0 20px var(--neon-orange);
            transform: translateY(-2px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 107, 53, 0.2);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            border-color: var(--neon-orange);
            box-shadow: 0 0 30px rgba(255, 107, 53, 0.2);
        }
        
        .stat-header {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--neon-cyan);
            margin-bottom: 0.5rem;
        }
        
        .stat-value {
            font-family: 'Orbitron', monospace;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff, var(--neon-gold));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        
        .chart-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 107, 53, 0.2);
        }
        
        .chart-title {
            font-family: 'Orbitron', monospace;
            color: var(--neon-cyan);
            margin-bottom: 1rem;
        }
        
        canvas { max-height: 300px; }
        
        .control-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 107, 53, 0.2);
        }
        
        button {
            background: linear-gradient(135deg, var(--neon-orange), var(--neon-pink));
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            color: #fff;
            font-family: 'Orbitron', monospace;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 20px var(--neon-orange);
        }
        
        .prediction-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .prediction-table th, .prediction-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255, 107, 53, 0.2);
        }
        
        .prediction-table th {
            color: var(--neon-cyan);
            font-family: 'Orbitron', monospace;
        }
        
        .correct-badge {
            background: rgba(0, 255, 136, 0.2);
            color: var(--neon-green);
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            border: 1px solid var(--neon-green);
        }
        
        .incorrect-badge {
            background: rgba(255, 0, 255, 0.2);
            color: var(--neon-pink);
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            border: 1px solid var(--neon-pink);
        }
        
        .pending-badge {
            background: rgba(0, 243, 255, 0.2);
            color: var(--neon-cyan);
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            border: 1px solid var(--neon-cyan);
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            font-size: 0.7rem;
            color: rgba(255,255,255,0.3);
            font-family: 'Orbitron', monospace;
        }
        
        .time-badge {
            background: rgba(0, 243, 255, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            font-size: 0.6rem;
            color: var(--neon-cyan);
            font-family: 'Orbitron', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <i class="fab fa-bitcoin"></i>
                <h1>NEURAL PREDICTOR <span class="badge">CYBERPUNK v3.0</span></h1>
            </div>
            <div class="live-price">
                <div class="label">LIVE MARKET DATA <span class="time-badge">ET</span></div>
                <div class="price" id="currentPrice">—</div>
                <div class="label" id="priceUpdate" style="font-size:0.6rem;"></div>
            </div>
        </div>
        
        <div class="nav-buttons">
            <button class="nav-btn" onclick="showSection('dashboard')"><i class="fas fa-chart-line"></i> DASHBOARD</button>
            <button class="nav-btn" onclick="showSection('predictions')"><i class="fas fa-table-list"></i> LEDGER</button>
            <button class="nav-btn" onclick="showSection('stats')"><i class="fas fa-chart-simple"></i> STATS</button>
        </div>
        
        <div id="dashboard-section">
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-header">ACCURACY (≤1%)</div><div class="stat-value" id="accuracyValue">—%</div></div>
                <div class="stat-card"><div class="stat-header">AVG ERROR</div><div class="stat-value" id="avgErrorValue">—%</div></div>
                <div class="stat-card"><div class="stat-header">PREDICTIONS</div><div class="stat-value" id="predCount">—</div></div>
                <div class="stat-card"><div class="stat-header">AI STATUS</div><div class="stat-value" id="modelStatus">ONLINE</div></div>
            </div>
            
            <div class="chart-card">
                <div class="chart-title"><i class="fas fa-chart-line"></i> PREDICTION vs REALITY</div>
                <canvas id="predictionChart"></canvas>
            </div>
            
            <div class="chart-card">
                <div class="chart-title"><i class="fas fa-chart-column"></i> ERROR DISTRIBUTION</div>
                <canvas id="errorChart"></canvas>
            </div>
            
            <div class="control-panel">
                <button id="retrainBtn"><i class="fas fa-sync-alt"></i> RETRAIN AI</button>
                <span id="retrainMsg" style="margin-left:1rem; color:var(--neon-cyan)"></span>
            </div>
        </div>
        
        <div id="predictions-section" style="display:none;">
            <div class="chart-card">
                <div class="chart-title"><i class="fas fa-table-list"></i> PREDICTION LEDGER <span class="time-badge">EASTERN TIME</span></div>
                <div style="overflow-x:auto;">
                    <table class="prediction-table">
                        <thead><tr><th>TIME (ET)</th><th>PREDICTED</th><th>ACTUAL</th><th>ERROR %</th><th>MODEL</th><th>STATUS</th></tr></thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div id="stats-section" style="display:none;">
            <div class="chart-card">
                <div class="chart-title"><i class="fas fa-chart-simple"></i> MODEL PERFORMANCE</div>
                <div id="detailedStats"></div>
            </div>
        </div>
        
        <div class="footer">
            <i class="fas fa-microchip"></i> XGBOOST NEURAL ENSEMBLE • 15-MINUTE HORIZON • ALL TIMES EASTERN (ET)
        </div>
    </div>
    
    <script>
        let predictionChart, errorChart;
        
        function showSection(section) {
            document.getElementById('dashboard-section').style.display = section === 'dashboard' ? 'block' : 'none';
            document.getElementById('predictions-section').style.display = section === 'predictions' ? 'block' : 'none';
            document.getElementById('stats-section').style.display = section === 'stats' ? 'block' : 'none';
        }
        
        async function fetchJSON(url) {
            const res = await fetch(url);
            return res.json();
        }
        
        async function updatePrice() {
            const data = await fetchJSON('/current_price');
            document.getElementById('currentPrice').innerHTML = `$${data.price.toLocaleString()}`;
            document.getElementById('priceUpdate').innerHTML = data.timestamp;
        }
        
        async function updateStats() {
            const data = await fetchJSON('/accuracy_summary');
            document.getElementById('accuracyValue').innerHTML = `${data.accuracy_percent}%`;
            document.getElementById('avgErrorValue').innerHTML = `${data.avg_error_percent}%`;
            document.getElementById('predCount').innerHTML = data.last_n_predictions;
        }
        
        async function updateTable() {
            const predictions = await fetchJSON('/recent_predictions');
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            predictions.forEach(p => {
                const row = tbody.insertRow();
                row.insertCell(0).innerHTML = p.prediction_time;
                row.insertCell(1).innerHTML = `$${p.predicted_price.toLocaleString()}`;
                row.insertCell(2).innerHTML = p.actual_price ? `$${p.actual_price.toLocaleString()}` : '—';
                row.insertCell(3).innerHTML = p.error_percent ? `${p.error_percent}%` : '—';
                row.insertCell(4).innerHTML = `<span style="color:#ff6b35">${p.model_used || 'neural'}</span>`;
                let status = p.is_correct === true ? '<span class="correct-badge">✓ ACCURATE</span>' : 
                            (p.is_correct === false ? '<span class="incorrect-badge">✗ DEVIATED</span>' : 
                            '<span class="pending-badge">⌛ VALIDATING</span>');
                row.insertCell(5).innerHTML = status;
            });
        }
        
        async function updateCharts() {
            const predictions = await fetchJSON('/recent_predictions?limit=20');
            const validated = predictions.filter(p => p.validated).slice(0,10).reverse();
            const labels = validated.map(p => p.prediction_time);
            const predicted = validated.map(p => p.predicted_price);
            const actual = validated.map(p => p.actual_price);
            
            if (predictionChart) predictionChart.destroy();
            predictionChart = new Chart(document.getElementById('predictionChart'), {
                type: 'line',
                data: { labels, datasets: [
                    { label: 'NEURAL PREDICTION', data: predicted, borderColor: '#ff6b35', borderWidth: 3, tension: 0.3, pointRadius: 5, pointBackgroundColor: '#ff6b35' },
                    { label: 'ACTUAL PRICE', data: actual, borderColor: '#00ff88', borderWidth: 3, tension: 0.3, pointRadius: 5, pointBackgroundColor: '#00ff88' }
                ]}
            });
            
            const errors = await fetchJSON('/error_history');
            const errorLabels = errors.map(e => e.time);
            const errorVals = errors.map(e => e.error_percent);
            if (errorChart) errorChart.destroy();
            errorChart = new Chart(document.getElementById('errorChart'), {
                type: 'bar',
                data: { labels: errorLabels, datasets: [{ label: 'ERROR %', data: errorVals, backgroundColor: '#ff6b35', borderRadius: 8 }] }
            });
        }
        
        async function updateDetailedStats() {
            const stats = await fetchJSON('/detailed_stats');
            let html = '<div class="stats-grid">';
            for (const [model, data] of Object.entries(stats)) {
                html += `<div class="stat-card"><div class="stat-header">${model.toUpperCase()} AI</div>
                        <div class="stat-value">${data.accuracy}%</div>
                        <div>PREDICTIONS: ${data.count}</div>
                        <div>ERROR: ${data.avg_error}%</div></div>`;
            }
            html += '</div>';
            document.getElementById('detailedStats').innerHTML = html;
        }
        
        async function refreshAll() {
            await Promise.all([updatePrice(), updateStats(), updateTable(), updateCharts(), updateDetailedStats()]);
        }
        
        document.getElementById('retrainBtn').addEventListener('click', async () => {
            const msg = document.getElementById('retrainMsg');
            msg.innerHTML = '<i class="fas fa-spinner fa-spin"></i> RETRAINING...';
            const res = await fetch('/retrain', { method: 'POST' });
            const data = await res.json();
            msg.innerHTML = data.success ? '✅ AI RETRAINED' : '⚠️ AUTO-RETRAIN ACTIVE';
            setTimeout(() => msg.innerHTML = '', 3000);
        });
        
        refreshAll();
        setInterval(refreshAll, 10000);
        setInterval(updatePrice, 5000);
    </script>
</body>
</html>
    '''

if __name__ == '__main__':
    print("🚀 Futuristic Bitcoin API starting on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
