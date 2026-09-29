#!/usr/bin/env python3
"""
Enhanced HYPE API with ML predictions and improved validation.
"""

import sqlite3
import json
import sys
import os
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import threading
import time
import pytz

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from hype.config import DB_PATH, MODEL_DIR, THRESHOLD_FILE, SYMBOL, PREDICTION_HORIZON
from shared.predictor import get_predictor
from hype import engine as hype_engine
from shared.dashboard import coin_dashboard_html
from shared.utils import setup_logging

app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logging("HYPE-API")

# Initialize ML predictor
hype_predictor = get_predictor(
    symbol=SYMBOL,
    model_dir=MODEL_DIR,
    feature_engineer=hype_engine,
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
        if dt_str.endswith('Z'):
            dt_str = dt_str.replace('Z', '+00:00')
        utc_dt = datetime.fromisoformat(dt_str)
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        et_dt = utc_dt.astimezone(ET)
        return et_dt.strftime('%I:%M:%S %p ET').lstrip('0')
    except Exception:
        return dt_str

def get_threshold():
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            return float(f.read().strip())
    except Exception:
        return 2.0  # HYPE uses wider threshold due to higher volatility

def get_current_price():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT price, timestamp FROM prices ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            local_ts = format_local_time(row[1])
            return row[0], local_ts
        return None, None
    except Exception:
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
    except Exception:
        return None, None

def log_prediction(predicted_price, model_used, confidence_interval=None):
    """Log prediction with enhanced metadata."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        pred_time = datetime.now(timezone.utc).isoformat()
        
        c.execute('''INSERT INTO predictions 
                     (prediction_time, predicted_price, model_used, checked) 
                     VALUES (?, ?, ?, 0)''',
                  (pred_time, predicted_price, model_used))
        conn.commit()
        conn.close()
        
        logger.info(f"Logged prediction: ${predicted_price:.4f} using {model_used}")
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
                            f"Validated HYPE pred {pid}: ${pred_price:.4f} -> ${actual:.4f} "
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
            time.sleep(30)

threading.Thread(target=validate_old_predictions, daemon=True).start()

@app.route('/predict')
def predict():
    """Make prediction using ML model."""
    try:
        current, _ = get_price_for_validation()
        if not current:
            logger.warning("No price data available for prediction")
            return jsonify({'error': 'No price data'}), 400
        
        result = hype_predictor.predict(current)
        
        if result and result.get('success'):
            log_prediction(
                result['predicted_price'], 
                result.get('model_used', 'unknown')
            )
            
            response = {
                'success': True,
                'current_price': round(result['current_price'], 4),
                f'predicted_price_{PREDICTION_HORIZON}min': round(result['predicted_price'], 4),
                'change_percent': round(result['change_percent'], 4),
                'model_used': result.get('model_used', 'unknown'),
                'prediction_horizon': PREDICTION_HORIZON,
                'confidence_interval': result.get('confidence_interval', {}),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if result.get('fallback'):
                response['fallback_used'] = True
                logger.warning("Used fallback prediction for HYPE")
            
            return jsonify(response)
        else:
            logger.error("ML prediction failed, using fallback")
            predicted = current * 1.0005
            log_prediction(predicted, 'fallback')
            return jsonify({
                'success': True,
                'current_price': round(current, 4),
                f'predicted_price_{PREDICTION_HORIZON}min': round(predicted, 4),
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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM prices")
        count, min_ts, max_ts = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE checked=1 AND actual_price IS NOT NULL")
        validated_preds = cursor.fetchone()[0]
        
        model_exists = os.path.exists(os.path.join(MODEL_DIR, f"xgb_{PREDICTION_HORIZON}min_latest.pkl"))
        
        conn.close()
        
        last_data_age = None
        if max_ts:
            try:
                last_ts = datetime.fromisoformat(max_ts)
                last_data_age = (datetime.now(timezone.utc) - last_ts).total_seconds()
            except Exception:
                pass
        
        status = {
            'status': 'healthy',
            'service': 'HYPE',
            'version': '2.0.0',
            'data_points': count,
            'validated_predictions': validated_preds,
            'model_available': model_exists,
            'last_data_seconds_ago': round(last_data_age, 1) if last_data_age else None,
            'prediction_horizon': PREDICTION_HORIZON,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
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
            'service': 'HYPE',
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
    avg_usd = round(row[1] or 0, 4)
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
        local_time = format_local_time(r[1])
        out.append({'id':r[0], 'prediction_time':local_time, 'predicted_price':r[2], 'actual_price':r[3],
                    'error':r[4], 'validated':bool(r[5]), 'is_correct':bool(r[6]) if r[6] is not None else None,
                    'model_used':r[7], 'error_percent':r[8]})
    return jsonify({'predictions': out})

@app.route('/data_info')
def data_info():
    """Get data collection information."""
    info = hype_engine.get_data_info()
    return jsonify(info)
@app.route('/')
@app.route('/dashboard')
def dashboard():
    endpoints = [
        { 'method': 'GET', 'path': '/', 'desc': 'This dashboard page. Returns the API overview.' },
        { 'method': 'GET', 'path': '/health', 'desc': 'Service health check with data and model status.' },
        { 'method': 'GET', 'path': '/predict', 'desc': 'Make a single-horizon ML prediction.' },
        { 'method': 'GET', 'path': '/current_price', 'desc': 'Latest collected price and timestamp.' },
        { 'method': 'GET', 'path': '/accuracy_summary', 'desc': 'Accuracy and error summary over recent predictions.' },
        { 'method': 'GET', 'path': '/recent_predictions', 'desc': 'List of recent validated predictions.' },
        { 'method': 'GET', 'path': '/data_info', 'desc': 'Data collection and training readiness information.' },
    ]
    html = coin_dashboard_html('hype', 'HYPE', endpoints, coin_emoji='&#128640;', stats_endpoint='/data_info', stats_title='Data Info')
    return Response(html, mimetype='text/html')


if __name__ == '__main__':
    from hype.config import API_HOST, API_PORT
    logger.info(f"Starting {SYMBOL} API on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False)
