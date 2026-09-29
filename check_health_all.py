import sqlite3, json, os, time, urllib.request
from datetime import datetime

base = '/home/chain-deaction/local_prediction/predition-app/prediction'
coins = [('btc', 'bitcoin'), ('eth', 'eth'), ('sol', 'sol'), ('bnb', 'bnb'),
         ('xrp', 'xrp'), ('doge', 'doge'), ('hype', 'hype'), ('near', 'near'), ('zec', 'zec')]
now = time.time()

print('=== DATA COLLECTION ===')
for coin, db in coins:
    try:
        p = f'{base}/{db}/{db}_prices.db'
        conn = sqlite3.connect(p)
        n, first, last = conn.execute(
            'SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM prices').fetchone()
        t_last = datetime.fromisoformat(last.replace('Z', '+00:00'))
        age_s = (datetime.now(t_last.tzinfo) - t_last).total_seconds()
        t_first = datetime.fromisoformat(first.replace('Z', '+00:00'))
        span_h = (t_last - t_first).total_seconds() / 3600
        # cadence over last 200
        ts = [r[0] for r in conn.execute('SELECT timestamp FROM prices ORDER BY id DESC LIMIT 200')]
        ts = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in ts]
        gaps = [(ts[i-1]-ts[i]).total_seconds() for i in range(1, len(ts)) if 0 < (ts[i-1]-ts[i]).total_seconds() < 60]
        gap = sum(gaps)/len(gaps) if gaps else -1
        srcs = conn.execute('SELECT source, COUNT(*) c FROM (SELECT source FROM prices ORDER BY id DESC LIMIT 200) GROUP BY source ORDER BY c DESC').fetchall()
        try:
            sigs = conn.execute('SELECT COUNT(*) FROM contract_signals').fetchone()[0]
            resolved = conn.execute('SELECT COUNT(*) FROM contract_signals WHERE outcome IS NOT NULL').fetchone()[0]
        except Exception:
            sigs, resolved = 0, 0
        print(f'{coin.upper():5s} | {n:>8,} ticks | span {span_h:6.1f}h | gap {gap:4.1f}s | last {age_s:4.0f}s ago | src {srcs[0][0]} | sigs {sigs} ({resolved} resolved)')
    except Exception as e:
        print(f'{coin.upper()}: DB ERR {e}')

print()
print('=== MODELS (age in min, - = missing) ===')
for coin, db in coins:
    mdir = f'{base}/{db}/models'
    row = f'{coin.upper():5s} |'
    for f, tag in [('xgb_5min_latest.pkl', 'reg5'), ('xgb_5min_clf.pkl', 'clf5'), ('kalshi_contract_latest.pkl', 'contract')]:
        p = f'{mdir}/{f}'
        if os.path.exists(p):
            row += f' {tag}={((now - os.path.getmtime(p))/60):5.0f}m |'
        else:
            row += f' {tag}=NONE  |'
    print(row)

print()
print('=== LIVE /kalshi ===')
try:
    d = json.load(urllib.request.urlopen('http://localhost:5000/kalshi', timeout=90))
    for c in ['btc', 'eth', 'sol', 'bnb', 'xrp', 'doge', 'hype', 'near', 'zec']:
        v = d.get(c)
        if not v:
            print(f'{c.upper()}: MISSING')
            continue
        dec = v.get('decision', {})
        cm = v.get('contract_model') or {}
        pr = v.get('pricing', {})
        rs = v.get('round_series') or {}
        print(f"{c.upper():5s} | {dec.get('recommendation','n/a'):36s} | contract={cm.get('p_up')} | strike={pr.get('round_open_price')} cur={pr.get('current_price')} | pts={len(rs.get('points', []))}")
except Exception as e:
    print('kalshi err:', e)
