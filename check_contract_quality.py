import sqlite3, joblib, json, sys, os
sys.path.insert(0, '/home/chain-deaction/local_prediction/predition-app/prediction')
os.chdir('/home/chain-deaction/local_prediction/predition-app/prediction')

base = '/home/chain-deaction/local_prediction/predition-app/prediction'

print('=== CONTRACT MODEL METRICS ===')
for coin, db in [('eth','eth'),('sol','sol'),('xrp','xrp'),('doge','doge'),('near','near'),('zec','zec'),('btc','bitcoin')]:
    try:
        p = f'{base}/{db}/models/kalshi_contract_latest.pkl'
        m = joblib.load(p)
        met = m[3] if len(m) > 3 else {}
        print(f"{coin.upper():5s} acc={met.get('val_acc')} auc={met.get('val_auc')} brier={met.get('val_brier')} n={met.get('n_samples')} keys={list(met.keys())[:8]}")
    except Exception as e:
        print(f'{coin}: {type(e).__name__} {e}')

print()
print('=== SIGNAL SCHEMA ===')
conn = sqlite3.connect(f'{base}/eth/eth_prices.db')
cols = conn.execute('PRAGMA table_info(contract_signals)').fetchall()
print([c[1] for c in cols])

print()
print('=== SIGNAL OUTCOMES (new coins) ===')
for coin in ['eth','sol','xrp','doge','near','zec']:
    try:
        conn = sqlite3.connect(f'{base}/{coin}/{coin}_prices.db')
        rows = conn.execute('SELECT * FROM contract_signals ORDER BY id DESC LIMIT 10').fetchall()
        cols = [c[1] for c in conn.execute('PRAGMA table_info(contract_signals)')]
        oi = cols.index('outcome') if 'outcome' in cols else None
        for r in rows[:6]:
            print('  ', coin, dict(zip(cols, r)))
    except Exception as e:
        print(f'{coin}: {e}')
    print()
