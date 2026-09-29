import os, sqlite3, json, glob
base = '/home/chain-deaction/local_prediction/predition-app/prediction'

for coin, db in [('btc', 'bitcoin/bitcoin_prices.db'), ('bnb', 'bnb/bnb_prices.db'), ('hype', 'hype/hype_prices.db')]:
    p = f'{base}/{db}'
    size_mb = os.path.getsize(p) / 1e6
    conn = sqlite3.connect(p)
    n, first, last = conn.execute('SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM prices').fetchone()
    span_h = None
    try:
        from datetime import datetime
        t0 = datetime.fromisoformat(first.replace('Z', '+00:00'))
        t1 = datetime.fromisoformat(last.replace('Z', '+00:00'))
        span_h = (t1 - t0).total_seconds() / 3600
    except Exception:
        pass
    try:
        sig = conn.execute('SELECT COUNT(*) FROM contract_signals').fetchone()[0]
    except Exception:
        sig = 0
    print(f'{coin.upper()}: {n:,} ticks | {size_mb:.0f} MB | span {span_h/24:.1f} days | contract_signals {sig}')

print()
print('=== LAST RETRAINS (from predictor logs) ===')
for coin in ['btc', 'bnb', 'hype']:
    log = f'{base}/logs/{coin}-predictor.log'
    try:
        lines = open(log, errors='replace').read().strip().splitlines()
        trains = [l for l in lines if 'Trained' in l or 'Retrain' in l]
        print(f'{coin.upper()}:')
        for l in trains[-4:]:
            print('   ', l[:160])
    except Exception as e:
        print(coin, 'log err', e)
