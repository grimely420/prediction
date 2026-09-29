import json, sqlite3, urllib.request
from datetime import datetime

d = json.load(urllib.request.urlopen('http://localhost:5000/kalshi/btc', timeout=30))
rs = d.get('round_series')
if rs:
    print('series strike:', rs['strike'], '| pts:', len(rs['points']))
    print('first:', rs['points'][:3])
    print('last :', rs['points'][-3:])
else:
    print('round_series MISSING')

base = '/home/chain-deaction/local_prediction/predition-app/prediction'
for coin, db in [('btc', 'bitcoin/bitcoin_prices.db'), ('bnb', 'bnb/bnb_prices.db'), ('hype', 'hype/hype_prices.db')]:
    conn = sqlite3.connect(f'{base}/{db}')
    ts = [r[0] for r in conn.execute('SELECT timestamp FROM prices ORDER BY id DESC LIMIT 60')]
    ts = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in ts]
    gaps = [(ts[i-1]-ts[i]).total_seconds() for i in range(1, len(ts)) if 0 < (ts[i-1]-ts[i]).total_seconds() < 60]
    if gaps:
        print(f'{coin}: new avg gap {sum(gaps)/len(gaps):.2f}s over {len(gaps)} gaps')
