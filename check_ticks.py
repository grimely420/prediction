import sqlite3
from datetime import datetime
base = '/home/chain-deaction/local_prediction/predition-app/prediction'
for coin, db in [('btc', 'bitcoin/bitcoin_prices.db'), ('bnb', 'bnb/bnb_prices.db'), ('hype', 'hype/hype_prices.db')]:
    conn = sqlite3.connect(f'{base}/{db}')
    rows = conn.execute("SELECT timestamp, price, source FROM prices ORDER BY id DESC LIMIT 12").fetchall()
    print(coin.upper())
    for ts, p, s in rows:
        print('   ', ts, p, s)
    # avg gap over last 500 rows
    ts_list = [r[0] for r in conn.execute("SELECT timestamp FROM prices ORDER BY id DESC LIMIT 500").fetchall()]
    ts_list = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in ts_list]
    if len(ts_list) > 2:
        gaps = [(ts_list[i-1] - ts_list[i]).total_seconds() for i in range(1, len(ts_list))]
        gaps = [g for g in gaps if 0 < g < 300]
        if gaps:
            print(f'   avg_gap={sum(gaps)/len(gaps):.1f}s  n={len(gaps)}')
    print()
