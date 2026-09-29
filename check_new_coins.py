import sqlite3, json, urllib.request

base = '/home/chain-deaction/local_prediction/predition-app/prediction'
for coin in ['eth', 'sol', 'xrp', 'doge', 'near', 'zec']:
    try:
        conn = sqlite3.connect(f'{base}/{coin}/{coin}_prices.db')
        n, latest, src = conn.execute(
            'SELECT COUNT(*), MAX(timestamp), (SELECT source FROM prices ORDER BY id DESC LIMIT 1) FROM prices').fetchone()
        px = conn.execute('SELECT price FROM prices ORDER BY id DESC LIMIT 1').fetchone()
        print(f'{coin.upper():5s}: {n} ticks | last {latest} | {src} | px={px[0] if px else None}')
    except Exception as e:
        print(f'{coin.upper()}: ERR {e}')

print()
try:
    d = json.load(urllib.request.urlopen('http://localhost:5000/kalshi', timeout=60))
    for c, v in d.items():
        dec = v.get('decision', {})
        print(f"{c.upper():5s} {dec.get('recommendation', 'n/a'):34s} YES {dec.get('yes_cents')}c")
except Exception as e:
    print('kalshi err:', e)
