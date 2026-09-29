import json, os, time, urllib.request
from datetime import datetime, timezone

def get(path):
    return json.load(urllib.request.urlopen(f'http://localhost:5000{path}', timeout=30))

base = '/home/chain-deaction/local_prediction/predition-app/prediction'
now = time.time()

print('=== CONTRACT MODEL FILES (age) ===')
for coin, d in [('btc', 'bitcoin'), ('bnb', 'bnb'), ('hype', 'hype')]:
    mdir = f'{base}/{d}/models'
    for f in sorted(os.listdir(mdir)):
        if 'contract' in f:
            age = (now - os.path.getmtime(f'{mdir}/{f}')) / 60
            print(f'  {coin}: {f}  ({age:.0f} min ago)')

print()
print('=== CONTRACT SIGNAL STATS ===')
for coin in ['btc', 'bnb', 'hype']:
    s = get(f'/contract_stats/{coin}')
    keys = list(s.keys())
    print(f'{coin.upper()}: {json.dumps(s, indent=None)[:600]}')
    print()
