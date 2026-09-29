import json, urllib.request

def get(path):
    return json.load(urllib.request.urlopen(f'http://localhost:5000{path}', timeout=30))

print('=== /kalshi (all coins) ===')
allk = get('/kalshi')
for coin in ['btc', 'bnb', 'hype']:
    d = allk.get(coin)
    if not d:
        print(coin, 'MISSING')
        continue
    dec = d['decision']
    pr = d['pricing']
    rs = d.get('round_series') or {}
    cm = d.get('contract_model') or {}
    print(f"{coin.upper():4s} | {dec['recommendation']:32s} | YES {dec['yes_cents']}c NO {dec['no_cents']}c | "
          f"contract p={cm.get('p_up')} | strike {pr['round_open_price']} | cur {pr['current_price']} | "
          f"chart pts {len(rs.get('points', []))}")

print()
print('=== /contract_stats ===')
for coin in ['btc', 'bnb', 'hype']:
    s = get(f'/contract_stats/{coin}')
    print(f"{coin.upper()}: resolved={s.get('resolved_signals', 0)} pending={s.get('pending_signals', 0)} "
          f"win_rate={s.get('win_rate_pct')}%")
    for ph, v in (s.get('by_phase') or {}).items():
        print(f"   {ph}: n={v.get('total')} wins={v.get('wins')} wr={v.get('win_rate_pct')}%")
