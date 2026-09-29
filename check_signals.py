import sqlite3
base = '/home/chain-deaction/local_prediction/predition-app/prediction'
for coin, db in [('btc', 'bitcoin/bitcoin_prices.db'), ('bnb', 'bnb/bnb_prices.db'), ('hype', 'hype/hype_prices.db')]:
    conn = sqlite3.connect(f'{base}/{db}')
    rows = conn.execute(
        'SELECT round_start, phase, prob_yes, bias, strike_price, current_price, resolved '
        'FROM contract_signals ORDER BY id DESC LIMIT 6').fetchall()
    print(coin.upper())
    for r in rows:
        print('  ', r)
