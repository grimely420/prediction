import sqlite3
base = '/home/chain-deaction/local_prediction/predition-app/prediction'
cutoff = '2026-09-24T21:13:00'
for coin, db in [('btc', 'bitcoin/bitcoin_prices.db'), ('bnb', 'bnb/bnb_prices.db'), ('hype', 'hype/hype_prices.db')]:
    conn = sqlite3.connect(f'{base}/{db}')
    total = conn.execute('SELECT COUNT(*) FROM predictions WHERE prediction_time > ?', (cutoff,)).fetchone()[0]
    rows = conn.execute(
        "SELECT horizon_min, COUNT(*), "
        "SUM(CASE WHEN (predicted_price-current_price)*(actual_price-current_price)>0 THEN 1 ELSE 0 END), "
        "AVG(ABS(error_pct)) "
        "FROM predictions WHERE checked=1 AND prediction_time > ? "
        "GROUP BY horizon_min", (cutoff,)).fetchall()
    print(f'{coin}: logged_since_deploy={total}')
    for h, n, corr, mae in rows:
        print(f'   {h}m: validated={n} dir_acc={100*(corr or 0)/n:.1f}% mae={mae:.4f}%')
    rate = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction_time > strftime('%Y-%m-%dT%H:%M:%S','now','-15 minutes')"
    ).fetchone()[0]
    print(f'   write_rate_last15min={rate} ({rate/15:.1f}/min)')
