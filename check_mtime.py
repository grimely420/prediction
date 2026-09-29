import os, time
base = "/home/chain-deaction/local_prediction/predition-app/prediction"
now = time.time()
for coin, d in [("btc", "bitcoin"), ("bnb", "bnb"), ("hype", "hype")]:
    for f in ["xgb_5min_clf.pkl", "xgb_15min_clf.pkl", "kalshi_contract_latest.pkl"]:
        p = f"{base}/{d}/models/{f}"
        if os.path.exists(p):
            print(coin, f, round((now - os.path.getmtime(p)) / 60, 1), "min ago")
