#!/usr/bin/env bash
# Nightly backup for the prediction system.
# - sqlite .backup of every coin DB (WAL-safe, consistent snapshots)
# - git bundle of the full repo history
# - retention: 14 days
set -uo pipefail

APP=/home/chain-deaction/local_prediction/predition-app
DEST=/home/chain-deaction/prediction_backups
TS=$(date +%Y%m%d)
mkdir -p "$DEST"

fail=0
for db in "$APP"/prediction/*/*_prices.db "$APP"/prediction/hype/hype_hl.db; do
    [ -f "$db" ] || continue
    name=$(basename "$db")
    if ! sqlite3 "$db" ".backup '$DEST/${TS}_${name}'" 2>/dev/null; then
        cp "$db" "$DEST/${TS}_${name}" || fail=1
    fi
done

# Full git history bundle (code + all commits, off-state snapshot)
(cd "$APP" && git bundle create "$DEST/${TS}_repo.bundle" --all) 2>/dev/null || true

# Archive-then-delete app-dir logs not written in 30 days (rotated logs are
# already handled by RotatingFileHandler; this clears the legacy ones)
find "$APP/prediction/logs" -name '*.log' -mtime +30 -delete 2>/dev/null || true

# Retention
find "$DEST" -type f -mtime +14 -delete

echo "backup done: $(ls "$DEST" | wc -l) files in $DEST"
exit $fail
