#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

RUNNER_PATH="$PROJECT_ROOT/scripts/run_index_maintenance.sh"
CRON_SCHEDULE="${1:-0 3 * * *}"

if [[ ! -f "$RUNNER_PATH" ]]; then
    echo "No se encontró el ejecutor: $RUNNER_PATH" >&2
    exit 1
fi

chmod +x "$RUNNER_PATH"

CURRENT_CRONTAB="$(crontab -l 2>/dev/null || true)"

FILTERED_CRONTAB="$(
    printf '%s\n' "$CURRENT_CRONTAB" |
        grep -Fv "$RUNNER_PATH" || true
)"

{
    printf '%s\n' "$FILTERED_CRONTAB"
    printf '%s %q\n' "$CRON_SCHEDULE" "$RUNNER_PATH"
} | sed '/^[[:space:]]*$/d' | crontab -

echo "Cron instalado:"
echo "$CRON_SCHEDULE $RUNNER_PATH"
echo ""
echo "Verificación:"
echo "crontab -l"
