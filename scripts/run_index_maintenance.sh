#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
INDEX_SCRIPT="$PROJECT_ROOT/scripts/index_documents.py"
LOG_DIRECTORY="$PROJECT_ROOT/storage/maintenance"
LOG_PATH="$LOG_DIRECTORY/index-maintenance-$(date +%Y%m%d).log"

if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "No se encontró Python ejecutable en: $PYTHON_PATH" >&2
    exit 1
fi

if [[ ! -f "$INDEX_SCRIPT" ]]; then
    echo "No se encontró el script: $INDEX_SCRIPT" >&2
    exit 1
fi

mkdir -p "$LOG_DIRECTORY"
cd "$PROJECT_ROOT"

{
    echo "[$(date --iso-8601=seconds)] Inicio del mantenimiento."
    "$PYTHON_PATH" "$INDEX_SCRIPT" "$@"
    EXIT_CODE=$?
    echo "[$(date --iso-8601=seconds)] Fin del mantenimiento. Código: $EXIT_CODE"
    exit "$EXIT_CODE"
} >> "$LOG_PATH" 2>&1
