#!/usr/bin/env bash
# autoresearch/run_experiment.sh
#
# Run one experiment end-to-end. Redirects stdout+stderr to run.log so the
# terminal buffer is not flooded. Extracts the key metric and prints it.
#
# Usage:
#   ./autoresearch/run_experiment.sh
#   ./autoresearch/run_experiment.sh --dry-run   # just verify harness loads
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${AUTORESEARCH_ENV:-datathon}"
TIMEOUT_SECS="${AUTORESEARCH_TIMEOUT:-900}"

# Resolve a usable python binary for the env. Prefer the env's python directly
# (no shell activation needed, works from any parent shell).
if [[ -x "$HOME/miniconda3/envs/$ENV_NAME/bin/python" ]]; then
    PY="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
elif [[ -x "$HOME/anaconda3/envs/$ENV_NAME/bin/python" ]]; then
    PY="$HOME/anaconda3/envs/$ENV_NAME/bin/python"
elif command -v conda >/dev/null 2>&1; then
    PY="$(conda run -n "$ENV_NAME" --no-capture-output which python)"
else
    PY="$(command -v python)"
fi
echo "[autoresearch] using python: $PY"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[autoresearch] dry run: verifying harness"
    "$PY" -m autoresearch.prepare
    exit 0
fi

LOG="$ROOT/autoresearch/run.log"
: > "$LOG"
echo "[autoresearch] launching train.py (timeout ${TIMEOUT_SECS}s)"

if timeout "${TIMEOUT_SECS}" "$PY" -m autoresearch.train > "$LOG" 2>&1; then
    echo "[autoresearch] run finished."
else
    ec=$?
    echo "[autoresearch] run exited with code $ec (timeout or crash). tail:"
    tail -n 40 "$LOG"
    exit "$ec"
fi

echo
echo "── key metrics ──"
grep -E "^val_mae_rev:|^val_mae_cogs:|^val_rmse_rev:|^val_r2_rev:" "$LOG" || {
    echo "(no metrics printed — run crashed, see $LOG)"
    exit 1
}
