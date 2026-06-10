#!/bin/bash
# Model-matrix battery driver: one `go test` process per model (crash
# isolation). Usage: run-matrix.sh [cpu|gpu] [models-dir]
# Logs land in ~/litert-matrix/<backend>/<model>.log.
backend=${1:-cpu}
models=${2:-$HOME/models}
cd "$(dirname "$0")"
export LITERT_LIB="$(pwd)/lib"
export LITERT_LM_BACKEND=$backend
export LITERT_DECODE_STATS=1
outdir="$HOME/litert-matrix/$backend"
mkdir -p "$outdir"
for m in "$models"/*.litertlm; do
  base=$(basename "$m" .litertlm)
  export LITERT_LM_MODELS=$m
  echo "=== $base [$backend] $(date +%H:%M:%S) ==="
  go test ./lm -run 'TestModelMatrix$' -v -timeout 30m -count=1 > "$outdir/$base.log" 2>&1
  echo "exit=$?"
  grep -E -- '--- (PASS|FAIL|SKIP)|greedy \(|turn 2|decode:|^FAIL|open:' "$outdir/$base.log"
done
echo "BATTERY DONE [$backend]"
