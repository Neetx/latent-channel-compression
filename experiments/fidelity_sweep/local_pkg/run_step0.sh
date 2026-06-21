#!/usr/bin/env bash
# Step 0 full re-baseline of math500 in local bf16.
#   Phase 1: sampled bit-rate ladder (bits 0,2,4,8), batch=16, --no-capture
#   Phase 2: greedy paired fidelity (bits 0,4), batch=2, capture (Tier-2)
# Robust: continues on per-run error; each run logs separately; final summary.
export HF_HOME="$HOME/lcc/hf-cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
PY="$HOME/lcc/.venv/bin/python"
DRIVER="experiments/fidelity_sweep/local_pkg/fidelity_local.py"
OUT="${LCC_RUN_ROOT:-$HOME/lcc/runs}/step0"
mkdir -p "$OUT"
N="${STEP0_N:-250}"

echo "=== STEP0 START n=$N $(date '+%Y-%m-%d %H:%M:%S') ==="

echo "### PHASE 1: sampled accuracy ladder (batch=16) ###"
for b in 0 2 4 8; do
  echo "--- ladder bits=$b START $(date '+%H:%M:%S') ---"
  /usr/bin/time -v "$PY" "$DRIVER" --bits "$b" --t 3 --n-samples "$N" --batch-size 16 --no-capture \
      > "$OUT/ladder_b${b}_n${N}.log" 2>&1
  echo "--- ladder bits=$b EXIT=$? END $(date '+%H:%M:%S') ---"
done

echo "### PHASE 2: greedy paired fidelity (batch=2, capture) ###"
for b in 0 4; do
  echo "--- fidelity bits=$b START $(date '+%H:%M:%S') ---"
  /usr/bin/time -v "$PY" "$DRIVER" --bits "$b" --t 3 --n-samples "$N" --batch-size 2 \
      > "$OUT/fidelity_b${b}_n${N}.log" 2>&1
  echo "--- fidelity bits=$b EXIT=$? END $(date '+%H:%M:%S') ---"
done

echo "=== STEP0 DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "--- LADDER (sampled) accuracies ---"
for b in 0 2 4 8; do
  echo "  bits=$b : $(grep -oE 'accuracy=[0-9.]+%' "$OUT/ladder_b${b}_n${N}.log" | tail -1)"
done
echo "--- FIDELITY (greedy) accuracies ---"
for b in 0 4; do
  echo "  bits=$b : $(grep -oE 'accuracy=[0-9.]+%' "$OUT/fidelity_b${b}_n${N}.log" | tail -1)"
done
