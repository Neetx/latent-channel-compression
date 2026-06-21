#!/usr/bin/env bash
# Step 2 — sequential_scaled × mbppplus: the high-baseline test of the guiding
# hypothesis (does compression break on code once the system is actually good at it,
# removing Step 1's floor-effect confound?).
#   Phase 1: sampled bit-rate ladder (bits 0,2,4,8), batch=$LADDER_B, --no-capture
#   Phase 2: greedy paired fidelity (bits 0,4), batch=$CAP_B, capture
# Batches are env-overridable because a 4B agent's KV at max_new_tokens=4000 is large
# (set them from probe_scaled_feasibility.py). Conservative defaults below.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
PY="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DRIVER="experiments/fidelity_sweep/local_pkg/fidelity_local.py"
OUT="${LCC_RUN_ROOT:-$HOME/lcc/runs}/step2_scaled_mbppplus"
mkdir -p "$OUT"
N="${STEP2_N:-250}"
LADDER_B="${STEP2_LADDER_B:-4}"   # set from the VRAM probe
CAP_B="${STEP2_CAP_B:-1}"         # greedy capture is the tightest case

echo "=== STEP2 scaled/mbppplus START n=$N ladder_b=$LADDER_B cap_b=$CAP_B $(date '+%Y-%m-%d %H:%M:%S') ==="

echo "### PHASE 1: sampled ladder (batch=$LADDER_B) ###"
for b in 0 2 4 8; do
  echo "--- ladder bits=$b START $(date '+%H:%M:%S') ---"
  /usr/bin/time -v "$PY" "$DRIVER" --style sequential_scaled --dataset mbppplus --bits "$b" --t 3 \
      --n-samples "$N" --batch-size "$LADDER_B" --no-capture --out "$OUT" \
      > "$OUT/ladder_b${b}_n${N}.log" 2>&1
  echo "--- ladder bits=$b EXIT=$? END $(date '+%H:%M:%S') ---"
done

echo "### PHASE 2: greedy paired fidelity (batch=$CAP_B, capture) ###"
for b in 0 4; do
  echo "--- fidelity bits=$b START $(date '+%H:%M:%S') ---"
  /usr/bin/time -v "$PY" "$DRIVER" --style sequential_scaled --dataset mbppplus --bits "$b" --t 3 \
      --n-samples "$N" --batch-size "$CAP_B" --out "$OUT" \
      > "$OUT/fidelity_b${b}_n${N}.log" 2>&1
  echo "--- fidelity bits=$b EXIT=$? END $(date '+%H:%M:%S') ---"
done

echo "=== STEP2 scaled/mbppplus DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
for b in 0 2 4 8; do
  echo "  ladder bits=$b : $(grep -oE 'accuracy=[0-9.]+%' "$OUT/ladder_b${b}_n${N}.log" 2>/dev/null | tail -1)"
done
for b in 0 4; do
  echo "  fidelity bits=$b : $(grep -oE 'accuracy=[0-9.]+%' "$OUT/fidelity_b${b}_n${N}.log" 2>/dev/null | tail -1)"
done
