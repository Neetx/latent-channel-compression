#!/usr/bin/env bash
# Step 1 — mbppplus / sequential_light (the "trajectory IS the output" cell).
#   Phase 1: sampled bit-rate ladder (bits 0,2,4,8), batch=16, --no-capture
#   Phase 2: greedy paired fidelity (bits 0,4), batch=2, capture. output_scores over
#            up to 4000 tokens is memory-heavy (~2.4 GB/seq); batch=2 (~5 GB worst case)
#            is the safe sweet spot on 16 GB and ~2x faster than batch=1. Greedy is
#            batch-invariant, so batch choice does not change the per-problem result.
# latent_length=16 and temperature=0.2 are applied automatically by run.py's
# RELEASE_RECOMMENDED_SETTINGS / infer_temperature for mbppplus.
export HF_HOME="$HOME/lcc/hf-cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 2
PY="$HOME/lcc/.venv/bin/python"
DRIVER="experiments/fidelity_sweep/local_pkg/fidelity_local.py"
OUT="${LCC_RUN_ROOT:-$HOME/lcc/runs}/step1_mbppplus"
mkdir -p "$OUT"
N="${STEP1_N:-250}"

echo "=== STEP1 mbppplus START n=$N $(date '+%Y-%m-%d %H:%M:%S') ==="

echo "### PHASE 1: sampled ladder (batch=16) ###"
for b in 0 2 4 8; do
  echo "--- ladder bits=$b START $(date '+%H:%M:%S') ---"
  /usr/bin/time -v "$PY" "$DRIVER" --dataset mbppplus --bits "$b" --t 3 \
      --n-samples "$N" --batch-size 16 --no-capture \
      > "$OUT/ladder_b${b}_n${N}.log" 2>&1
  echo "--- ladder bits=$b EXIT=$? END $(date '+%H:%M:%S') ---"
done

echo "### PHASE 2: greedy paired fidelity (batch=2, capture) ###"
for b in 0 4; do
  echo "--- fidelity bits=$b START $(date '+%H:%M:%S') ---"
  /usr/bin/time -v "$PY" "$DRIVER" --dataset mbppplus --bits "$b" --t 3 \
      --n-samples "$N" --batch-size 2 \
      > "$OUT/fidelity_b${b}_n${N}.log" 2>&1
  echo "--- fidelity bits=$b EXIT=$? END $(date '+%H:%M:%S') ---"
done

echo "=== STEP1 mbppplus DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "--- LADDER accuracies ---"
for b in 0 2 4 8; do
  echo "  bits=$b : $(grep -oE 'accuracy=[0-9.]+%' "$OUT/ladder_b${b}_n${N}.log" 2>/dev/null | tail -1)"
done
echo "--- FIDELITY accuracies ---"
for b in 0 4; do
  echo "  bits=$b : $(grep -oE 'accuracy=[0-9.]+%' "$OUT/fidelity_b${b}_n${N}.log" 2>/dev/null | tail -1)"
done
