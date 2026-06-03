#!/bin/bash
# Push one fidelity-sweep kernel with a given (bits, T, n, b).
#
# Usage: bin/push_fidelity_kernel.sh <bits> <T> <n_samples> <batch_size>
#   bits         — 0 (REF) or N>0 (Variant B N-bit)
#   T            — num_recursive_rounds (channel-traversal depth)
#   n_samples    — math500 problems (e.g. 50)
#   batch_size   — keep 4 for T4 fp32
set -e

BITS="$1"; T="$2"; N="$3"; B="$4"
if [ -z "$BITS" ] || [ -z "$T" ] || [ -z "$N" ] || [ -z "$B" ]; then
    echo "usage: $0 <bits> <T> <n_samples> <batch_size>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$PROJECT_ROOT/experiments/fidelity_sweep/kernel_pkg"
TMP="/tmp/fidelity_vb${BITS}_T${T}_n${N}_b${B}"

rm -rf "$TMP" && mkdir -p "$TMP"
cp "$BASE/fidelity_kernel.py" "$TMP/fidelity_kernel.py"

# Patch in the per-run defaults via sed (Kaggle doesn't expose env vars to kernels)
sed -i.bak "s/VARIANT_B_BITS = int(os.environ.get(\"VARIANT_B_BITS\", \"0\"))/VARIANT_B_BITS = int(os.environ.get(\"VARIANT_B_BITS\", \"${BITS}\"))/" "$TMP/fidelity_kernel.py"
sed -i.bak "s/NUM_RECURSIVE_ROUNDS = int(os.environ.get(\"NUM_RECURSIVE_ROUNDS\", \"3\"))/NUM_RECURSIVE_ROUNDS = int(os.environ.get(\"NUM_RECURSIVE_ROUNDS\", \"${T}\"))/" "$TMP/fidelity_kernel.py"
sed -i.bak "s/N_SAMPLES = int(os.environ.get(\"N_SAMPLES\", \"50\"))/N_SAMPLES = int(os.environ.get(\"N_SAMPLES\", \"${N}\"))/" "$TMP/fidelity_kernel.py"
sed -i.bak "s/BATCH_SIZE = int(os.environ.get(\"BATCH_SIZE\", \"4\"))/BATCH_SIZE = int(os.environ.get(\"BATCH_SIZE\", \"${B}\"))/" "$TMP/fidelity_kernel.py"
rm "$TMP/fidelity_kernel.py.bak"

# REF vs INT4 slug
SLUG_BITS="${BITS}"
[ "$BITS" = "0" ] && SLUG_BITS="ref"

cat > "$TMP/kernel-metadata.json" << JSON
{
  "id": "<YOUR_KAGGLE_USERNAME>/rmas-fid-${SLUG_BITS}-T${T}-n${N}",
  "title": "RMAS fidelity ${SLUG_BITS} T=${T} n=${N}",
  "code_file": "fidelity_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": ["<YOUR_KAGGLE_USERNAME>/lqc-src-bundle"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

cd "$PROJECT_ROOT"
./bin/kaggle kernels push -p "$TMP" --accelerator NvidiaTeslaT4 2>&1 | tail -2
