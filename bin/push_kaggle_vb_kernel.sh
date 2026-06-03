#!/bin/bash
# Push a Kaggle T4 kernel for Variant B experiments with custom parameters.
#
# Usage: bin/push_kaggle_vb_kernel.sh <bits> <n_samples> <batch_size>
#   bits         — 0 (baseline, no quantization) or N>0 (Variant B N-bit)
#   n_samples    — e.g. 50, 250
#   batch_size   — e.g. 4 (fp32 safe on T4 16GB)
#
# Examples:
#   bin/push_kaggle_vb_kernel.sh 4 250 4    # Variant B 4-bit at n=250 b=4
#   bin/push_kaggle_vb_kernel.sh 0 250 4    # baseline at n=250 b=4
set -e

BITS="$1"; N="$2"; B="$3"
if [ -z "$BITS" ] || [ -z "$N" ] || [ -z "$B" ]; then
    echo "usage: $0 <bits> <n_samples> <batch_size>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$PROJECT_ROOT/experiments/variant_b_ladder_t4_kaggle/kernel_pkg"
TMP="/tmp/vb_n${N}_b${B}_bits${BITS}"

rm -rf "$TMP" && mkdir -p "$TMP"
cp "$BASE/phase0g_kernel.py" "$TMP/phase0g_kernel.py"

# Patch defaults via sed
sed -i.bak "s/VARIANT_B_BITS = int(os.environ.get(\"VARIANT_B_BITS\", \"0\"))/VARIANT_B_BITS = int(os.environ.get(\"VARIANT_B_BITS\", \"${BITS}\"))/" "$TMP/phase0g_kernel.py"
sed -i.bak "s/N_SAMPLES = int(os.environ.get(\"N_SAMPLES\", \"50\"))/N_SAMPLES = int(os.environ.get(\"N_SAMPLES\", \"${N}\"))/" "$TMP/phase0g_kernel.py"
sed -i.bak "s/BATCH_SIZE = int(os.environ.get(\"BATCH_SIZE\", \"4\"))/BATCH_SIZE = int(os.environ.get(\"BATCH_SIZE\", \"${B}\"))/" "$TMP/phase0g_kernel.py"
rm "$TMP/phase0g_kernel.py.bak"

# Pretty slug: "baseline" for bits=0, else "vbN"
SLUG_BITS="${BITS}"
[ "$BITS" = "0" ] && SLUG_BITS="baseline"

cat > "$TMP/kernel-metadata.json" << JSON
{
  "id": "<YOUR_KAGGLE_USERNAME>/rmas-vb${SLUG_BITS}-n${N}-b${B}",
  "title": "rmas vb${SLUG_BITS} n${N} b${B}",
  "code_file": "phase0g_kernel.py",
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
