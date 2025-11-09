#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 path/to/lib.so"
  exit 1
fi

SO="$1"

if [ ! -f "$SO" ]; then
  echo "Error: file not found: $SO"
  exit 1
fi

# 选工具：优先 nvdisasm，没有就用 cuobjdump
if command -v nvdisasm &>/dev/null; then
  DISASM="nvdisasm -g -c"
elif command -v cuobjdump &>/dev/null; then
  DISASM="cuobjdump --dump-sass"
else
  echo "Error: nvdisasm or cuobjdump not found in PATH."
  echo "Please ensure CUDA bin directory is in PATH."
  exit 1
fi

TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_OUT"' EXIT

echo "[INFO] Disassembling $SO using: $DISASM"
$DISASM "$SO" > "$TMP_OUT"

echo "[INFO] Scanning for FP4 MMA instructions..."

# 1. 精确匹配：包含 mma 且操作数类型带 .fp4.fp4（Blackwell FP4 TensorCore 的典型形式）
if grep -Eiq 'mma([.:][^ ]*)*\.fp4\.fp4' "$TMP_OUT"; then
  echo "✅ Detected FP4 MMA instructions in $SO:"
  grep -Ein 'mma([.:][^ ]*)*\.fp4\.fp4' "$TMP_OUT" | head -n 40
  exit 0
fi

# 2. 宽松匹配：看看是否出现任何 .fp4 相关标记（可能是新指令命名）
if grep -Eiq '\.fp4' "$TMP_OUT"; then
  echo "⚠️ Found FP4-related tokens, but no direct 'mma.*.fp4.fp4' pattern."
  echo "Top related lines:"
  grep -Ein '\.fp4' "$TMP_OUT" | head -n 40
  exit 2
fi

echo "❌ No FP4 MMA instructions detected."
echo "   Likely using FP8/BF16 TensorCore fallback (no native FP4 MMA in this .so)."
exit 3
