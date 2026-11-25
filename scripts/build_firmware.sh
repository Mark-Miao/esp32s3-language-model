#!/usr/bin/env bash
# Export int8 model.bin, then build the ESP-IDF firmware.
#   scripts/build_firmware.sh [checkpoint]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:-checkpoints/best.pt}
PY=${PYTHON:-python}

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[ -f models/tokenizer/vocab.json ] || { echo "tokenizer missing; run training first" >&2; exit 1; }

$PY tools/export.py "$CKPT" --out models/model.bin \
    --tokenizer models/tokenizer --dtype int8 --check

if [ -z "${IDF_PATH:-}" ]; then
  echo "ESP-IDF not active. Run: . \$IDF_PATH/export.sh" >&2
  exit 1
fi

cd firmware
idf.py build
echo
echo "Flash with: idf.py -p <PORT> -b 921600 flash"
