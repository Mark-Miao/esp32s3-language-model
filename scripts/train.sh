#!/usr/bin/env bash
# Full pipeline: download -> dialogues -> BPE -> ids -> train.
# Extra args are passed to download_data.py (e.g. --include-nc).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
command -v "$PY" >/dev/null || {
  echo "python not found; create a venv and 'pip install -r requirements.txt' first" >&2
  exit 1
}

INC_NC=0
for a in "$@"; do [ "$a" = "--include-nc" ] && INC_NC=1; done

"$PY" tools/download_data.py --accept-license "$@"

if [ "$INC_NC" = "1" ]; then
  "$PY" tools/build_dialogues.py --include-nc
else
  "$PY" tools/build_dialogues.py
fi

"$PY" tools/train_bpe.py --raw data/raw.txt --vocab-size 4096 --out models/tokenizer
"$PY" tools/build_ids.py --dialogues data/dialogues.jsonl --tokenizer models/tokenizer \
    --out data/chat_ids.bin --ctx 512
"$PY" tools/train.py --config configs/chat-lccc.json

echo
echo "Done. Next:"
echo "  scripts/build_firmware.sh"
