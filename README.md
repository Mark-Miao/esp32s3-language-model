# esp32s3-language-model

**English** | [简体中文](README.zh-CN.md)

Train a ~2.5M-parameter language model on your PC and run it on an ESP32-S3.

This repository is a **toolchain**, not a model release. It contains the data
pipeline, the training code, the quantizer, and the firmware. It does **not**
contain datasets or pretrained weights; both are produced locally by the user.

```
corpus ──► BPE tokenizer ──► uint16 ids ──► PyTorch training ──► int8 model.bin ──► firmware
```

## Design

| Item | Value |
|---|---|
| Architecture | decoder-only Transformer, pre-norm RMSNorm, SiLU MLP, tied embeddings |
| Parameters | 2,582,560 (d=160, 6 layers, 5 heads, ctx 512, MLP 4×) |
| Tokenizer | byte-level BPE, 4096 tokens, special markers for user / assistant / end |
| Training | CPU only, ~7.5M tokens, 6000 steps, several hours on 4 cores |
| Quantization | int8 weights, per-output-channel scales, float32 activations |
| Model file | 2.5 MiB int8 (v2 format, tokenizer embedded) |
| Firmware | ESP-IDF, USB-CDC protocol, KV cache, weights in PSRAM |
| Hardware | ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (N16R8) |

The firmware embeds the model in the app image, so the whole deployment is a
single `eslm.bin` plus the bootloader and partition table.

## Requirements

- Python 3.10+ and ~4 GB RAM
- ESP-IDF v5.3 or newer for the firmware
- An ESP32-S3 board with 8 MB octal PSRAM (N16R8 or larger)

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# download corpora (license notice is printed; see DATA.md), build, train
scripts/train.sh --accept-license

# export int8 weights and build the firmware
. $IDF_PATH/export.sh
scripts/build_firmware.sh

# flash (use the USB port, not the UART port)
cd firmware && idf.py -p /dev/ttyACM0 -b 921600 flash
```

## Using your own corpus

`tools/build_dialogues.py --custom path.jsonl` accepts one JSON object per line
in any of these shapes:

```json
{"turns": [["user", "你好"], ["assistant", "你好呀"]]}
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好呀"}]}
{"conversation": [{"human": "你好", "assistant": "你好呀"}]}
```

Turns are filtered (2–8 turns, 2–40 characters each, mostly Chinese). Then:

```bash
python tools/build_dialogues.py --custom my.jsonl
python tools/train_bpe.py --raw data/raw.txt --vocab-size 4096 --out models/tokenizer
python tools/build_ids.py --dialogues data/dialogues.jsonl --tokenizer models/tokenizer
python tools/train.py --config configs/chat-lccc.json
```

You are responsible for the rights to any corpus you use.

## Serial protocol

The firmware listens on native USB-CDC (the port labelled `USB`). Logs go to
UART0. Lines are `\n`-terminated; generation is streamed back as UTF-8.

```
PING                      -> PONG eslm fw=1 ctx=512
INFO                      -> INFO params=... ctx=... dtype=int8 ...
CHAT <n> <temp> <id...>   -> generated text, then "[OK] n=.. ms=.. tok_s=.."
RESET                     -> OK
```

`CHAT` takes token ids produced by `models/tokenizer`, feeds them as the prompt,
and generates until `<|e|>` or `n` tokens. A minimal pyserial client:

```python
import serial
from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer("models/tokenizer/vocab.json", "models/tokenizer/merges.txt")
tok.add_special_tokens(["<|u|>", "<|a|>", "<|e|>"])
ids = [tok.token_to_id("<|u|>")] + tok.encode("你好", add_special_tokens=False).ids \
      + [tok.token_to_id("<|a|>")]

ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
ser.write(f"CHAT 80 0.6 {' '.join(map(str, ids))}\n".encode())
print(ser.read_until(b"[OK]").decode("utf-8", "replace"))
```

## Model file format (v2)

```
"ESLM" | u32 version=2 | u32 dtype(0=f32, 1=int8) | u32 vocab, d, n_layer, n_head, ctx
       | u32 special_u, special_a, special_e | u32 vocab_blob_bytes
u32 token_offsets[vocab+1] | u8 token_bytes[vocab_blob_bytes]
weights: tok_emb, pos_emb, then per layer rms1, wq, wk, wv, wo, rms2, w1, w2, finally rms_f
```

2-D matrices are row-major (row = output channel). In int8 mode each matrix is
preceded by `f32 scales[out]` and padded to a 4-byte boundary; 1-D norm vectors
stay float32. `tools/export.py --check` re-reads the file and compares it with
PyTorch (`max|diff| = 0` for f32).

## Data

Datasets are downloaded by `tools/download_data.py` and are never committed.
See [DATA.md](DATA.md) for sources, licenses, and handling obligations.

- **LCCC** — research-use license, user-generated Weibo text
- **Slim-LCCC** — Apache-2.0, derived from LCCC
- **Muice** — CC-BY-NC-4.0, non-commercial, opt-in only

## Limitations

- 2.5M parameters learn the *shape* of short Chinese chat, not knowledge. The
  model produces plausible-sounding but frequently wrong or empty replies.
- Training on spontaneous conversation means the output can be rude or
  nonsensical. It is not suitable for customer-facing use.
- On-device throughput depends on context length and your board; measure it.
  The 2.58M int8 model is compute-bound on the S3 (single-digit to low tens of
  tokens per second).

## Legal

This project is licensed under **AGPL-3.0** (see [LICENSE](LICENSE)).
Commercial use is permitted *only* under the terms of that license: if you
modify the software and make it available, including as a network service, you
must offer the complete corresponding source under the same license.

**Intended use:** research, education, and hobbyist experimentation.

**Prohibited uses:** generating illegal content; fraud, scams, or deceptive
impersonation; deepfakes of real people; harassment; surveillance; medical,
legal, or financial advice; safety-critical or production systems.

**No warranty.** The software and any models you train are provided "as is",
without warranty of any kind. You are solely responsible for the data you use,
the models you train, and the content they generate, including compliance with
copyright, privacy, and other applicable law. Datasets may contain personal
information; do not redistribute them.

ESP32 and ESP32-S3 are trademarks of Espressif Systems. This project is not
affiliated with or endorsed by Espressif. ModelScope is a service of Alibaba
Group; this project is not affiliated with it either.

## Layout

```
README.zh-CN.md             Chinese translation
configs/chat-lccc.json      training configuration
tools/
  download_data.py          fetch corpora (license gate)
  build_dialogues.py        filter/normalize dialogues, or --custom
  train_bpe.py              train the BPE tokenizer
  build_ids.py              dialogues -> uint16 token stream
  train.py                  PyTorch training
  export.py                 checkpoint -> model.bin (f32 / int8, v1 / v2)
  bpe_utils.py              BPE vocab -> device byte blob
firmware/
  CMakeLists.txt, partitions.csv, sdkconfig.defaults
  main/{main.c, infer.c, infer.h}
scripts/
  train.sh                  end-to-end data + training
  build_firmware.sh         export + idf.py build
```

## Contributions

By submitting a pull request you agree that your contribution is licensed
under AGPL-3.0, the same license as the rest of this repository.

## Acknowledgments

- LCCC / CDial-GPT, Tsinghua University (COAI)
- Slim-LCCC-zh and Muice dataset maintainers on ModelScope
- ESP-IDF and the ESP32-S3 hardware documentation
- The `llama2.c` project for popularizing readable C inference
