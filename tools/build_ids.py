#!/usr/bin/env python3
"""Tokenize dialogues into the uint16 id stream consumed by tools/train.py.

Template per exchange:
    <|u|> user text <|a|> assistant text <|e|>

Output: data/chat_ids.bin + data/chat_ids_meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
from tokenizers import ByteLevelBPETokenizer

SPECIALS = ["<|u|>", "<|a|>", "<|e|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="data/dialogues.jsonl")
    ap.add_argument("--tokenizer", default="models/tokenizer")
    ap.add_argument("--out", default="data/chat_ids.bin")
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()

    vocab = os.path.join(args.tokenizer, "vocab.json")
    merges = os.path.join(args.tokenizer, "merges.txt")
    if not (os.path.exists(vocab) and os.path.exists(merges)):
        sys.exit(f"tokenizer not found in {args.tokenizer}; run tools/train_bpe.py first")

    tok = ByteLevelBPETokenizer(vocab, merges)
    tok.add_special_tokens(SPECIALS)
    U, A, E = (tok.token_to_id(s) for s in SPECIALS)

    ids, kept, skipped = [], 0, 0
    for line in open(args.dialogues, encoding="utf-8"):
        try:
            turns = json.loads(line)["turns"]
        except Exception:  # noqa: BLE001
            continue
        seq = []
        for role, text in turns:
            enc = tok.encode(text, add_special_tokens=False).ids
            if role == "user":
                seq += [U] + enc
            else:
                seq += [A] + enc + [E]
        if not seq or len(seq) > args.ctx:
            skipped += 1
            continue
        ids += seq
        kept += 1

    if not ids:
        sys.exit("no token ids produced")

    arr = np.asarray(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    arr.tofile(args.out)
    meta = {"vocab_size": tok.get_vocab_size(), "special_ids": {"u": U, "a": A, "e": E},
            "tokens": int(arr.size), "dialogues": kept, "skipped_long": skipped,
            "ctx": args.ctx}
    with open(os.path.join(os.path.dirname(args.out) or ".", "chat_ids_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ids] {kept} dialogues, {skipped} skipped, {arr.size/1e6:.2f}M tokens "
          f"({arr.nbytes/2**20:.1f} MiB) -> {args.out}")
    print(f"[ids] special u={U} a={A} e={E}, vocab={meta['vocab_size']}")


if __name__ == "__main__":
    main()
