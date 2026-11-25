#!/usr/bin/env python3
"""Train a BPE tokenizer on the dialogue corpus.

Outputs models/tokenizer/{vocab.json,merges.txt}. Special tokens:
  <|u|> user turn, <|a|> assistant turn, <|e|> end of exchange
"""
import argparse
import json
import os

from tokenizers import ByteLevelBPETokenizer

SPECIALS = ["<|u|>", "<|a|>", "<|e|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw.txt")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--out", default="models/tokenizer")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(files=[args.raw], vocab_size=args.vocab_size, min_frequency=2,
              special_tokens=SPECIALS, show_progress=True)
    tok.save_model(args.out)

    vocab = json.load(open(os.path.join(args.out, "vocab.json"), encoding="utf-8"))
    sizes = [len(k.encode("utf-8")) for k in vocab]
    single = sum(1 for s in sizes if s == 1)
    print(f"[bpe] vocab={len(vocab)}  single-byte={single}  "
          f"avg-token-bytes={sum(sizes)/len(sizes):.2f}")
    for s in SPECIALS:
        print(f"  {s} -> id {tok.token_to_id(s)}")


if __name__ == "__main__":
    main()
