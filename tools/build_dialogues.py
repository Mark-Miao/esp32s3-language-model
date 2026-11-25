#!/usr/bin/env python3
"""Build data/dialogues.jsonl (+ data/raw.txt) from LCCC/Slim-LCCC or your own file.

Default sources (downloaded by tools/download_data.py):
  data/raw/LCCC-base-split.zip        LCCC base, streamed with ijson
  data/raw/LCCC_sharegpt_10K.jsonl    Slim-LCCC, Apache-2.0
  data/raw/muice_train.jsonl          optional, CC-BY-NC-4.0 (--include-nc)

Custom corpus: pass --custom path.jsonl. Accepted schemas per line:
  {"turns": [["user", "..."], ["assistant", "..."]]}
  {"conversations": [{"from": "human|gpt", "value": "..."}]}
  {"conversation": [{"human": "...", "assistant": "..."}]}
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import zipfile

BAD = re.compile(r"http|www\.|@[A-Za-z0-9_]+|转发微博|//@|微信|公众号")
CJK = re.compile(r"[一-鿿]")
PUNCT_ONLY = re.compile(r"^[\W_]+$")


def clean(s):
    return s.replace(" ", "").replace("　", "").strip()


def ok_utt(s, lo=2, hi=40):
    if not (lo <= len(s) <= hi):
        return False
    if BAD.search(s) or PUNCT_ONLY.match(s):
        return False
    return len(CJK.findall(s)) >= max(1, int(len(s) * 0.3))


def normalize(obj):
    """Turn one record into [[role, text], ...] or None."""
    turns = []
    if isinstance(obj.get("turns"), list):
        for item in obj["turns"]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                turns.append([str(item[0]), clean(str(item[1]))])
    elif isinstance(obj.get("conversations"), list):
        for m in obj["conversations"]:
            role = "user" if m.get("from") in ("human", "user") else "assistant"
            turns.append([role, clean(str(m.get("value", "")))])
    elif isinstance(obj.get("conversation"), list):
        for ev in obj["conversation"]:
            if ev.get("human"):
                turns.append(["user", clean(str(ev["human"]))])
            if ev.get("assistant"):
                turns.append(["assistant", clean(str(ev["assistant"]))])
    turns = [t for t in turns if t[1]]
    if len(turns) % 2:
        turns = turns[:-1]
    if not (2 <= len(turns) <= 8):
        return None
    if not all(ok_utt(text) for _, text in turns):
        return None
    if sum(len(text) for _, text in turns) > 480:
        return None
    return turns


def lccc(zip_path, limit):
    try:
        import ijson
    except ImportError:
        sys.exit("ijson is required for LCCC: pip install ijson")
    out, seen = [], set()
    with zipfile.ZipFile(zip_path) as z:
        with z.open("LCCC-base_train.json") as f:
            for item in ijson.items(f, "item"):
                utts = [clean(u) for u in item if isinstance(u, str)]
                utts = [u for u in utts if u]
                if not (2 <= len(utts) <= 8):
                    continue
                if len(utts) % 2:
                    utts = utts[:-1]
                if len(utts) < 2 or not all(ok_utt(u) for u in utts):
                    continue
                text = "".join(utts)
                if len(text) > 240:
                    continue
                h = hashlib.md5(text.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                out.append([["user" if i % 2 == 0 else "assistant", u]
                            for i, u in enumerate(utts)])
                if len(out) >= limit:
                    break
    return out


def read_jsonl(path):
    out = []
    if not path or not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        t = normalize(obj)
        if t:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/dialogues.jsonl")
    ap.add_argument("--raw-out", default="data/raw.txt")
    ap.add_argument("--custom", default=None, help="use your own JSONL instead of the datasets")
    ap.add_argument("--max-lccc", type=int, default=300000)
    ap.add_argument("--include-nc", action="store_true", help="also mix the CC-BY-NC-4.0 data")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    if args.custom:
        dialogues = read_jsonl(args.custom)
        if not dialogues:
            sys.exit(f"no usable dialogues in {args.custom}")
        print(f"[custom] {len(dialogues)} dialogues")
    else:
        lc = os.path.join(args.raw_dir, "LCCC-base-split.zip")
        if not os.path.exists(lc):
            sys.exit(f"missing {lc}; run tools/download_data.py --accept-license first")
        dialogues = lccc(lc, args.max_lccc)
        print(f"[lccc] {len(dialogues)} dialogues")

    slim = read_jsonl(os.path.join(args.raw_dir, "LCCC_sharegpt_10K.jsonl"))
    print(f"[slim] {len(slim)} dialogues")
    dialogues += slim

    if args.include_nc:
        mui = read_jsonl(os.path.join(args.raw_dir, "muice_train.jsonl"))
        print(f"[muice] {len(mui)} dialogues (CC-BY-NC-4.0)")
        dialogues += mui

    if not dialogues:
        sys.exit("no dialogues collected")

    random.Random(args.seed).shuffle(dialogues)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.raw_out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in dialogues:
            f.write(json.dumps({"turns": t}, ensure_ascii=False) + "\n")

    seen, size = set(), 0
    with open(args.raw_out, "w", encoding="utf-8") as f:
        for t in dialogues:
            for _, text in t:
                if text in seen:
                    continue
                seen.add(text)
                f.write(text + "\n")
                size += len(text.encode()) + 1

    print(f"[out] {len(dialogues)} dialogues -> {args.out}")
    print(f"[out] {size/2**20:.1f} MiB, {len(seen)} unique lines -> {args.raw_out}")


if __name__ == "__main__":
    main()
