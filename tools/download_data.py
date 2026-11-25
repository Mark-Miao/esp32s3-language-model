#!/usr/bin/env python3
"""Download the dialogue corpora used by the default training pipeline.

No data is bundled with this repository. Files are fetched from ModelScope
into data/raw/ after you accept the license notices below.

  LCCC-base-split.zip   LCCC Custom license (research use)
  LCCC_sharegpt_10K     Apache-2.0, derived from LCCC
  train.jsonl (Muice)   CC-BY-NC-4.0, NON-COMMERCIAL, opt-in via --include-nc

You are responsible for complying with each dataset's terms. Do not
redistribute the downloaded files.

    python tools/download_data.py --accept-license [--include-nc]
"""
import argparse
import hashlib
import os
import sys
import urllib.request

API = "https://www.modelscope.cn/api/v1/datasets/{ds}/repo?Revision=master&FilePath={path}"

SOURCES = {
    "lccc": {
        "url": API.format(ds="OmniData/LCCC", path="raw/LCCC-base-split.zip"),
        "file": "LCCC-base-split.zip",
        "min_bytes": 300 * 2**20,
        "license": "LCCC Custom license (research use only)",
        "opt_in": False,
    },
    "slim": {
        "url": API.format(ds="AI-ModelScope/Slim-LCCC-zh",
                          path="LCCC_sharegpt_10K.jsonl"),
        "file": "LCCC_sharegpt_10K.jsonl",
        "min_bytes": 2_000_000,
        "license": "Apache-2.0 (derived from LCCC)",
        "opt_in": False,
    },
    "muice": {
        "url": API.format(ds="Moemuu/Muice-Dataset", path="train.jsonl"),
        "file": "muice_train.jsonl",
        "min_bytes": 500_000,
        "license": "CC-BY-NC-4.0 (NON-COMMERCIAL)",
        "opt_in": True,
    },
}

NOTICE = """\
This script downloads third-party datasets. Their licenses are NOT the AGPL
license that covers this repository:

  LCCC          research use only (see the CDial-GPT / LCCC terms)
  Slim-LCCC     Apache-2.0, derived from LCCC
  Muice         CC-BY-NC-4.0, non-commercial, opt-in

The downloads may contain user-generated text, including personal
information. Do not redistribute them, and do not publish models trained on
them without reviewing the source licenses and applicable privacy law.
"""


def fetch(url, dst, min_bytes):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "esp32s3-language-model"})
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            size += len(chunk)
            if total:
                pct = size * 100 // total
                sys.stdout.write(f"\r  {size/2**20:8.1f}/{total/2**20:.1f} MiB ({pct:3d}%)")
                sys.stdout.flush()
    print()
    if size < min_bytes:
        os.unlink(tmp)
        raise RuntimeError(f"download too small ({size} bytes); expected >= {min_bytes}")
    os.replace(tmp, dst)
    print(f"  sha256 {h.hexdigest()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--accept-license", action="store_true",
                    help="required: confirm you will comply with the dataset licenses")
    ap.add_argument("--include-nc", action="store_true",
                    help="also download the CC-BY-NC-4.0 dataset (non-commercial only)")
    args = ap.parse_args()

    if not args.accept_license:
        print(NOTICE)
        sys.exit("Refusing to download without --accept-license")

    print(NOTICE)
    for key, src in SOURCES.items():
        if src["opt_in"] and not args.include_nc:
            print(f"[skip] {key}: {src['license']} (use --include-nc to opt in)")
            continue
        dst = os.path.join(args.out_dir, src["file"])
        print(f"[get ] {key}: {src['license']}")
        try:
            fetch(src["url"], dst, src["min_bytes"])
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {key}: {e}")
    print(f"\nDone. Files are in {args.out_dir}/ (not tracked by git).")


if __name__ == "__main__":
    main()
