#!/usr/bin/env python3
"""Export a trained checkpoint to model.bin (the PC <-> ESP32 contract).

Layout (all little-endian):

  char  magic[4] = "ESLM"
  u32   version = 1
  u32   dtype   (0 = float32, 1 = int8 weights)
  u32   vocab, d_model, n_layer, n_head, ctx_len

followed by tensors in this exact order:

  tok_emb  [vocab, d]
  pos_emb  [ctx,   d]
  per layer i in 0..n_layer-1:
      rms1   fp32 [d]
      wq     [d, d]        rows = output channel
      wk     [d, d]
      wv     [d, d]
      wo     [d, d]
      rms2   fp32 [d]
      w1     [mlp, d]
      w2     [d, mlp]
  rms_f    fp32 [d]

dtype=1: every 2-D matrix is stored as  fp32 scales[out]  then  int8 data[out*in];
         every 1-D vector stays fp32.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GPT  # noqa: E402

MAGIC = b"ESLM"
VERSION = 1


def write_matrix(f, w, int8):
    w = w.detach().float().numpy()
    if not int8:
        f.write(w.astype("<f4").tobytes())
        return w.size * 4
    out, inn = w.shape
    scale = (np.abs(w).max(axis=1) / 127.0).astype("<f4")
    scale = np.where(scale == 0, np.float32(1e-8), scale)
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    f.write(scale.tobytes())
    f.write(q.tobytes())
    pad = (-(out * inn)) % 4            # keep following matrices 4-byte aligned
    if pad:
        f.write(b"\x00" * pad)
    return out * 4 + out * inn + pad


def write_vector(f, v):
    v = v.detach().float().numpy()
    f.write(v.astype("<f4").tobytes())
    return v.size * 4


def export(ckpt_path, out_path, int8, tokenizer_dir=None):
    sd = torch.load(ckpt_path, map_location="cpu")
    cfg = sd["cfg"]
    model = GPT(cfg)
    model.load_state_dict(sd["model"])
    model.eval()

    V, d, L = cfg["vocab_size"], cfg["d_model"], cfg["n_layer"]
    H, ctx = cfg["n_head"], cfg["ctx_len"]
    mlp = cfg["mlp_mult"] * d
    n_params = model.num_params()

    vocab_part = None
    if tokenizer_dir:
        from tokenizers import ByteLevelBPETokenizer
        from bpe_utils import vocab_blob
        tok = ByteLevelBPETokenizer(os.path.join(tokenizer_dir, "vocab.json"),
                                    os.path.join(tokenizer_dir, "merges.txt"))
        tok.add_special_tokens(["<|u|>", "<|a|>", "<|e|>"])
        offsets, blob = vocab_blob(tok)
        blob += b"\x00" * ((4 - len(blob) % 4) % 4)   # keep weights 4-byte aligned
        u, a, e = (tok.token_to_id(t) for t in ("<|u|>", "<|a|>", "<|e|>"))
        vocab_part = (offsets, blob, int(u), int(a), int(e))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        if vocab_part:
            offsets, blob, u, a, e = vocab_part
            f.write(struct.pack("<11I", 2, 1 if int8 else 0, V, d, L, H, ctx, u, a, e, len(blob)))
            f.write(np.asarray(offsets, dtype="<u4").tobytes())
            f.write(blob)
            nbytes = 48 + 4 * (V + 1) + len(blob)
        else:
            f.write(struct.pack("<8I", VERSION, 1 if int8 else 0, V, d, L, H, ctx, 0))
            nbytes = 36
        sd_ = model.state_dict()

        nbytes += write_matrix(f, sd_["tok.weight"], int8)
        nbytes += write_matrix(f, sd_["pos.weight"], int8)
        for i in range(L):
            p = f"blocks.{i}."
            nbytes += write_vector(f, sd_[p + "r1.w"])
            qkv = sd_[p + "qkv.weight"]              # [3d, d]
            for name, sl in (("wq", slice(0, d)), ("wk", slice(d, 2 * d)), ("wv", slice(2 * d, 3 * d))):
                nbytes += write_matrix(f, qkv[sl], int8)
            nbytes += write_matrix(f, sd_[p + "proj.weight"], int8)
            nbytes += write_vector(f, sd_[p + "r2.w"])
            nbytes += write_matrix(f, sd_[p + "w1.weight"], int8)   # [mlp, d]
            nbytes += write_matrix(f, sd_[p + "w2.weight"], int8)   # [d, mlp]
        nbytes += write_vector(f, sd_["norm.w"])

    print(f"[export] params {n_params:,}  dtype={'int8' if int8 else 'f32'}  "
          f"file {nbytes/2**20:.2f} MiB -> {out_path}")
    return cfg, n_params, nbytes


def check(cfg, out_path, ckpt_path, int8):
    """Re-read model.bin and compare against the PyTorch weights."""
    sd = torch.load(ckpt_path, map_location="cpu")
    model = GPT(sd["cfg"])
    model.load_state_dict(sd["model"])
    ref = model.state_dict()

    V, d, L = cfg["vocab_size"], cfg["d_model"], cfg["n_layer"]
    ctx, mlp = cfg["ctx_len"], cfg["mlp_mult"] * cfg["d_model"]

    with open(out_path, "rb") as f:
        assert f.read(4) == MAGIC
        ver = struct.unpack("<I", f.read(4))[0]
        if ver == 2:
            vals = struct.unpack("<10I", f.read(40))
            dt = vals[0]
            v_, d_, l_, h_, c_ = vals[1:6]
            blob_len = vals[9]
            f.seek(4 * (V + 1) + blob_len, 1)
        else:
            rest = struct.unpack("<7I", f.read(28))
            dt = rest[0]
            v_, d_, l_, h_, c_ = rest[1:6]
        assert (v_, d_, l_, h_, c_) == (V, d, L, cfg["n_head"], ctx), "layout mismatch"
        assert dt == (1 if int8 else 0), "dtype flag mismatch"

        def rd_mat(out, inn):
            if not int8:
                a = np.frombuffer(f.read(out * inn * 4), dtype="<f4").reshape(out, inn)
                return torch.from_numpy(a.copy())
            scale = np.frombuffer(f.read(out * 4), dtype="<f4")
            q = np.frombuffer(f.read(out * inn), dtype=np.int8).reshape(out, inn).astype(np.float32)
            pad = (-(out * inn)) % 4
            if pad:
                f.seek(pad, 1)
            return torch.from_numpy(q * scale[:, None])

        def rd_vec(n):
            a = np.frombuffer(f.read(n * 4), dtype="<f4")
            return torch.from_numpy(a.copy())

        worst = 0.0

        def cmp(name, got, exp):
            nonlocal worst
            diff = (got - exp.float()).abs().max().item()
            worst = max(worst, diff)
            if diff > (1e-2 if int8 else 1e-3):
                print(f"  MISMATCH {name}: max|diff|={diff:.3e}")

        cmp("tok_emb", rd_mat(V, d), ref["tok.weight"])
        cmp("pos_emb", rd_mat(ctx, d), ref["pos.weight"])
        for i in range(L):
            p = f"blocks.{i}."
            cmp(p + "rms1", rd_vec(d), ref[p + "r1.w"])
            qkv = ref[p + "qkv.weight"]
            for j, name in enumerate(("wq", "wk", "wv")):
                cmp(p + name, rd_mat(d, d), qkv[j * d:(j + 1) * d])
            cmp(p + "wo", rd_mat(d, d), ref[p + "proj.weight"])
            cmp(p + "rms2", rd_vec(d), ref[p + "r2.w"])
            cmp(p + "w1", rd_mat(mlp, d), ref[p + "w1.weight"])
            cmp(p + "w2", rd_mat(d, mlp), ref[p + "w2.weight"])
        cmp("rms_f", rd_vec(d), ref["norm.w"])

    ok = worst < (1e-2 if int8 else 1e-3)
    print(f"[check] dtype={'int8' if int8 else 'f32'}  max|diff| = {worst:.3e}  "
          f"({'OK' if ok else 'FAIL'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoints/best.pt")
    ap.add_argument("--out", default="models/model.bin")
    ap.add_argument("--dtype", choices=["f32", "int8"], default="f32")
    ap.add_argument("--tokenizer", default=None, help="models/tokenizer -> write v2 with vocab blob")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    int8 = args.dtype == "int8"
    cfg, _, _ = export(args.ckpt, args.out, int8, args.tokenizer)
    json.dump(cfg, open(os.path.join(os.path.dirname(args.out) or ".", "model_cfg.json"), "w"), indent=2)
    if args.check:
        check(cfg, args.out, args.ckpt, int8)


if __name__ == "__main__":
    main()
