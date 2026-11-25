#!/usr/bin/env python3
"""Train a small decoder-only Transformer for ESP32-S3 deployment.

The data file is either raw UTF-8 bytes (byte-level model, uint8) or a uint16
token stream produced by tools/build_ids.py, depending on the config.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, nh = cfg["d_model"], cfg["n_head"]
        self.nh, self.hd = nh, d // nh
        self.r1 = RMSNorm(d)
        self.r2 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, cfg["mlp_mult"] * d, bias=False)
        self.w2 = nn.Linear(cfg["mlp_mult"] * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.r1(x)
        qkv = self.qkv(h).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        h = self.r2(x)
        return x + self.w2(F.silu(self.w1(h)))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        V, d, L, ctx = cfg["vocab_size"], cfg["d_model"], cfg["n_layer"], cfg["ctx_len"]
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if cfg.get("tie_embeddings", True):
            self.head.weight = self.tok.weight
        self.apply(self._init)
        # GPT-2 style: scale residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- data
def load_data(path, dtype=np.uint8):
    if not os.path.exists(path):
        raise SystemExit(f"data file not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, bs, ctx, rng):
    ix = rng.integers(0, len(data) - ctx - 1, size=bs)
    x = np.stack([np.asarray(data[i:i + ctx], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(data[i + 1:i + 1 + ctx], dtype=np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def estimate_loss(model, data, cfg, rng, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg["train"]["batch_size"], cfg["ctx_len"], rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, cfg):
    t = cfg["train"]
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    prog = (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"])
    prog = min(1.0, prog)
    return t["lr"] * (t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chat-lccc.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    data_path = args.data or cfg["data"]["file"]
    if args.steps:
        cfg["train"]["max_steps"] = args.steps
    if args.batch:
        cfg["train"]["batch_size"] = args.batch

    torch.manual_seed(cfg["train"]["seed"])
    rng = np.random.default_rng(cfg["train"]["seed"])
    torch.set_num_threads(os.cpu_count() or 4)

    ddt = np.uint16 if cfg["data"].get("data_dtype") == "uint16" else np.uint8
    data = load_data(data_path, ddt)
    n_val = min(len(data) // 20, 2_000_000)
    train_data, val_data = data[:-n_val], data[-n_val:]

    model = GPT(cfg)
    print(f"[eslm] {cfg['name']}  params={model.num_params():,}  "
          f"fp32={model.num_params() * 4 / 2**20:.2f}MiB  int8={model.num_params() / 2**20:.2f}MiB")
    unit = "tokens" if ddt is np.uint16 else "bytes"
    print(f"[eslm] data={data_path} ({len(data)/1e6:.2f}M {unit})  "
          f"train={len(train_data)/1e6:.2f}M  val={len(val_data)/1e6:.2f}M")

    start_step, opt_sd, best_init = 0, None, float("inf")
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        start_step = int(sd.get("step", 0))
        opt_sd = sd.get("opt")
        best_init = float(sd.get("val_loss") or float("inf"))
        print(f"[eslm] resumed from {args.resume} at step {start_step} "
              f"(best val {best_init:.4f})")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["train"]["lr"], betas=(0.9, 0.95), eps=1e-8)

    if opt_sd:
        opt.load_state_dict(opt_sd)
        print("[eslm] optimizer state restored")

    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.train()
    t0 = time.time()
    best = best_init

    for step in range(start_step, cfg["train"]["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        x, y = get_batch(train_data, cfg["train"]["batch_size"], cfg["ctx_len"], rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()

        if step % 50 == 0:
            dt = time.time() - t0
            done = step - start_step + 1
            tok = done * cfg["train"]["batch_size"] * cfg["ctx_len"]
            print(f"step {step:5d}/{cfg['train']['max_steps']}  loss {loss.item():.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {tok/dt/1000:.1f}k tok/s  "
                  f"eta {(cfg['train']['max_steps']-step)*dt/done/60:.0f} min", flush=True)

        if step % cfg["train"]["eval_every"] == 0 and step > 0:
            vl = estimate_loss(model, val_data, cfg, rng)
            print(f"  [eval] step {step}  val_loss {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg,
                            "step": step, "val_loss": vl}, os.path.join(out_dir, "best.pt"))

        if step % cfg["train"]["save_every"] == 0 and step > 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "step": step},
                       os.path.join(out_dir, "last.pt"))

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg,
                "step": cfg["train"]["max_steps"]}, os.path.join(out_dir, "last.pt"))
    print(f"[eslm] done in {(time.time()-t0)/60:.1f} min, best val_loss {best:.4f}")


if __name__ == "__main__":
    main()
