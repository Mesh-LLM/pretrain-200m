#!/usr/bin/env python3
"""
Pretrain a 200M parameter transformer from scratch on Common Crawl + goose session data.
Memory-efficient: streams tokenization, doesn't load everything at once.
"""
import os
import sys
import math
import time
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Config
# ============================================================

CONFIG = {
    # Model (200M params)
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "head_dim": 64,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "tie_embeddings": True,
    "intermediate_size": 2048,

    # Training
    "batch_size": 8,
    "grad_accum_steps": 64,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_iters": 2000,
    "max_iters": 100000,
    "lr_decay": "cosine",
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "bf16": True,

    # Data
    "goose_data_path": "./goose_all_combined.txt",
    "fineweb_cache": "./hf_cache",
    "train_bin": "./train.bin",
    "val_bin": "./val.bin",
    "val_tokens": 10_000_000,

    # Checkpointing
    "checkpoint_dir": "./checkpoints",
    "checkpoint_interval": 2000,
    "log_interval": 25,

    # Data mix
    "goose_upsample": 5,
    "target_total_tokens": 2_000_000_000,  # 2B tokens (faster than 4B)
}

# ============================================================
# Model (Llama-style)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
    def forward(self, x):
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rope = RoPE(head_dim, max_seq_len)
    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, d_model, bias=False)
        self.w3 = nn.Linear(d_model, intermediate_size, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, intermediate_size, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, head_dim, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, intermediate_size)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = nn.ModuleList([
            Block(cfg["d_model"], cfg["n_heads"], cfg["head_dim"],
                  cfg["intermediate_size"], cfg["max_seq_len"])
            for _ in range(cfg["n_layers"])
        ])
        self.norm = RMSNorm(cfg["d_model"])
        if cfg["tie_embeddings"]:
            self.lm_head = lambda x: x @ self.tok_emb.weight.T
        else:
            self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        return logits, None

# ============================================================
# Memory-efficient tokenization
# ============================================================

def tokenize_data(goose_text_path, output_train, output_val, val_tokens=10_000_000):
    """Stream-tokenize goose data + FineWeb-Edu, save as .bin files."""
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    target = CONFIG["target_total_tokens"]
    val_count = val_tokens
    train_count = target - val_count

    print(f"Target: {target:,} total tokens ({train_count/1e9:.1f}B train + {val_count/1e6:.0f}M val)")

    # Open output files
    train_file = open(output_train, 'wb')
    val_file = open(output_val, 'wb')

    total_tokens = 0
    written_train = 0
    written_val = 0

    # Step 1: Tokenize goose data in chunks and write (upsampled)
    print("Step 1: Tokenizing goose data...")
    goose_chunk_size = 1_000_000  # 1MB chunks
    goose_written = 0
    for rep in range(CONFIG["goose_upsample"]):
        with open(goose_text_path, 'r') as f:
            while True:
                chunk = f.read(goose_chunk_size)
                if not chunk:
                    break
                tokens = tokenizer.encode(chunk)
                arr = np.array(tokens, dtype=np.uint16)
                if written_val < val_count:
                    to_val = min(len(arr), val_count - written_val)
                    val_file.write(arr[:to_val].tobytes())
                    written_val += to_val
                    arr = arr[to_val:]
                if len(arr) > 0:
                    train_file.write(arr.tobytes())
                    written_train += len(arr)
                    goose_written += len(arr)
        print(f"  Rep {rep+1}/{CONFIG['goose_upsample']}: {goose_written:,} goose tokens written")
    print(f"  Total goose tokens: {goose_written:,}")

    # Step 2: Stream FineWeb-Edu and tokenize
    print("Step 2: Downloading and tokenizing FineWeb-Edu...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)

    cc_written = 0
    for i, example in enumerate(ds):
        tokens = tokenizer.encode(example["text"])
        arr = np.array(tokens, dtype=np.uint16)

        if written_val < val_count:
            to_val = min(len(arr), val_count - written_val)
            val_file.write(arr[:to_val].tobytes())
            written_val += to_val
            arr = arr[to_val:]

        if len(arr) > 0:
            train_file.write(arr.tobytes())
            written_train += len(arr)
            cc_written += len(arr)

        total_tokens = written_train + written_val

        if (i + 1) % 5000 == 0:
            print(f"  FineWeb-Edu: {i+1} docs, CC tokens: {cc_written:,} "
                  f"({cc_written/1e6:.0f}M), total: {total_tokens/1e6:.0f}M / {target/1e6:.0f}M")

        if written_train >= train_count:
            break

    train_file.close()
    val_file.close()

    print(f"\nDone!")
    print(f"  Train: {written_train:,} tokens ({written_train/1e9:.2f}B)")
    print(f"  Val: {written_val:,} tokens ({written_val/1e6:.0f}M)")
    print(f"  Goose portion: {goose_written/written_train*100:.1f}% of train")
    print(f"  Saved: {output_train} ({os.path.getsize(output_train)/1e9:.2f} GB)")
    print(f"  Saved: {output_val} ({os.path.getsize(output_val)/1e6:.0f} MB)")

# ============================================================
# Data
# ============================================================

class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, n_samples):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_samples = n_samples
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data) - self.seq_len - 1)
        chunk = self.data[i:i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

# ============================================================
# Training
# ============================================================

def get_lr(it):
    if it < CONFIG["warmup_iters"]:
        return CONFIG["learning_rate"] * it / CONFIG["warmup_iters"]
    if it >= CONFIG["max_iters"]:
        return CONFIG["min_lr"]
    decay_ratio = (it - CONFIG["warmup_iters"]) / (CONFIG["max_iters"] - CONFIG["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"bf16: {CONFIG['bf16']}")

    if not os.path.exists(CONFIG["train_bin"]):
        print("No tokenized data found. Running tokenization...")
        tokenize_data(CONFIG["goose_data_path"], CONFIG["train_bin"], CONFIG["val_bin"], CONFIG["val_tokens"])

    train_data = np.memmap(CONFIG["train_bin"], dtype=np.uint16, mode='r')
    val_data = np.memmap(CONFIG["val_bin"], dtype=np.uint16, mode='r')
    n_train_samples = len(train_data) // CONFIG["max_seq_len"]
    n_val_samples = len(val_data) // CONFIG["max_seq_len"]

    train_ds = TokenDataset(CONFIG["train_bin"], CONFIG["max_seq_len"], n_train_samples)
    val_ds = TokenDataset(CONFIG["val_bin"], CONFIG["max_seq_len"], min(n_val_samples, 1000))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    model = Transformer(CONFIG).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                      weight_decay=CONFIG["weight_decay"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device, enabled=CONFIG["bf16"])

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    model.train()
    best_val_loss = float('inf')
    iter_num = 0
    t0 = time.time()

    print(f"\nTraining: {CONFIG['max_iters']} iters, batch={CONFIG['batch_size']}, "
          f"grad_accum={CONFIG['grad_accum_steps']}")

    train_iter = iter(train_loader)
    while iter_num < CONFIG["max_iters"]:
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(CONFIG["grad_accum_steps"]):
            try: x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=CONFIG["bf16"]):
                _, loss = model(x, y)
            loss = loss / CONFIG["grad_accum_steps"]
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups: pg["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

        if iter_num % CONFIG["log_interval"] == 0:
            elapsed = time.time() - t0
            tok_per_sec = (iter_num + 1) * CONFIG["batch_size"] * CONFIG["grad_accum_steps"] * CONFIG["max_seq_len"] / max(elapsed, 1)
            print(f"iter {iter_num:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s | {elapsed/60:.1f} min")

        if iter_num % 500 == 0 or iter_num == CONFIG["max_iters"] - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=CONFIG["bf16"]):
                        _, loss = model(x, y)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            print(f"  val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'iter': iter_num, 'val_loss': val_loss, 'config': CONFIG},
                           os.path.join(CONFIG["checkpoint_dir"], "best.pt"))
            model.train()

        if iter_num > 0 and iter_num % CONFIG["checkpoint_interval"] == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'iter': iter_num, 'config': CONFIG},
                       os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_{iter_num}.pt"))
        iter_num += 1

    torch.save({'model': model.state_dict(), 'config': CONFIG},
               os.path.join(CONFIG["checkpoint_dir"], "final.pt"))
    print(f"\nTraining complete! best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    if "--tokenize" in sys.argv:
        tokenize_data(CONFIG["goose_data_path"], CONFIG["train_bin"], CONFIG["val_bin"], CONFIG["val_tokens"])
    else:
        train()
