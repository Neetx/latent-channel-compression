#!/usr/bin/env python3
"""Disk-safe feasibility probe for sequential_scaled on a 16 GB GPU.

The pipeline loads one agent at a time, so the VRAM ceiling is the largest single
agent. This probe downloads ONLY that agent (Sequential-Scaled-Planner-Gemma3-4B,
~8.7 GB — fits a tight disk) instead of the full ~24 GB tier, measures its bf16
weight footprint on-GPU, runs a tiny generation to confirm it executes on Blackwell,
then analytically projects total VRAM (weights + KV cache) at max_new_tokens=4000 for
each batch size — telling us whether scaled is locally runnable and at what batch.
"""
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B"
MAXTOK = 4000
GB = 1e9
BUDGET = 15.5  # leave headroom under 16 GB

print(f"[1/4] download {REPO} (largest scaled agent only) ...", flush=True)
path = snapshot_download(repo_id=REPO, repo_type="model")

print("[2/4] load bf16 on cuda ...", flush=True)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.to("cuda").eval()
torch.cuda.synchronize()
weight_gb = torch.cuda.memory_allocated() / GB
print(f"     weights on GPU: {weight_gb:.2f} GB")

print("[3/4] tiny generation sanity (batch=2, 32 tok) ...", flush=True)
enc = tok(["def add(a, b):", "Write a function to"], return_tensors="pt", padding=True).to("cuda")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=32, do_sample=False)
print(f"     OK generated shape={tuple(out.shape)}")

print("[4/4] project VRAM at max_new_tokens=4000 per batch ...", flush=True)
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)


def cget(name):
    """Read an attr from the config, falling back to .text_config (Gemma3 etc.)."""
    v = getattr(cfg, name, None)
    if v is None and hasattr(cfg, "text_config"):
        v = getattr(cfg.text_config, name, None)
    return v


n_layers = cget("num_hidden_layers")
n_kv = cget("num_key_value_heads") or cget("num_attention_heads")
hidden = cget("hidden_size")
n_heads = cget("num_attention_heads")
head_dim = cget("head_dim") or (hidden // n_heads if hidden and n_heads else None)
vocab = cget("vocab_size")
# KV bytes per (token, batch): 2 (K,V) * layers * kv_heads * head_dim * 2 bytes (bf16)
kv_per_tok = 2 * n_layers * n_kv * head_dim * 2
print(f"     config: layers={n_layers} kv_heads={n_kv} head_dim={head_dim} vocab={vocab}")
print(f"     KV/token = {kv_per_tok/1e6:.2f} MB   (weights {weight_gb:.2f} GB, ~1 GB activations assumed)")
print()
print(f"     {'batch':>5} | {'KV@4000':>9} | {'total':>7} | fits<{BUDGET}GB ?")
for B in (1, 2, 4, 8, 16):
    kv = kv_per_tok * MAXTOK * B / GB
    total = weight_gb + kv + 1.0
    print(f"     {B:>5} | {kv:>7.2f}GB | {total:>5.1f}GB | {'YES' if total < BUDGET else 'NO (OOM risk)'}")
print()
print("NOTE: sampled ladder has no output_scores; greedy CAPTURE adds "
      f"~vocab*batch*4000*4 bytes (~{vocab*MAXTOK*4/GB:.1f} GB at batch=1) on top → capture needs an even smaller batch.")
