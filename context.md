# Parameter Golf Project — Context for AI Assistants

## What is this project?

We are participating in OpenAI's Parameter Golf Challenge (March 18 – April 30, 2026).
The goal: train the best possible language model that fits inside a **16MB compressed artifact**, evaluated by bits-per-byte (BPB) on the FineWeb validation set. Lower BPB = better model.

Repository: https://github.com/openai/parameter-golf

---

## Our Approach

We are NOT just tuning hyperparameters. We are making **architectural changes** to the baseline transformer that nobody on the current leaderboard has tried.

### Core Stack

```
MASA (Matrix Atom Sharing) on Attention
  + SwiGLU activation on MLP
  + int6 QAT (Quantization-Aware Training)
  + zstd compression
```

### Why this combination?

The current leaderboard (best: 1.1428 BPB) stacks tricks on top of a standard transformer:
quantization, tokenizer hacks, optimizer tuning. Nobody has changed the fundamental attention
architecture. MASA does exactly that.

---

## Architecture Details

### Baseline (what we start from)

- 9 layers, 512 model dim, 1024 vocab
- Standard QKV attention: each layer has its own Q, K, V, Output matrices (512×512 each)
- MLP: 512 → 1024 → 512, ReLU² activation, mlp_mult=2
- Tied embeddings (input = output projection)
- GQA: 8 query heads, 4 KV heads
- RoPE positional encoding (no learned parameters)
- Muon optimizer
- Baseline BPB: ~1.2244

### Our Changes

**Change 1: MASA on CausalSelfAttention**

Instead of each layer having unique Q, K, V, Output matrices:
- Define N shared "base" matrices (atoms) — we plan N=6
- Each layer gets tiny mix coefficients (basically free in parameter cost)
- Each layer's QKV = weighted combination of shared bases
- Expected saving: ~66% reduction in attention parameters
- Paper: arxiv MASA 2025 (Matrix Atom Sharing Attention)

**Change 2: SwiGLU on MLP**

Replace ReLU² activation with SwiGLU (used in LLaMA).
- Requires 3 linear projections instead of 2
- Compensate by slightly reducing mlp_mult
- Consistently improves quality at no parameter cost
- Simple swap, low risk

**Change 3: int6 QAT**

Train with simulated int6 quantization in the forward pass.
- Model learns to be robust to low precision during training
- Better than post-training quantization
- Enables more weights to fit in 16MB

---

## File Structure

```
train_gpt.py              ← everything lives here (1500 line limit enforced)
  ├── Hyperparameters     ← config via environment variables
  ├── Rotary              ← RoPE positional encoding (don't touch)
  ├── CausalSelfAttention ← MASA goes here
  ├── MLP                 ← SwiGLU goes here
  ├── Block               ← wraps attention + MLP (don't touch)
  ├── GPT                 ← stacks blocks + embedding (minor wiring changes)
  └── main()              ← training loop + compression (don't touch)
```

---

## Current Status

- [x] Understand baseline architecture
- [x] Colab environment set up (Tesla T4, 15GB VRAM)
- [x] Dataset downloaded (sp1024, 1 shard)
- [ ] Smoke test baseline run
- [ ] Implement SwiGLU
- [ ] Implement MASA
- [ ] Smoke test modified architecture
- [ ] Full training run on RunPod
- [ ] Write up results

---

## Developer Context

- **Who:** Student developer, hackathon background (ThrustGuard, Lockr, Deep Care)
- **Stack comfort:** Python, FastAPI, PyTorch basics
- **Hardware:** Google Colab T4 for iteration, RunPod H100 for final runs
- **Goal:** Learn architecture deeply, produce a blog post / LinkedIn writeup, submit as non-record if needed
- **NOT goal:** Win the leaderboard (though would be nice)

---

## Key Concepts (for context)

- **BPB (bits-per-byte):** evaluation metric. lower = better. current SOTA 1.1428.
- **Parameters/weights:** the numbers the model learns. stored in the 16MB file.
- **QKV matrices:** the fat part of attention. 4 matrices × 9 layers = where MASA saves.
- **Residual connections:** `x = x + layer(x)` pattern throughout. don't break these.
- **Tied embeddings:** input embedding = output projection transposed. already in baseline.
- **GQA:** 8 query heads share 4 KV heads. already in baseline, keep it.

---

## What NOT to change

- Training loop (`main()`)
- Data loading
- Compression / quantization pipeline (post-training)
- RoPE (`Rotary` class)
- `Block` class structure (residual connections)
- Hyperparameters config class

---

## Constraints to respect

- Final artifact ≤ 16,000,000 bytes (code + compressed model)
- No external downloads during evaluation
- Everything self-contained in train_gpt.py
- Non-record track: no 10-minute training limit (we target this track)
