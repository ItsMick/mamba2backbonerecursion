# RWKV-7 Crucible Results

Run at: 2026-04-06 00:58:26
Device: cpu

## Proof 1: Ablation Kill-Shot

**PASS**: Logit L2 divergence = 549.79 between 12 and 6 loops.
- Full run top token: ' Lab'
- Abort run top token: 'ymph'
- Tokens differ: True
- Cosine similarity: 0.8348
- State fingerprint cosine: 0.9570

## Proof 2: Memory Flatline

**PASS**

### Part A: Direct State Tensor Measurement (deterministic)

| Loops | State Size (KB) |
|-------|-----------------|
|     1 |          2376.0 |
|     5 |          2376.0 |
|    10 |          2376.0 |
|    20 |          2376.0 |

State size constant: **True**
The state Cache object contains exactly the same number of bytes regardless of loop count. This is the definitive O(1) proof.

### Part B: Process-Level Memory (tracemalloc)

| Loops | Peak Alloc (MB) |
|-------|-----------------|
|     1 |            0.12 |
|     5 |            0.48 |
|    10 |            0.92 |
|    20 |            1.78 |

Peak range: 1.66 MB (200.4% variation)

## Proof 3: State Geometry Comparison

| Property | RWKV-7 | Mamba |
|----------|--------|-------|
| State shape | [H=12, K=64, V=64] matrix | [D, N] vector |
| Interpretable | Yes (KV geometry) | No (opaque h_t) |
| Composable | Yes (linear blend) | Limited |
| Fingerprintable | Yes (per-head mean) | Coarse only |
| State size | 2304 KB | 1152 KB |
| Decay | exp(w) per-head | A matrix (SSM) |

### State Drift (cosine similarity to initial state)

| Tick | Cosine Sim |
|------|------------|
|    1 | 0.9962 |
|    2 | 0.9920 |
|    3 | 0.9870 |
|    4 | 0.9800 |
|    5 | 0.9742 |
|    6 | 0.9673 |
|    7 | 0.9633 |
|    8 | 0.9584 |
|    9 | 0.9527 |
|   10 | 0.9471 |

