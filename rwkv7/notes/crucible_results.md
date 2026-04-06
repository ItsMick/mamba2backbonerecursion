# RWKV-7 Crucible Results

Run at: 2026-04-06 00:52:14
Device: cpu

## Proof 1: Ablation Kill-Shot

**PASS**: Logit L2 divergence = 549.79 between 12 and 6 loops.
- Full run top token: ' TB'
- Abort run top token: 'дан'
- Tokens differ: True
- Cosine similarity: 0.8348
- State fingerprint cosine: 0.9570

## Proof 2: Memory Flatline

**MARGINAL**: Delta range = 84.07 MB

| Loops | Before (MB) | After (MB) | Delta (MB) |
|-------|-------------|------------|------------|
|     1 |     2365.78 |    2374.70 |       8.93 |
|     5 |     2374.70 |    2374.70 |       0.00 |
|    10 |     2374.70 |    2374.70 |       0.00 |
|    20 |     2374.70 |    2458.77 |      84.07 |

Architecture: O(1) single-token stateful decode
Note: CPU RSS measurement is noisy. Re-run on CUDA for precise VRAM flatline.

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

